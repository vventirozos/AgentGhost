#!/usr/bin/env python3
"""Progress monitor for the Track-B4 ablation harness (ablation_trackb4.py).

Shows, for a running pilot or full run:  task x/y within the current arm/pass,
which arm of which repeat, an overall % bar, the driver-turn rate, and a rough
ETA. It reads only files the harness already writes — the driver stdout log and
the per-arm agent logs — so it never touches the run.

USAGE
    # one-shot
    python scripts/ablation_monitor.py ablation_out/b4-pilot-20260715
    # live, refreshing every 20s
    python scripts/ablation_monitor.py ablation_out/b4-20260715 --watch 20
    # auto-detect the newest b4-* run under ablation_out/
    python scripts/ablation_monitor.py

The report-dir is the `--report-dir` you passed to ablation_trackb4.py. The
driver stdout log is expected at `<report-dir>.log` (the redirect convention);
override with --driver-log. Progress is counted from DRIVER-tagged turns (the
`dv` request-id prefix), so interleaved self-play turns are never miscounted.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Harness constants (kept in sync with ablation_trackb4.py / trackb4_tasks.py).
ARM_ORDER = ("treatment", "treatment_uniform", "control")
DEFAULT_BATTERY = 35            # candidate pool size (load_b4_battery)
DEFAULT_SEEDING = 8            # load_b4_seeding
DEFAULT_IDLE_EPOCHS = 8
DEFAULT_EPOCH_SLEEP = 70.0
_DRIVER_TAG = "DV"

_ANSI = re.compile(r"\033\[[0-9;]*m")
_END = re.compile(r"└─\s+" + _DRIVER_TAG + r"\s+request finished", re.IGNORECASE)
_REPEAT = re.compile(r"=== B4 repeat (\d+)/(\d+) ===")
_PILOT_PASS = re.compile(r"=== pilot pass (\d+)/(\d+) ===")
_READY = re.compile(r"(?:ready|NOT READY): (\w+) on ")
_BATTERY = re.compile(r"battery: (\d+)/(\d+) tasks")


def count_driver_turns(agent_log: Path) -> int:
    """Completed DRIVER turns in an agent log (ANSI-stripped DV END frames)."""
    try:
        text = _ANSI.sub("", agent_log.read_text(errors="replace"))
    except OSError:
        return 0
    return len(_END.findall(text))


@dataclass
class RunConfig:
    mode: str            # "pilot" | "full" | "unknown"
    repeats: int         # pilot passes, or full repeats
    arms_per_repeat: int  # 1 for pilot, 2 or 3 for full
    seeding: int
    battery: int

    @property
    def turns_per_arm(self) -> int:
        # A pilot "arm" (one pass) is battery-only; a full arm is seed + battery.
        return (self.seeding + self.battery) if self.mode == "full" else self.battery

    @property
    def total_turns(self) -> int:
        return self.repeats * self.arms_per_repeat * self.turns_per_arm


def detect_config(driver_text: str, battery_override: int | None) -> RunConfig:
    """Infer mode + totals from the driver stdout log."""
    passes = _PILOT_PASS.findall(driver_text)
    repeats_full = _REPEAT.findall(driver_text)
    bat = _BATTERY.findall(driver_text)
    battery = (battery_override if battery_override
               else int(bat[-1][0]) if bat else DEFAULT_BATTERY)
    if passes:
        total = max(int(p[1]) for p in passes)
        return RunConfig("pilot", total, 1, DEFAULT_SEEDING, battery)
    if repeats_full:
        total = max(int(r[1]) for r in repeats_full)
        # arms_per_repeat defaults to 3. It is only 2 under --no-frontier-arm,
        # which drops the treatment_uniform arm — and since the arm order is
        # treatment → treatment_uniform → control, seeing `control` boot while
        # `treatment_uniform` never has is the unambiguous signal. Early on
        # (before control boots) we correctly stay at 3.
        arms_seen = set(_READY.findall(driver_text))
        apr = 2 if ("control" in arms_seen
                    and "treatment_uniform" not in arms_seen) else 3
        return RunConfig("full", total, apr, DEFAULT_SEEDING, battery)
    return RunConfig("unknown", 1, 1, DEFAULT_SEEDING, battery)


@dataclass
class Progress:
    cfg: RunConfig
    done: int
    total: int
    repeat_now: int
    arm_now: str
    arm_idx: int          # 1-based within the repeat
    turns_in_arm: int
    phase: str
    elapsed_s: float


def compute_progress(cfg: RunConfig, driver_text: str, report_dir: Path,
                     elapsed_s: float, idle_stall: bool) -> Progress:
    if cfg.mode == "pilot":
        turns = count_driver_turns(report_dir / "pilot.log")
        passes = _PILOT_PASS.findall(driver_text)
        pass_now = int(passes[-1][0]) if passes else 1
        in_arm = turns - (pass_now - 1) * cfg.battery
        in_arm = max(0, min(in_arm, cfg.battery))
        return Progress(cfg, turns, cfg.total_turns, pass_now, "pilot",
                        pass_now, in_arm, "probe", elapsed_s)

    # full run
    reps = _REPEAT.findall(driver_text)
    repeat_now = int(reps[-1][0]) if reps else 1
    readies = _READY.findall(driver_text)
    arm_now = readies[-1] if readies else "?"
    arms_started = len(readies)                 # includes the current arm
    arms_done = max(0, arms_started - 1)
    cur = count_driver_turns(report_dir / f"{arm_now}.log")
    done = arms_done * cfg.turns_per_arm + cur
    arm_idx = ((arms_started - 1) % cfg.arms_per_repeat) + 1 if arms_started else 1
    if cur < cfg.seeding:
        phase = "seed"
    elif cur == cfg.seeding:
        phase = "idle" if idle_stall else "seed→probe"
    else:
        phase = "probe"
    return Progress(cfg, done, cfg.total_turns, repeat_now, arm_now, arm_idx,
                    cur, phase, elapsed_s)


def _fmt_dur(s: float) -> str:
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _bar(frac: float, width: int = 28) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def format_status(p: Progress) -> str:
    cfg = p.cfg
    frac = (p.done / p.total) if p.total else 0.0
    rate = (p.done / p.elapsed_s) if p.elapsed_s > 0 and p.done else 0.0  # turns/s
    remaining = max(0, p.total - p.done)
    eta = (remaining / rate) if rate > 0 else None
    per_arm = cfg.turns_per_arm

    lines = []
    if cfg.mode == "pilot":
        lines.append(f"B4 PILOT · pass {p.repeat_now}/{cfg.repeats}  "
                     f"(battery {cfg.battery} tasks × {cfg.repeats} passes)")
        lines.append(f"  task {p.turns_in_arm}/{cfg.battery} in this pass · "
                     f"phase {p.phase}")
    elif cfg.mode == "full":
        lines.append(f"B4 FULL · repeat {p.repeat_now}/{cfg.repeats} · "
                     f"arm {p.arm_now} ({p.arm_idx}/{cfg.arms_per_repeat}) · "
                     f"phase {p.phase}")
        lines.append(f"  task {p.turns_in_arm}/{per_arm} in this arm "
                     f"(seed {cfg.seeding} + probe {cfg.battery})")
    else:
        lines.append("B4 run — mode not yet detectable (still booting?)")

    lines.append(f"  overall {_bar(frac)} {frac*100:5.1f}%  "
                 f"{p.done}/{p.total} driver turns")
    rate_min = rate * 60
    eta_txt = _fmt_dur(eta) if eta is not None else "—"
    lines.append(f"  elapsed {_fmt_dur(p.elapsed_s)} · "
                 f"{rate_min:.2f} turns/min · ETA ~{eta_txt}"
                 + ("  (idle window — ETA paused)" if p.phase == "idle" else ""))
    return "\n".join(lines)


def _run_start(driver_log: Path, report_dir: Path) -> float:
    """Best-effort wall-clock start of the run (for elapsed)."""
    for cand in (driver_log, report_dir):
        try:
            st = os.stat(cand)
            return getattr(st, "st_birthtime", st.st_ctime)
        except OSError:
            continue
    return time.time()


def _autodetect_report_dir() -> Path | None:
    here = Path(__file__).resolve().parent.parent
    runs = sorted(glob.glob(str(here / "ablation_out" / "b4-*")),
                  key=lambda p: os.path.getmtime(p), reverse=True)
    runs = [r for r in runs if os.path.isdir(r)]
    return Path(runs[0]) if runs else None


def _snapshot(report_dir: Path, driver_log: Path, battery_override: int | None,
              idle_gap: float) -> Progress:
    try:
        driver_text = driver_log.read_text(errors="replace")
    except OSError:
        driver_text = ""
    cfg = detect_config(driver_text, battery_override)
    elapsed = time.time() - _run_start(driver_log, report_dir)
    # Idle-window heuristic: the current arm log has gone quiet mid-arm.
    idle_stall = False
    if cfg.mode == "full":
        m = _READY.findall(driver_text)
        if m:
            alog = report_dir / f"{m[-1]}.log"
            try:
                idle_stall = (time.time() - os.path.getmtime(alog)) > idle_gap
            except OSError:
                idle_stall = False
    return compute_progress(cfg, driver_text, report_dir, elapsed, idle_stall)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report_dir", nargs="?", default=None,
                    help="the --report-dir passed to ablation_trackb4.py "
                         "(default: newest ablation_out/b4-*)")
    ap.add_argument("--driver-log", default=None,
                    help="driver stdout log (default: <report-dir>.log)")
    ap.add_argument("--battery", type=int, default=None,
                    help="override detected battery size")
    ap.add_argument("--watch", type=float, nargs="?", const=20.0, default=None,
                    metavar="SECS", help="refresh every SECS (default 20)")
    ap.add_argument("--idle-gap", type=float,
                    default=DEFAULT_EPOCH_SLEEP * 1.5,
                    help="seconds of arm-log silence that reads as an idle window")
    args = ap.parse_args()

    report_dir = Path(args.report_dir) if args.report_dir else _autodetect_report_dir()
    if report_dir is None:
        print("No report-dir given and none found under ablation_out/b4-*.",
              file=sys.stderr)
        return 2
    driver_log = (Path(args.driver_log) if args.driver_log
                  else report_dir.with_suffix(report_dir.suffix + ".log")
                  if report_dir.suffix else Path(str(report_dir) + ".log"))

    def once() -> Progress:
        p = _snapshot(report_dir, driver_log, args.battery, args.idle_gap)
        return p

    if args.watch is None:
        print(format_status(once()))
        return 0

    try:
        while True:
            p = once()
            os.system("clear")
            print(f"── {report_dir.name} ── {time.strftime('%H:%M:%S')} "
                  f"(refresh {args.watch:.0f}s, Ctrl-C to stop)\n")
            print(format_status(p))
            # Stop watching once every turn is accounted for.
            if p.total and p.done >= p.total:
                print("\n✓ all driver turns complete — run finishing "
                      "(report being written).")
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
