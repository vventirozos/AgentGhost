"""The handheld's own state: battery, link, backlight, volume, panel, face rate.

Qt-free on purpose. The client is a frameless always-on-top kiosk — while it
runs there is no way to reach the desktop's own brightness, volume or network
indicators — so everything the operator would otherwise read off a system
tray has to be read (and set) from inside it. The reading and the deciding
live here, where they can be executed in tests; ``client.py`` only draws.
"""

from __future__ import annotations

import math
import os
import re

# ── battery ─────────────────────────────────────────────────────────────────

POWER_ROOT = "/sys/class/power_supply"
LOW_BATTERY_PCT = 15


def _read(path: str):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_battery(root: str = POWER_ROOT):
    """``(percent | None, state)`` — state is charging / discharging / full /
    unknown. The uConsole's gauge is ``axp20x-battery``; any supply whose name
    says battery (or axp) and that reports a capacity is accepted, so the same
    code reads a laptop when the client is run on one."""
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return None, "unknown"
    for name in names:
        low = name.lower()
        if "bat" not in low and "axp" not in low:
            continue
        raw = _read(os.path.join(root, name, "capacity"))
        if raw is None:
            continue
        # A PERIPHERAL's battery (a paired headset, a mouse) is listed here
        # too, with scope "Device"; it is not what powers this machine.
        if (_read(os.path.join(root, name, "scope")) or "").lower() == "device":
            continue
        try:
            pct = max(0, min(100, int(float(raw))))
        except (ValueError, OverflowError):          # "n/a", "inf"
            continue
        status = (_read(os.path.join(root, name, "status")) or "").lower()
        if status.startswith("charg"):
            state = "charging"
        elif status.startswith("discharg"):
            state = "discharging"
        elif status in ("full", "not charging"):
            state = "full"
        else:
            state = "unknown"
        return pct, state
    return None, "unknown"


def on_battery(state: str) -> bool:
    return state == "discharging"


# ── wifi ────────────────────────────────────────────────────────────────────

WIRELESS_PATH = "/proc/net/wireless"
_WIFI_LINE_RE = re.compile(r"^\s*\w+:\s+\S+\s+(-?\d+)\.?")


def read_wifi(path: str = WIRELESS_PATH):
    """Link quality as a percentage, or None with no wireless link.

    ``/proc/net/wireless`` reports quality out of 70 — a file read, where
    ``nmcli`` would be a subprocess every few seconds.
    """
    text = _read(path)
    if not text:
        return None
    for line in text.splitlines()[2:]:
        m = _WIFI_LINE_RE.match(line)
        if m:
            return max(0, min(100, round(int(m.group(1)) * 100 / 70)))
    return None


def wifi_bars(pct) -> str:
    if pct is None:
        return ""
    bars = "▂▄▆█"
    n = 1 + min(3, max(0, int(pct) // 25)) if pct > 0 else 0
    return bars[:n] or "▁"


# ── the status readout ──────────────────────────────────────────────────────

def status_html(agent_ok, wifi_pct, bat_pct, bat_state, clock: str,
                ok: str, danger: str, dim: str) -> str:
    """The bottom-right readout: ``● ▂▄▆  ⚡ 100%  10:30 AM``.

    * the dot is the AGENT: green when it answered the last poll, red when it
      did not, dim before the first poll — the operator used to learn the
      agent was unreachable only by sending a message into a ConnectError;
    * the bolt appears only while CHARGING (it used to be printed always, so
      it carried no information), and the percentage turns red under
      ``LOW_BATTERY_PCT`` on battery.
    """
    dot_color = dim if agent_ok is None else (ok if agent_ok else danger)
    parts = [f"<span style='color:{dot_color};'>●</span>"]
    bars = wifi_bars(wifi_pct)
    if bars:
        parts.append(bars)
    if bat_pct is None:
        parts.append("--%")
    else:
        text = f"{'⚡ ' if bat_state == 'charging' else ''}{bat_pct}%"
        if on_battery(bat_state) and bat_pct < LOW_BATTERY_PCT:
            text = f"<span style='color:{danger};'>{text}</span>"
        parts.append(text)
    parts.append(clock)
    return "&nbsp;&nbsp;&nbsp;".join(parts)


# ── backlight ───────────────────────────────────────────────────────────────

BACKLIGHT_ROOT = "/sys/class/backlight"


def backlight_dir(root: str = BACKLIGHT_ROOT):
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return None
    for name in names:
        d = os.path.join(root, name)
        if os.path.exists(os.path.join(d, "brightness")):
            return d
    return None


def read_backlight(root: str = BACKLIGHT_ROOT):
    """``(current, maximum)`` or None when there is no backlight to drive."""
    d = backlight_dir(root)
    if d is None:
        return None
    try:
        return (int(_read(os.path.join(d, "brightness"))),
                int(_read(os.path.join(d, "max_brightness"))))
    except (TypeError, ValueError):
        return None


_SIGNED_RE = re.compile(r"[+-]\d+\Z")


def step_level(arg: str, current: int, maximum: int, floor: int = 0,
               step: int = 1):
    """Resolve ``""`` / ``+`` / ``-`` / ``+N`` / ``-N`` / ``N`` against a
    level. Returns the new level, ``current`` for an empty argument (a
    query), or None for nonsense.
    """
    arg = (arg or "").strip()
    if not arg:
        return current
    if arg in ("+", "up"):
        target = current + step
    elif arg in ("-", "down"):
        target = current - step
    elif _SIGNED_RE.match(arg):
        # A SIGNED number is a change, not a level: `/vol -10` turned the
        # volume to zero when it was read as "set it to minus ten".
        target = current + int(arg)
    else:
        try:
            target = int(arg)
        except ValueError:
            return None
    return max(floor, min(maximum, target))


def set_backlight(arg: str, root: str = BACKLIGHT_ROOT):
    """Apply a ``/bright`` argument. Returns ``(level, maximum)`` or None.

    The floor is ONE, never zero: at zero the panel is black and the command
    that would turn it back up has to be typed blind.
    """
    d = backlight_dir(root)
    cur = read_backlight(root)
    if d is None or cur is None:
        return None
    level = step_level(arg, cur[0], cur[1], floor=1)
    if level is None:
        return None
    if level != cur[0]:
        try:
            with open(os.path.join(d, "brightness"), "w") as fh:
                fh.write(str(level))
        except OSError:
            return None
    return level, cur[1]


# ── volume (PipeWire) ───────────────────────────────────────────────────────

_VOLUME_RE = re.compile(r"Volume:\s*(\d+(?:[.,]\d+)?)")
SINK = "@DEFAULT_AUDIO_SINK@"


def parse_volume(text: str):
    """``wpctl get-volume`` output → percent, or None."""
    m = _VOLUME_RE.search(text or "")
    if not m:
        return None
    try:
        # A comma decimal: this project has met locales that print one.
        return round(float(m.group(1).replace(",", ".")) * 100)
    except ValueError:
        return None


def volume_muted(text: str) -> bool:
    return "[MUTED]" in (text or "")


def volume_command(arg: str, current: int):
    """The ``wpctl`` argv for a ``/vol`` argument, or None for a query or
    nonsense. Absolute and capped at 100% — a relative ``10%+`` with no limit
    walks PipeWire past unity gain into clipping."""
    arg = (arg or "").strip()
    if not arg:
        return None
    level = step_level(arg, current, 100, floor=0, step=10)
    if level is None:
        return None
    return ["wpctl", "set-volume", SINK, f"{level}%"]


# ── the panel ───────────────────────────────────────────────────────────────

def panel_env(environ=None, uid=None) -> dict:
    """Environment for ``wlopm``. A client started by the session has these
    already; one restarted over ssh (deploy.sh) has neither, and ``wlopm``
    then fails with "failed to connect to display"."""
    env = dict(os.environ if environ is None else environ)
    uid = os.getuid() if uid is None else uid
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
    env.setdefault("WAYLAND_DISPLAY", "wayland-0")
    return env


def panel_is_off(wlopm_output: str):
    """``wlopm`` prints one ``<output> on|off`` line per panel. True when
    every panel is off, False when any is on, None when it said nothing
    (not Wayland, tool missing) — unknown is not "off"."""
    states = [ln.split()[-1].lower() for ln in (wlopm_output or "").splitlines()
              if len(ln.split()) >= 2 and ln.split()[-1].lower() in ("on", "off")]
    if not states:
        return None
    return all(s == "off" for s in states)


# ── face frame rate ─────────────────────────────────────────────────────────

FACE_FULL = 0          # uncapped
FACE_PAUSED = -1


def _env_num(name: str, default: float, environ=None) -> float:
    """A numeric env knob; anything that is not a finite number (a typo,
    ``nan``, ``1e999``) is the default — this is read inside a Qt timer slot,
    where an exception has nowhere to go."""
    try:
        value = float((os.environ if environ is None else environ).get(name, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def face_rate(idle_s: float, busy: bool, battery: bool, environ=None) -> int:
    """How fast the face should render: FACE_FULL, a frame cap, or FACE_PAUSED.

    Measured on the device (2026-10-01): with nothing happening the client
    took 44.5% of a core and the web renderer 29.9% — the face redrawing at
    full rate behind glass that re-blends on every frame. Nobody is watching
    most of that. So:

    * a turn in flight, speech, recording, or recent input → full rate;
    * idle past ``GHOST_FACE_IDLE_S`` → a slow cap (slower on battery);
    * idle past ``GHOST_FACE_SLEEP_S`` → paused. The panel blanks at 600 s
      (swayidle), so by then the frames are being drawn for a dark screen.

    Every threshold is an env knob so it can be tuned on the device without a
    redeploy; setting ``GHOST_FACE_IDLE_FPS=0`` disables the slow tier.
    """
    if busy:
        return FACE_FULL
    idle_after = _env_num("GHOST_FACE_IDLE_S", 45, environ)
    sleep_after = _env_num("GHOST_FACE_SLEEP_S", 600, environ)
    if sleep_after > 0 and idle_s >= sleep_after:
        return FACE_PAUSED
    if idle_s < idle_after:
        return FACE_FULL
    cap = _env_num("GHOST_FACE_BATTERY_FPS" if battery else "GHOST_FACE_IDLE_FPS",
                   5 if battery else 10, environ)
    if cap <= 0:
        return FACE_FULL            # the slow tier is switched off
    # At least 1: a knob of 0.5 truncated to 0 — which means UNCAPPED, the
    # opposite of what was asked for.
    return max(1, int(cap))
