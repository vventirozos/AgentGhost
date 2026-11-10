"""Face 2026-11-09 (§4TU): every mood label has a look, green means "it went
well", and the dead twitch / kick code is gone.

Operator, picking from the face ideas: "do 1 and 2 and also clean up (7)".
1  the six labels that looked neutral (content, calm, down, bored, curious,
   restless) get their own look and colours, placed by the agent's two mood
   numbers (pleasantness → warmth/light, energy → tempo);
2  a finished turn's outcome row ("✅ … turn outcome ok|verified|corrected")
   or a verifier pass sends a green front from the rim to the kernel;
7  the idle twitch (guard `!FORM === …`, always false) and the 'kick' tool
   dialect (only the empty form, which draws nothing) are removed.
"""
import ast
import filecmp
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests import test_interface_face_motion_2026_09_20 as M

_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "interface" / "static"
_JS = (_STATIC / "matrix_graph.js").read_text(encoding="utf-8")


def _labels():
    tree = ast.parse((_ROOT / "src" / "ghost_agent" / "selfhood" / "mood.py").read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LABELS":
            return [e.value for e in n.value.elts if isinstance(e, ast.Constant)]
    raise AssertionError("LABELS not found")


def _table(name):
    i = _JS.index(f"const {name} = {{")
    return _JS[i:_JS.index("\n};", i)]


def _looks():
    out = {}
    for m in re.finditer(r"^    (\w+): \{([^}]*)\}", _table("MOOD_LOOKS"), re.M):
        out[m.group(1)] = {k: float(v) for k, v in re.findall(r"(\w+): (-?[0-9.]+)", m.group(2))}
    return out


def _rings():
    return {m.group(1): re.findall(r"#([0-9a-f]{6})", m.group(2))
            for m in re.finditer(r"^    (\w+): +\[([^\]]*)\]", _table("MOOD_RINGS"), re.M)}


def test_every_mood_label_the_agent_can_report_has_a_look_and_colours():
    labels = _labels()
    # CURIOUS is a constant in mood.py; the tuple holds the others
    labels = set(labels) | {"curious"}
    looks, rings = _looks(), _rings()
    assert labels <= set(looks), labels - set(looks)
    assert labels <= set(rings), labels - set(rings)
    fields = set(looks["cheerful"])
    for k, v in looks.items():
        assert set(v) == fields, k


def test_the_colours_stay_dark_and_distinct():
    rings = _rings()
    for k, ring in rings.items():
        assert len(ring) == 5, k
        for h in ring:
            assert max(int(h[i:i + 2], 16) for i in (0, 2, 4)) <= 0x9e, (k, h)   # < 0.62: bloom lifts them
    as_t = {k: tuple(v) for k, v in rings.items()}
    assert len(set(as_t.values())) == len(as_t)
    # bored is the flat one: its stops sit closer together than any other's
    def spread(r):
        c = [[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in r]
        return max(abs(a - b) for x in c for y in c for a, b in zip(x, y))
    assert spread(rings["bored"]) == min(spread(r) for r in rings.values())


def test_neighbouring_moods_read_differently():
    L = _looks()
    # low energy: calm stays up and bright, tired sinks and dozes, down is shallow and grey
    assert L["calm"]["droop"] > L["tired"]["droop"] and L["calm"]["bright"] > L["tired"]["bright"]
    assert L["tired"]["doze"] > L["down"]["doze"] > L["calm"]["doze"]
    assert L["down"]["breath"] < 1 < L["tired"]["breath"]
    # high energy: irritable is hot and rigid, restless cool and loose
    assert L["irritable"]["heat"] > L["restless"]["heat"] and L["restless"]["wob"] > 3 * L["irritable"]["wob"]
    # cheerful has more tempo, curious more kernel
    assert L["cheerful"]["flow"] > L["curious"]["flow"] and L["curious"]["kernel"] > L["cheerful"]["kernel"]
    # pleasant ones are at least neutral-bright, unpleasant ones dimmer
    for k in ("cheerful", "content", "curious"):
        assert L[k]["bright"] >= 1, k
    for k in ("down", "bored", "tired"):
        assert L[k]["bright"] < 1, k


# ── executed: the green front and the outcome rows ──────────────────────────

pytestmark_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

_ROW_OK = "│  94  21:36:40  ✅  +12.5s  turn outcome              ok · no tools · 123 chars"
_ROW_VER = "\u001b[32m│  59  14:05:42  ✅  + 125s  turn outcome              verified · tools: x\u001b[0m"
_ROW_FAIL = "│  59  14:05:42  ❌  + 125s  turn outcome              failed · tools: x"
_SCEN = r"""
import { face, tick, emit } from './harness.mjs';
face.setForm('moodtesseract'); tick(240);
const st = () => face.getDebugState().tesseract;
const pure = [ROW_OK, ROW_VER, ROW_FAIL, 'thinking: the turn outcome ok was fine', 'turn outcome ok'].map(face.turnOutcomeOf);
const seeds0 = st().seeds.slice();
const idle = st().success;
face.noteLogLine(ROW_FAIL); tick(2);
const afterFail = st().success.t;
face.noteLogLine(ROW_OK); tick(1);
const startedOk = st().success;
const again = face.noteSuccess(0.6);           // same strength while running: no restart
const fronts = [], envs = [];
for (let f = 0; f < 200; f++) { tick(1); const s = st().success; fronts.push(s.front); envs.push(s.env); }
const mid = st().seeds.slice();
const end = st().success;
face.noteLogLine(ROW_VER); tick(1);
const ver = st().success;
const stronger = face.noteSuccess(0.6);        // a weaker one while the verified front runs: ignored
tick(300);                                      // let it clear
const cleared = st().success;
face.noteVerdict('refute'); tick(1);
const refuted = st().success;
face.noteVerdict('pass'); tick(1);
const passed = st().success;
emit({ pure, idle, afterFail, startedOk, again, fronts, envs, end, ver, stronger, cleared, refuted, passed,
       debugKeys: Object.keys(face.getDebugState()) });
""".replace("ROW_OK", json.dumps(_ROW_OK)).replace("ROW_VER", json.dumps(_ROW_VER)).replace("ROW_FAIL", json.dumps(_ROW_FAIL))


@pytest.fixture(scope="module")
def green(tmp_path_factory):
    if shutil.which("node") is None:
        pytest.skip("node not on PATH")
    root = tmp_path_factory.mktemp("green")
    M._write_harness(root, _STATIC / "matrix_graph.js")
    (root / "green.mjs").write_text(_SCEN, encoding="utf-8")
    r = subprocess.run(["node", "green.mjs"], cwd=root, env=dict(os.environ), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_only_a_turn_outcome_row_counts(green):
    assert green["pure"] == ["ok", "verified", "failed", None, None]   # free text and a bare phrase do not
    assert green["idle"]["t"] < 0 and green["idle"]["env"] == 0
    assert green["afterFail"] < 0, "a failed turn must not turn green"


def test_the_front_runs_from_the_rim_to_the_kernel_and_clears(green):
    s = green["startedOk"]
    assert s["t"] >= 0 and s["amp"] == pytest.approx(0.6)
    assert green["again"] is False
    fr = [f for f, e in zip(green["fronts"], green["envs"]) if e > 0]
    assert fr[0] > 0.85 and fr[-1] < 0.2, (fr[0], fr[-1])
    assert all(b <= a + 1e-9 for a, b in zip(fr, fr[1:])), "the front only moves inward"
    assert max(green["envs"]) == pytest.approx(0.6, abs=0.02)
    assert green["end"]["t"] < 0 and green["end"]["env"] == 0   # 2.6 s at 60 fps < 200 frames


def test_a_verified_turn_is_stronger_and_not_overridden(green):
    assert green["ver"]["amp"] == 1 and green["ver"]["t"] >= 0
    assert green["stronger"] is False


def test_the_dead_code_is_gone(green):
    assert "twitchAmp" not in green["debugKeys"]
    for gone in ("idleTwitch", "twitchAmp", "toolKick", "toolRadial", "!FORM ===", "tool: ['kick'"):
        assert gone not in _JS, gone


def test_both_routes_feed_the_front(green):
    # the log route lives in matrix_graph (so the handheld, which feeds the same
    # lines through noteLogLine, gets it too — startedOk above); a verifier pass
    # also starts it, at full strength, and a refute does not
    assert green["cleared"]["t"] < 0
    assert green["refuted"]["t"] < 0
    assert green["passed"]["t"] >= 0 and green["passed"]["amp"] == 1
    hand = _STATIC.parent / "externals" / "clockwork_ghost" / "webface" / "matrix_graph.js"
    assert filecmp.cmp(hand, _STATIC / "matrix_graph.js", shallow=False)   # the handheld runs the same face
