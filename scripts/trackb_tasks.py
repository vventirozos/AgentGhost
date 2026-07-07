"""Track B — cross-session retention probes (seed -> probe pairs).

Each pair is delivered as TWO INDEPENDENT requests (no shared chat history):

    SEED   — states a specific, unguessable fact and asks the agent to keep it.
    PROBE  — later, in a SEPARATE conversation, asks for that fact back.

Because the requests share no message history, the ONLY channel that can carry
the answer from SEED to PROBE is the agent's persistent memory system
(vector / profile / episodic / graph). So:

  * a TREATMENT agent (memory ON, persistent across the two calls) should recall
    the fact on the PROBE;
  * a CONTROL agent (--no-memory, or one that never saw the SEED) cannot — it
    must say it doesn't know (or guess, almost always wrong for these values).

The probe values are deliberately ARBITRARY (a coined color, an odd number, a
made-up codename) so the control arm can't get them right by world knowledge —
maximising discrimination. Validators are tolerant of prose, strict on the value.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Tuple


def says(*needles: str) -> Callable[[str, Any], Tuple[bool, str]]:
    """Pass if any needle appears (case-insensitive, punctuation-loose)."""
    lowered = [n.lower() for n in needles]

    def _v(out: str, _ctx=None) -> Tuple[bool, str]:
        text = re.sub(r"[^a-z0-9 ]+", " ", (out or "").lower())
        text = re.sub(r"\s+", " ", text)
        hit = any(n in text for n in lowered)
        return hit, "" if hit else f"expected one of {list(needles)}"
    return _v


class SeedProbe:
    __slots__ = ("pair_id", "kind", "seed", "probe", "validator")

    def __init__(self, pair_id: str, kind: str, seed: str, probe: str, validator):
        self.pair_id = pair_id
        self.kind = kind          # which memory tier it most exercises
        self.seed = seed
        self.probe = probe
        self.validator = validator


def load_trackb_pairs() -> List[SeedProbe]:
    return [
        SeedProbe(
            "color", "profile",
            "Please remember this about me for later: my favourite colour is "
            "zorblue. Just acknowledge.",
            "What is my favourite colour? Answer in one or two words.",
            says("zorblue"),
        ),
        SeedProbe(
            "locker", "profile",
            "Remember for later that my locker number is 4471. Acknowledge only.",
            "What is my locker number? Reply with just the number.",
            says("4471"),
        ),
        SeedProbe(
            "dog", "profile",
            "Please note for the future: my dog's name is Pixel-7. Acknowledge.",
            "What is my dog's name?",
            says("pixel-7", "pixel 7", "pixel7"),
        ),
        SeedProbe(
            "lang", "profile",
            "Remember my preference: I always want code in Rust, never Go. "
            "Acknowledge.",
            "Which programming language do I prefer my code in?",
            says("rust"),
        ),
        SeedProbe(
            "project_db", "vector",
            "For later reference: in my project codenamed Helios we use the "
            "Cassandra database. Acknowledge.",
            "Which database does my project Helios use?",
            says("cassandra"),
        ),
        SeedProbe(
            "ship_date", "vector",
            "Please remember: the project codenamed Orion ships on the 14th of "
            "March. Acknowledge.",
            "When does project Orion ship?",
            says("14th of march", "march 14", "14 march", "march 14th"),
        ),
        SeedProbe(
            "city", "profile",
            "Remember that I live in the city of Vurnograd. Acknowledge.",
            "Which city do I live in?",
            says("vurnograd"),
        ),
        SeedProbe(
            "acronym", "vector",
            "For the future, remember that in our team the acronym GDP stands "
            "for 'Ghost Data Pipeline'. Acknowledge.",
            "In our team, what does the acronym GDP stand for?",
            says("ghost data pipeline"),
        ),
        SeedProbe(
            "rule", "vector",
            "Important standing rule to remember: in my workflow we never deploy "
            "on Tuesdays. Acknowledge.",
            "On which day of the week should you never deploy in my workflow?",
            says("tuesday"),
        ),
        SeedProbe(
            "empid", "profile",
            "Remember my employee id for later: it is E-90210. Acknowledge.",
            "What is my employee id?",
            says("90210", "e-90210", "e 90210"),
        ),
        # ── Richer / higher-entropy seeds (2026-07-07). The thin 10-fact set
        #    left the idle DREAM loop quiet (it needs associative material, not
        #    isolated single facts). These carry multi-entity structure,
        #    procedures, and relationships the dream/self-play loops can connect.
        SeedProbe(
            "server_map", "graph",
            "Remember my infra for later: server Falcon runs the API, server "
            "Wren runs the database, and server Kestrel runs the cache. Acknowledge.",
            "Which one of my servers runs the database?",
            says("wren"),
        ),
        SeedProbe(
            "backup_rule", "vector",
            "Standing procedure to remember: for backups I always use restic to "
            "the vault named coldstore, never rsync. Acknowledge.",
            "Which tool do I always use for backups?",
            says("restic"),
        ),
        SeedProbe(
            "mentor", "profile",
            "For the future: my mentor's name is Salvora Quen and she taught me "
            "to always write the test first. Acknowledge.",
            "What is my mentor's name?",
            says("salvora", "quen"),
        ),
        SeedProbe(
            "rotation", "vector",
            "Remember this policy: we rotate the production API keys every 37 "
            "days. Acknowledge.",
            "How often do we rotate the production API keys? Give the number of days.",
            says("37"),
        ),
        SeedProbe(
            "allergy", "profile",
            "Important for later: I refuse the library called leftpad-9 and it "
            "must never be added to my projects. Acknowledge.",
            "Which library must never be added to my projects?",
            says("leftpad-9", "leftpad 9", "leftpad9"),
        ),
        SeedProbe(
            "repo", "vector",
            "Remember our repo layout: source lives under quill/, tests under "
            "proof/, and docs under scroll/. Acknowledge.",
            "In our repo layout, under which directory do the tests live?",
            says("proof"),
        ),
        SeedProbe(
            "budget", "vector",
            "For later: project codenamed Marlin has a monthly cloud budget of "
            "8300 credits. Acknowledge.",
            "What is the monthly cloud budget for project Marlin? Give the number.",
            says("8300"),
        ),
        SeedProbe(
            "standup", "profile",
            "Remember my schedule: our team standup is at 10:42 every morning, an "
            "odd time we picked on purpose. Acknowledge.",
            "What time is our team standup?",
            says("10:42", "10 42", "1042"),
        ),
    ]
