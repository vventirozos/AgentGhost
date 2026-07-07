# Track B3 — idle-loop adjudication

harness meta: {'harness': 'trackb3', 'time_scale': 60.0, 'idle_epochs': 5}

## Probe outcomes
- control: 35/36 passed (97%)
- treatment: 35/36 passed (97%)
- treatment_uniform: 33/36 passed (92%)

## McNemar — treatment vs control (matched probe outcomes)
- matched pairs: 36  (both pass=34, both fail=0)
- discordant: treatment-only win b=1, control-only win c=1
- exact McNemar two-sided p = 1.0000  → no significant probe-success shift (see lesson yield)

## #27b — frontier vs uniform self-play (verified-lesson yield)
- frontier (default): lessons_by_source={'self_play': 2, 'reflection': 2} total_lessons=4 graduated_skills=0 proposed_macros=0
- uniform (--no-frontier-selfplay): lessons_by_source={'self_play': 2} total_lessons=2 graduated_skills=0 proposed_macros=0
- verdict input: frontier total=4 vs uniform total=2 (pre-registered criterion: keep frontier only if it out-yields uniform across repeats)

## Idle-loop learning artifacts (per repeat, per arm)
### repeat 0
- treatment: lessons_by_source={'self_play': 1} graduated_skills=0 proposed_macros=0
- treatment_uniform: lessons_by_source={'self_play': 1} graduated_skills=0 proposed_macros=0
- control: lessons_by_source={} graduated_skills=0 proposed_macros=0
### repeat 1
- treatment: lessons_by_source={'self_play': 1, 'reflection': 2} graduated_skills=0 proposed_macros=0
- treatment_uniform: lessons_by_source={'self_play': 1} graduated_skills=0 proposed_macros=0
- control: lessons_by_source={} graduated_skills=0 proposed_macros=0
