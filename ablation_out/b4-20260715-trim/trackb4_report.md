# Track B4 — grounded outcome battery

harness meta: {'harness': 'trackb4', 'time_scale': 60.0, 'idle_epochs': 8, 'battery_size': 32}

## Probe outcomes (grounded verify)
- control: 94/96 passed (98%)
- treatment: 94/96 passed (98%)
- treatment_uniform: 95/96 passed (99%)

## Treatment vs control
- matched pairs: 96 (both pass=92, both fail=0); discordant b=2 (treatment-only win), c=2 (control-only win)
- exact McNemar two-sided p = 1.0000
- task-STRATIFIED sign-flip: mean per-task delta = +0.000 over 32 tasks, p = 1.0000  (the primary test — repeats within a task are correlated)

## Mediation (did lessons actually surface during probes?)
- repeat 0 treatment: mediation_rate=75%
- repeat 0 treatment_uniform: mediation_rate=97%
- repeat 0 control: mediation_rate=100%
- repeat 1 treatment: mediation_rate=84%
- repeat 1 treatment_uniform: mediation_rate=16%
- repeat 1 control: mediation_rate=100%
- repeat 2 treatment: mediation_rate=100%
- repeat 2 treatment_uniform: mediation_rate=100%
- repeat 2 control: mediation_rate=100%
- pre-registered reading: outcomes-null + mediation≈0 → fix retrieval routing; outcomes-null + mediation healthy → idle output doesn't transfer at this scale.

## #27b — frontier vs uniform on the WEAK clusters ['sql', 'regex_parse', 'algo', 'concurrency']
- treatment: weak-cluster probes 47/48 (98%)
- treatment_uniform: weak-cluster probes 47/48 (98%)
- repeat 0 treatment: lessons_by_source={'perfection_protocol': 2, 'self_play': 1}
- repeat 0 treatment_uniform: lessons_by_source={'perfection_protocol': 2, 'self_play': 1}
- repeat 1 treatment: lessons_by_source={'self_play': 1}
- repeat 1 treatment_uniform: lessons_by_source={'self_play': 1}
- repeat 2 treatment: lessons_by_source={'perfection_protocol': 3, 'self_play': 1}
- repeat 2 treatment_uniform: lessons_by_source={'perfection_protocol': 1, 'self_play': 1}
- pre-registered rule (§4D item 6): KEEP frontier iff self-play yield ≥ uniform in ≥2/3 repeats AND weak-cluster delta ≥ 0; else flip default to uniform. PRM stays either way.

## Dream / idle-loop instrumentation
- repeat 0 treatment: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=40
- repeat 0 treatment_uniform: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=40
- repeat 0 control: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=39
- repeat 1 treatment: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=40
- repeat 1 treatment_uniform: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=40
- repeat 1 control: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=39
- repeat 2 treatment: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=39
- repeat 2 treatment_uniform: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=40
- repeat 2 control: auto_memories(seed)=0 failed_traj(seed)=0 dream_skips(final)=0 hydrations(final)=39
- gate reading: skips>0 with auto_memories≥3 = NEW BUG; auto_memories<3 = seeding still starves the gate.
