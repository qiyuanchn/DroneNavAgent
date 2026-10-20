# Mixed-mission matrix fixtures

`scripts/run_mixed_missions.py` loads `tasks_<N>.json` for the 2/4/8-stage
matrix. Each file has the same shape:

```json
{
  "schema": "vlm_nav.mixed_missions.tasks/1",
  "stages": 2,
  "groups": {
    "action":   [{"name": "...", "instruction": "...", "steps": [...]}],
    "semantic": [{"name": "...", "instruction": "...", "steps": [...]}],
    "mixed":    [{"name": "...", "instruction": "...", "steps": [...]}]
  }
}
```

Rules enforced by the loader:

- `len(steps) == stages` for every mission.
- action steps are `move/turn/hover/photo` only;
- semantic steps are `search/approach/confirm/verify/revisit/goto` only;
- mixed missions contain both categories and alternate them;
- every step validates through `vlm_nav.mixed_mission.MixedMissionSpec`.

The fixtures are small on purpose (0.5–1 m moves, 0.2 s hovers, mock semantic
service). They pin orchestration and evaluation, not scene-scale navigation.
`semantic_revisit_*` and `semantic_revisit_loop` are the delayed-revisit arms;
the driver's `--interrupt-step` adds the interruption/resume arm on top.

Real semantic runs need a VLM backend and the legacy perception/navigation
plugins; use `--semantic real` and a configured `model.provider` (or the
worker-backed provider from `vlm_nav.plugins.worker_model`). The mock service
is the CPU default so the matrix is reproducible without a model.
