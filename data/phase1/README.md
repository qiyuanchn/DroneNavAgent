# Phase-1 scenario presets

`compare_decision_loop.py` ships the `book` and `synth` presets two ways:

- the task lists in this directory (`tasks_book.json`, `tasks_synth.json`,
  `tasks_synth_fair.json`) are the real Phase-1 task definitions;
- `scenes/scenario_scene.ply` is a tiny synthetic smoke fixture, not the
  recorded evaluation scene. It only keeps the preset runnable/testable without
  a 3DGS data checkout.

For a real evaluation, point `VLM_NAV_PHASE1_DIR` at the recorded artifacts
(a directory laid out like `outputs/vlm_nav/verify_phase1_qwen`, with the task
files and `scenes/scenario_scene.ply`):

```bash
VLM_NAV_PHASE1_DIR=/path/to/phase1_artifacts \
python scripts/compare_decision_loop.py --scenario book --scenario synth ...
```

The `book` preset still reads the production combined scene from
`/scene_book.ply` (the container mount used by the recorded runs).
