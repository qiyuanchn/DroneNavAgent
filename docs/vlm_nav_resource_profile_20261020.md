# VLM-navigation resource profile and window-chair replay (2026-10-20)

Milestone 1 of the new direction: a reproducible resource baseline for the
VLM-navigation stack, a frozen observation replay set, and the 4090 port
inventory. No `vlm_nav/` core logic changed; the new code is
`scripts/profile_resources.py`, `scripts/replay_frames.py`,
`scripts/build_replay_window_chair.py` plus tests.

Generated artifacts (not tracked by Git, `.gitignore` excludes `outputs/`):

- `outputs/vlm_nav/resource_profile/report.md` and `report.json` - the measured
  A100 profile and the audit.
- `outputs/vlm_nav/replay/window_chair/` - 15 frozen views + `manifest.json` +
  `README.md` + the cached `scene/combo.ply` used to regenerate them.

## What was measured (A100-SXM4-80GB, GPU0)

Host: torch 2.7.1+cu128, CUDA 12.8, transformers 5.2.0, compute capability 8.0,
`CUDA_VISIBLE_DEVICES=0`, `TORCH_CUDA_ARCH_LIST=8.0;8.9;12.0`,
`TORCH_EXTENSIONS_DIR=outputs/vlm_nav/torch_extensions`. The profile runs every
GPU section in its own process; `report.json.sections_isolated` records the
source of each number. Peak is `torch.cuda.max_memory_allocated` unless stated.

| Component | Phase | Time | Peak allocated | Peak reserved |
|---|---|---|---:|---:|
| Qwen3-VL-8B BF16 (8.77B params) | load | 6.29 s | 16,722 MiB (16.33 GiB) | 16,742 MiB |
| Qwen3-VL-8B | 1 frame, first call | 1.20 s | 16,809 MiB | 16,846 MiB |
| Qwen3-VL-8B | 1 frame, steady (4 calls) | 0.53 s median (p95 1.12 s) | same | same |
| Qwen3-VL-8B | 8 views, first call | 1.81 s | 17,297 MiB | 17,366 MiB |
| Qwen3-VL-8B | 8 views, steady | 1.53 s median (p95 1.79 s) | same | same |
| Qwen3-VL-8B | 10 serial calls | 0.53 s/call | growth 0.0 MiB | growth 2.0 MiB |
| SAM3 + DINOv2 (FP32) | load | 2.57 s | 3,554 MiB | 3,558 MiB |
| SAM3 + DINOv2 | `detect` steady | 0.43 s (first 1.79 s) | 4,182 MiB | 4,268 MiB |
| SAM3 + DINOv2 | `embed_boxes` (1 crop) | 0.011 s | 3,681 MiB | 4,268 MiB |
| HiGS bedroom (730,548 Gaussians) | scene load | 0.92 s | 145 MiB | 162 MiB |
| HiGS bedroom | 1 render (steady) | 1.62 ms | 199 MiB | 228 MiB |
| HiGS bedroom | 8-view `render_batch` | 2.48 ms (0.31 ms/view) | 609 MiB | 668 MiB |
| HiGS bedroom | 8 production `scan_views` | 11.80 ms | 609 MiB | 668 MiB |
| HiGS classroom (984,287 Gaussians) | scene load | 0.30 s | 803 MiB | 828 MiB |
| HiGS classroom | 1 render | 1.88 ms | 866 MiB | 914 MiB |
| HiGS classroom | 8-view `render_batch` | 2.66 ms (0.33 ms/view) | 1,416 MiB | 1,514 MiB |
| HiGS classroom | 8 production `scan_views` | 13.36 ms | 1,413 MiB | 1,514 MiB |

The first scene in a process also pays CUDA/extension warmup, so its load
seconds are not pure I/O; the batch call is the native `render_batch` GPU path,
while `scan_views` is eight production `world.render()` calls including the CPU
copy.
| `confirm_candidate` (full flow) | renderer load | 0.76 s | 238 MiB | 262 MiB |
| `confirm_candidate` | VLM load | 5.55 s | 16,951 MiB | 16,964 MiB |
| `confirm_candidate` | CPU depth-mirror build | 0.40 s | 17,065 MiB | 17,088 MiB |
| `confirm_candidate` | 2 fresh views + VLM + depth | 7.60 s | 17,371 MiB | 17,410 MiB |
| `confirm_candidate` | pitched depth read | 0.73 s | 17,065 MiB | 17,410 MiB |

Token rate on the A100: ~29.6 tok/s for one frame (262 prompt tokens) and
~29.3 tok/s for eight views (1,816 prompt tokens), at 64 new tokens per call.
The eight-view call costs about 2.9x the single-frame latency in this profile
(1.53 s vs 0.53 s steady), mostly prefill of 1,816 prompt tokens.

The confirm flow on the holdout window-chair scene was `verdict=uncertain`
with `missing=[identity, distance, localization]`: the VLM judged the chair
correctly in both fresh views, but the box-depth measurement landed 1.3 m from
the candidate estimate (it measured the lamp in front of the chair), so the
position check refused the acceptance. That is the routine working as designed,
not a profile failure; it also shows why an independent measurement is needed.

The 10 serial calls left **0 MiB allocated growth and 2 MiB reserved growth**.
Transformers `generate` allocates a per-call KV cache and frees it; there is no
persistent KV cache across calls. (The observation cache / candidate store grow
in CPU memory, not in the CUDA allocator.)

## `--replay`: the frozen window-chair observation set

Path: `outputs/vlm_nav/replay/window_chair/`. Built from the holdout bedroom
combination `combo_000000_seed_31001_bedroom_1f3a2fb80c29` (wooden spindle-back
chair at FLU `[2.4573, 2.8265, -2.3721]`, LED lamp next to it), rendered with the
production HiGS backend (`SplatWorldHiGS`, 1,197,012 Gaussians after merging).

| property | value |
|---|---|
| views | 15: 3 `approach`, 8 `scan`, 4 `pitched` |
| camera | 320x180, hfov 86 deg, fx=fy=171.58, cx=160, cy=90 |
| frames | `frames/view_00.png` ... `view_14.png` (RGB, native sensor size) |
| manifest | SHA-256 per frame, pose FLU, yaw, pitch, per-view intrinsics |
| instruction | the two-stage bedroom task (chair by the window -> lamp) |
| scene | target name/index/position, source SPZ/PLY, Gaussian count |

Every view's yaw points at the chair; pitched views add a positive (nose-down)
body pitch (20/25/30/35 deg). The frames were checked visually: the chair is
visible from the scan headings and the window/bed are in frame at the
window-side headings; the approach views are on the window/bed side (the +x
side of this scene is a wall, so a farther view from there was rejected during
generation).

Load it with the script-level API (no core change):

```python
from scripts.replay_frames import load_replay

replay = load_replay("outputs/vlm_nav/replay/window_chair")
frames = replay.frames()             # [15, 3, 180, 320] float CHW
images = replay.pil_frames()         # 15 RGB PIL images, the VLM image path
first = replay.views[0]
print(first.pose_flu, first.yaw_deg, first.pitch_deg, first.intrinsics)
```

The loader refuses a broken set rather than degrading a benchmark: unknown
schema, fewer than 12 views, duplicate ids, a missing frame, an absolute or
`..`-escaping path, a non-finite pose, or a SHA-256 mismatch raises
`ReplayError`. `--verify` on the builder validates the same contract.

Comparable measurement commands:

```bash
# same VLM profile on the A100 and on a 4090; the frames do not move
CUDA_VISIBLE_DEVICES=0 python scripts/profile_resources.py --sections vlm \
    --replay --replay-dir outputs/vlm_nav/replay/window_chair \
    --repeat 5 --serial-calls 10 --out outputs/vlm_nav/resource_profile/sections/vlm

# rebuild the replay (GPU, once per scene conversion)
CUDA_VISIBLE_DEVICES=0 python scripts/build_replay_window_chair.py
python scripts/build_replay_window_chair.py --verify
```

Note: `profile_resources.py` uses the replay frames for the VLM and detector
sections whenever `--replay-dir` exists (the default is the frozen set), so a
run is replay-based unless a different directory is passed; `--replay` is the
explicit form. The confirm section necessarily uses the live scene (it needs
depth), not the replay.

## 4090 adaptation inventory

Hard-coded assumptions found in the code. The right-hand column is the action
for a 24 GB Ada card; none of it changes vlm_nav core behaviour.

| area | location | finding | action |
|---|---|---|---|
| VLM dtype | `vlm_nav/vlm.py:633` | `HFLocalVLM` defaults to `dtype="bfloat16"`; runners do not override it | Keep BF16 (Ada supports it). 8-bit is the first 24 GB lever; 4-bit only when the vision tower + renderer do not fit |
| VLM device | `vlm_nav/vlm.py:656`, `scripts/run_vlm_nav_3dgs.py:84`, `scripts/run_vlm_nav_tools.py:946` | `device_map=self.device`; runner default is `--device cpu` (silent slow path if forgotten) | Pass `--device cuda:0` explicitly; record it in the summary |
| VLM model path | `scripts/run_vlm_nav_3dgs.py:11`, `scripts/vlm_nav_eval.py:12`, `scripts/verify_long_horizon.py:160`, `scripts/diagnose_grounding.py:69`, `scripts/compare_decision_loop.py:67`, `scripts/verify_phase1.py:22` | `/models_ro/Qwen3-VL-8B` (container mount) or `/home/dataset-assist-0/models/Qwen3-VL-8B` in docs | Mount the same path or pass `--model-path`; the profile defaults to `/models_ro/Qwen3-VL-8B` |
| Detector paths | `vlm_nav/detectors.py:51-52` | `/models_ro/sam3`, `/models_ro/dinov2-base` | Mount or pass `--sam3-path`/`--dino-path`; record both |
| Detector device | `vlm_nav/detectors.py:405,679` | defaults to `cuda:0` regardless of `CUDA_VISIBLE_DEVICES` | Pin the physical GPU with `CUDA_VISIBLE_DEVICES`; `cuda:0` then maps to it |
| Detector precision | `vlm_nav/detectors.py:494-503` | SAM3 and DINOv2 load in FP32 (no dtype argument); the device is hardcoded `cuda:0`, with no detector-device CLI knob | ~3.5 GiB measured; keep FP32 for parity and pin the physical GPU with `CUDA_VISIBLE_DEVICES`; try BF16/FP16 only with a score-parity check |
| HiGS extension cache | `utils/gaussian_splatting/higs/native/build.py:31-49` | extension name hashes sources + flags + `TORCH_CUDA_ARCH_LIST` + torch/CUDA versions; module lands in `TORCH_EXTENSIONS_DIR` or `~/.cache/torch_extensions` | On 4090 set `TORCH_CUDA_ARCH_LIST=8.9` (or `8.0;8.9`), `TORCH_EXTENSIONS_DIR=<repo>/outputs/vlm_nav/torch_extensions`; expect a new hash/compile |
| Rasterizer arch list | `script/build_batch_rasterizer.sh:8,40` | default `8.0;8.6;8.9;9.0;12.0+PTX` already includes Ada 8.9 | Keep it, or narrow to `8.9` on a 4090-only host to cut build time |
| Renderer precision | `utils/gaussian_splatting/higs/native/backend.py:42-80,111-115` | means FP32, quaternion/scale/opacity/color and the render buffer FP16; output FP16 `[V,H,W,4]` | No change needed; the FP16 path is Ada-native |
| `cuda_raycast` extension | `sensor/cuda_raycast.py:22` | fixed extension name `diffaero_cuda_raycast` (no arch digest) | A cache built on A100 can be stale on 4090; clear/rebuild or set a per-arch `TORCH_EXTENSIONS_DIR` |
| Frame size | `vlm_nav/vlm.py:637`, `cfg/sensor/3dgs_rgb.yaml:3-5` | VLM input 640x360; sensor renders 320x180 | Lowering `image_size` is the cheapest 4090 lever but changes the benchmark; freeze it in the replay metadata |
| Generation cap | `vlm_nav/vlm.py:634`, `scripts/run_vlm_nav_3dgs.py:87` | 192 new tokens default | Record it; decode time/KV growth are not comparable otherwise |
| Scene assets | `vlm_nav/world.py:619-625`, replay builder | HiGS loads PLY; the replay source is a 1.2M-Gaussian SPZ converted once | Ship/pre-convert the PLY; report Gaussian count and scene scale |
| Environment | container / `scripts/verify_long_horizon.py:158` | `TORCH_CUDA_ARCH_LIST`, `TORCH_EXTENSIONS_DIR`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `HF_HOME` are shell/env, not code | Record them in the run summary; a stale extension cache is the most likely 4090 "silent" failure |

### Quantization: measured BF16 anchor, converted 8/4-bit numbers

`bitsandbytes` is **not installed** in the A100 container, so no 8-bit/4-bit
run was measured. The measured BF16 peak is the anchor; the other rows are
parameter-count conversions (`P x bytes`), explicitly not 4090 results:

| precision | weights | + 1.0 GiB overhead | measured/estimated total for the stack |
|---|---:|---:|---|
| BF16 (measured) | 16.33 GiB | 17.33 GiB | 16.72 GiB allocator peak; with HiGS renderer + FP32 detector the live set is ~21 GiB |
| INT8 | 8.17 GiB | 9.17 GiB | ~13 GiB live set; the first recommendation for 24 GB |
| INT4 (NF4) | 4.08 GiB | 5.08 GiB | ~9 GiB live set; for 16 GB or a very tight 24 GB budget |

Recommendation: run BF16 first on the 4090 and read the actual peak; if the
VLM + 1.2M-Gaussian renderer + FP32 detector do not fit with headroom for the
CUDA context and activations, move the VLM to 8-bit and re-run the replay set
on both hosts. 4-bit is a fallback for 16 GB, not the default: it changes
numeric behaviour and must be validated against the same replay before its
numbers are compared. The A100's 8-view call is ~1.5 s steady; a 4090 should be
measured, not inferred from TFLOPS.

## OOM / silent-fallback audit

The audit scanned `outputs/vlm_nav/verify_long_horizon_final6` and `final7`:
**90 summary.json files, 0 complete, 90 missing every required field**. Existing
summaries carry `stack_backend`/`stack_detail`, but not the VLM/detector/
renderer resources or the peak memory. The minimal fields an episode summary
must record (also emitted as a table in `report.md`):

| field | why |
|---|---|
| `runtime.backend.kind` | `hf` / `openai` / `mock`; a mock run must never look like a model run |
| `runtime.backend.model` | path or model id actually loaded |
| `runtime.backend.device` | `cuda:0` vs `cpu`; the GPU the process really used |
| `runtime.backend.dtype` | `bfloat16` / `float16` / `float32` |
| `runtime.backend.quantization` | `none` / `8bit` / `4bit` and the bitsandbytes fallback |
| `runtime.detector.requested` | `off` / `sam3` / `module:factory` as asked for |
| `runtime.detector.active` | whether the gate really ran or was skipped |
| `runtime.detector.reason` | missing weights / no CUDA / import error / disabled |
| `runtime.renderer.backend` | `higs_ut` / `cpu_splat` / `env` |
| `runtime.renderer.device` | the render device |
| `resources.peak_allocated_mib` | `torch.cuda.max_memory_allocated` for the episode |
| `resources.peak_reserved_mib` | `torch.cuda.max_memory_reserved` for the episode |
| `resources.peak_driver_used_mib` | driver-side used bytes (CUDA context included) |
| `fallbacks` | list of `{component, requested, active, reason, step}`; `[]` when clean |

Silent-fallback checkpoints the audit expects to see an event/field for:

1. **Backend selection** - `make_backend` / `--backend` defaults to
   `mock:oracle`; a mock run must say `kind=mock`.
2. **Detector availability** - `Sam3DinoDetector.available()` is false on
   missing weights, no CUDA, or an import error; `resolve_detector` returns
   `NullDetector`. Record requested/active/reason.
3. **Depth reads** - `world.depth_at_pixel` returns `None` (no Gaussian, no
   mirror, no model); `confirm_candidate` reports `no_depth_measurement`.
4. **Renderer backend** - HiGS needs a built CUDA extension; the CPU splat
   renderer is a different backend. Record backend/device/cache key.
5. **Quantization** - a requested 8/4-bit load without bitsandbytes reports
   `available=False` + `quantization_error`; never silently load BF16.
6. **VLM device** - the runners' `--device cpu` default is a silent slow path.
7. **OOM retry** - if a runner catches `torch.cuda.OutOfMemoryError` and retries
   with fewer views/smaller images/another device, append a `fallbacks` entry
   with the original request, the retry and the step.
8. **Scene load** - a failed SPZ/PLY load may skip a cell
   (`finish_reason=infeasible_skip`); record the scene path, scale and Gaussian
   count so a skip is not read as a navigation failure.

`scripts/profile_resources.py --sections audit --audit-root <run-dir> ...`
reports the gaps; it never fills them with guessed values.

## Caveats

- All numbers are A100 measurements; no 4090 was available. The INT8/INT4
  table is a conversion, not a 4090 result.
- `torch.cuda.max_memory_allocated/reserved` exclude the CUDA context; use the
  driver-side `mem_get_info` numbers (in `report.json`) for a 24 GB fit
  decision. The profile's confirm section records both.
- Latency varies run-to-run (the same 8-view call measured 1.08-1.53 s across
  runs); use the replay set and the same `--image-size`/`--max-new-tokens` when
  comparing hosts. The report keeps first-call, steady median and p95 separate.
- Per-component peaks are only comparable when each section runs in its own
  process: SAM3 is a process-wide singleton and a loaded VLM keeps its weights.
  `report.json.sections_isolated` records the source report for every section.

## Reproduction

```bash
# one process per GPU section (isolated peaks), then merge
CUDA_VISIBLE_DEVICES=0 python scripts/profile_resources.py --sections vlm \
    --replay --repeat 5 --serial-calls 10 --max-new-tokens 64 \
    --out outputs/vlm_nav/resource_profile/sections/vlm
CUDA_VISIBLE_DEVICES=0 python scripts/profile_resources.py --sections detector \
    --replay --detector-calls 3 --out outputs/vlm_nav/resource_profile/sections/detector
CUDA_VISIBLE_DEVICES=0 python scripts/profile_resources.py --sections render \
    --render-repeat 5 --render-views 8 --out outputs/vlm_nav/resource_profile/sections/render
CUDA_VISIBLE_DEVICES=0 python scripts/profile_resources.py --sections confirm \
    --confirm-max-new-tokens 192 --confirm-views 2 --confirm-steps 6 \
    --out outputs/vlm_nav/resource_profile/sections/confirm
python scripts/profile_resources.py --sections replay,audit --dry-run \
    --audit-root outputs/vlm_nav/verify_long_horizon_final6 \
    --audit-root outputs/vlm_nav/verify_long_horizon_final7 \
    --out outputs/vlm_nav/resource_profile/sections/replay_audit

python scripts/profile_resources.py --merge \
    outputs/vlm_nav/resource_profile/sections/*/report.json \
    --out outputs/vlm_nav/resource_profile
```
