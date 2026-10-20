# DroneNavAgent

面向 3D Gaussian Splatting（3DGS）场景的 VLM 语义导航 agent。模型看到第一人称
RGB 画面，通过工具调用完成“观察 → 定位 → 规划 → 飞行 → 验证”的闭环；同一套
agent 运行在 CPU 世界、HiGS GPU 世界和生产 3DGS 环境上。

代码从 DiffAero 的 `navagent` 分支整理而来（当前同步至 `b1794e25`），核心包改名
为 `vlm_nav`，不再依赖 `diffaero.vlm_nav` 导入路径。

## 架构

```
VLM backend (mock / hf / openai)
        │
        ▼
AgentRuntime ── 决策循环、预算、恢复、轨迹与 summary
        │
        ├── ToolRegistry（task-scoped 调用 + 统一完成契约）
        │     ├── perception: observe / ground_target / confirm_candidate / query_memory
        │     │               compare_candidates（含可选 detector gate）
        │     ├── navigation: propose_viewpoints / go_to_viewpoint / approach_candidate
        │     │                 check_path / return_to_observation / verify_goal / finish
        │     └── task: plan / execute_subtask / resume_plan（自规划与语义完成条件）
        ├── observations  统一 observation / candidate / viewpoint id 与图像存储
        ├── memory        identity/anchor-first 工作记忆、空间语义地图、episodic 证据
        ├── motion        任务级 move / turn / hover / photo、执行状态与恢复
        ├── mixed_mission 一句话任务链：动作 + 语义搜索/接近/确认/验证 + 控制
        ├── semantic      对现有 search/approach/confirm/verify 工具的服务化封装
        ├── task_planner  中英文确定性任务解析（可选 VLM 回退）
        ├── hardware      GPU/4090 资源画像与容量判定
        ├── plugins       插件生命周期宿主（manifest / service / cleanup / uninstall）
        │    ├── plugins/motion.py     flight-motion（motion.service@1）
        │    ├── plugins/mixed.py      task-executor（mission.service@1）
        │    ├── plugins/semantic.py   semantic-navigation（semantic.service@1）
        │    ├── plugins/legacy.py     perception / navigation / task / memory / model
        │    ├── plugins/worker.py     受管理 worker 进程（重型模型隔离）
        │    ├── plugins/profiles.py   motion-only / semantic-nav / mixed-mission / 4090-local
        │    └── plugins/infra.py      resource-manager / world-service / mcp-adapter
        └── world         CPU splat / HiGS GPU / 生产 EnvWorld / SUPER
        │
        ▼
MCP stdio server（同一 registry、同一 session）或脚本直接调用
```

- `vlm_nav/runtime.py`：主决策循环，负责 prompt、工具调用、失败恢复、完成判定。
- `vlm_nav/tools.py`：工具协议、registry、预算、证据与结果渲染；插件通过
  `register` / `unregister` 增删工具。
- `vlm_nav/motion.py`：任务级动作 schema 与执行器；长距离分段、固定朝向目标、
  仿真时间悬停、取消/恢复。
- `vlm_nav/plugins/`：M0 插件宿主（manifest、服务接口、作用域注册、资源引用计数、
  停用/卸载/数据清除），以及 `plugins/motion.py` 基础动作插件。
- `vlm_nav/tool_impl_*.py`：按感知 / 导航 / 任务拆分的工具实现。
- `vlm_nav/skills_approach.py`、`skills_search.py`、`skills_confirm.py`：接近、
  搜索与主动候选确认技能。
- `vlm_nav/detectors.py`：可选 SAM3 + DINOv2 检测门控（默认关闭，惰性导入）。
- `vlm_nav/observations.py`、`memory.py`：多视角观测和 identity/anchor-first 记忆。
- `vlm_nav/world.py`、`cpu_renderer.py`：世界后端；CPU 渲染器不依赖 CUDA。
- `vlm_nav/vlm.py`：`MockVLM`、`HFLocalVLM`、`OpenAIVLM`。
- `vlm_nav/mcp_server.py`：把同一套工具注册表暴露为 stdio MCP 服务。

## 目录

```
DroneNavAgent/
├── README.md
├── pyproject.toml              # 包名 vlm_nav；extras: hf / openai / mcp / test
├── vlm_nav/                    # 核心 agent 包（29 个模块 + plugins 子包）
│   ├── runtime.py  tools.py  actions.py  candidates.py
│   ├── motion.py  mixed_mission.py  task_planner.py  semantic.py
│   ├── vision.py  offline_batch.py  hardware.py
│   ├── observations.py  memory.py  world.py  cpu_renderer.py
│   ├── tool_impl_perception.py  tool_impl_navigation.py  tool_impl_task.py
│   ├── skills_approach.py  skills_search.py  skills_confirm.py
│   ├── detectors.py  vlm.py  prompts.py  harness.py
│   ├── plugins/__init__.py  plugins/{motion,mixed,semantic,task_planner}.py
│   ├── plugins/{legacy,profiles,worker,worker_model,infra,bootstrap}.py
│   ├── worker_echo.py  worker_model.py
│   └── mcp_server.py
├── scripts/                    # 运行、对比、验证、诊断与资源画像驱动
│   ├── run_vlm_nav_tools.py  run_vlm_nav_m2_memory.py  run_vlm_nav_3dgs.py
│   ├── run_motion_mission.py  run_mixed_mission.py  run_mixed_missions.py
│   ├── compare_agent_configs.py  compare_decision_loop.py
│   ├── verify_phase1.py  verify_long_horizon.py  diagnose_grounding.py
│   ├── profile_resources.py  check_hardware_profile.py
│   ├── replay_frames.py  build_replay_window_chair.py
│   ├── check_memory_identity.py  summarize_plan_ablation.py
│   └── vlm_nav_make_scene.py  vlm_nav_eval.py
├── docs/                       # 资源画像 / 4090 检查清单 / 混合任务契约
├── serving/                    # Qwen-VL vLLM 启服脚本与说明
├── tests/                      # agent 侧测试（44 个文件）
├── data/phase1/                # book/synth 场景的任务 JSON 与小型冒烟 PLY
├── data/mixed_missions/        # M4 的 2/4/8 阶段动作/语义/混合任务
└── third_party/diffaero        # git submodule：3DGS/HiGS/SUPER 验证环境
```

## 新增能力

同步自 DiffAero `navagent` 分支（`ce2239b3..b1794e25`）的能力：

- **task-scoped 工具调用与统一完成契约**：调用必须服务于当前任务/子任务，
  越界调用被拒绝；runtime、tools 与 MCP 共用同一完成判定。
- **主动候选确认**：`approach_candidate → 新视角观测 → accept/uncertain/reject`，
  结论由独立测量和证据门控决定，而不是外观或模型自述。
- **自规划模式**：模型可输出多阶段计划、按决策节点修订，并用语义
  `done_when` 条件推进子任务；修复了 fenced/截断 JSON 的解析。
- **identity/anchor-first 长期记忆**：候选以身份锚点而非位置索引，支持跨视角
  复核与拒绝记忆。
- **感知门控**：多视角射线三角化、instance/relation-anchor 共视检查、紧框
  crop 重试，以及可选的 SAM3 + DINOv2 检测器。
- **关系证据链**：面向 `chair near the window` 这类关系目标，跨观测累积关系
  证据，并用 support-cluster 测量校准接受/拒绝。
- **evaluator-free 通道与断点续跑**：agent 侧不依赖评测器；checkpoint 可逐字
  恢复（含计划与预算）。
- **资源画像与冻结回放**：`scripts/profile_resources.py` 测量显存、首调/稳态
  延迟和 token 速率，`scripts/replay_frames.py` 提供跨机器可比的冻结观测集；
  见 `docs/vlm_nav_resource_profile_20261020.md` 的 4090 检查清单。
- **任务级动作（M1）**：`motion` 把“前进 5 米、悬停 3 秒、转向后继续移动”
  作为带完成条件的任务链执行。5 米会按 ≤2 米分段，但目标是完整 5 米；
  被场景边界截断时返回 `blocked` 和剩余距离，不会虚报完成。朝向在动作开始时
  固定，悬停推进仿真时钟而不是等待模型调用墙钟，取消/恢复按已执行进度续跑。
- **插件生命周期（M0）**：manifest 声明服务、工具、提示词、事件、后台任务和
  数据/缓存目录；注册均返回可撤销句柄。停用保留用户数据，卸载移除私有缓存，
  `purge_data` 才删除私有数据；共享资源按引用计数释放；清理失败保留为
  `residual` 并支持重试。
- **一句话混合任务（M2）**：中英文确定性 planner 把
  “前进 5 米、悬停 3 秒、找到窗边的木椅、飞到它旁边并拍照”解析成有序任务链；
  动作步走 M1 执行器，语义步走既有 `search_target`/`approach_candidate`/
  `confirm_candidate`/`verify_goal`。语义步只有拿到 candidate / arrived /
  accepted / verified 证据才算完成，纯动作不会冒充语义目标完成。
- **插件迁移、worker 与 profile（M3）**：现有 perception/navigation/task/
  memory/model 模块通过 adapter plugin 注册，`registry_view()` 保证停用后无
  残留工具；重型 VLM 可放入受管理 worker 进程（JSON-line RPC、超时、进程组
  SIGTERM/SIGKILL、卸载即回收）；内置 `motion-only`、`semantic-nav`、
  `mixed-mission`、`4090-local` profile。
- **长程矩阵与 4090（M4）**：`run_mixed_missions.py` 跑 action / semantic /
  mixed × 2/4/8 阶段 × 种子的矩阵，记录成功率、错误完成、工具调用、恢复行为
  和峰值显存；`check_hardware_profile.py` 用 4090 24 GiB 预算判定组合驻留，
  CPU/无 CUDA 时明确返回 unavailable。

## 任务级动作与插件生命周期（M0/M1）

mission schema（`drone.motion.mission/1`）：

```json
{
  "mission_id": "m1_acceptance",
  "instruction": "前进 5 米，悬停 3 秒，左转 90 度，继续前进 1 米并拍照",
  "steps": [
    {"kind": "move", "direction": "forward", "meters": 5.0},
    {"kind": "hover", "seconds": 3.0},
    {"kind": "turn", "direction": "left", "degrees": 90},
    {"kind": "move", "direction": "forward", "meters": 1.0},
    {"kind": "photo", "label": "after_turn"}
  ]
}
```

`move` 的 `forward/backward/left/right/up/down` 指该动作开始时机体水平朝向；
`meters` 是目标位移而不是累计航程；`turn` 正角度为左转（FLU 逆时针）；
`hover` 的 `seconds` 是仿真/飞控时间。工具面为
`motion_mission / motion_move / motion_turn / motion_hover / motion_photo /
motion_cancel / motion_resume / motion_state`；也可通过
`motion.service@1` 服务被其他插件调用。

安装、运行、停用、卸载一次走完：

```bash
python scripts/run_motion_mission.py \
  --artifacts-dir outputs/motion_missions \
  --plugin-root outputs/motion_plugins \
  --out outputs/motion_missions/run
```

脚本复用 agent 的 `ToolRegistry`，输出 `summary.json`（每步状态、剩余距离/
角度/时间、照片路径）以及 deactivate/uninstall 报告。照片与 `*.state.json`
位于 `artifacts-dir`，不受插件卸载影响；恢复时已完成的步骤不会重放，
已存在的照片不会重拍。`--resume-state` 可从保存的状态继续。

## 一句话混合任务（M2）

默认指令就是 goal.md 的示例：

```bash
python scripts/run_mixed_mission.py --semantic mock \
  --artifacts-dir outputs/mixed_missions \
  --plugin-root outputs/mixed_plugins \
  --out outputs/mixed_missions/run
```

- `--semantic mock`：无 VLM/GPU 的确定性语义服务，跑通完整
  plan → install → activate → run → deactivate → uninstall。
- `--semantic real`：安装 perception/navigation/semantic 插件，走真实
  `search_target`→`approach_candidate`→`confirm_candidate`→`verify_goal`；
  没有模型后端时会在 search 步诚实失败，不降级成 mock。
- `--motion-only`：只跑动作链，用于排除视觉因素。
- 在现有 tool-loop runner 中启用同一套插件：
  `python -m scripts.run_vlm_nav_tools ... --mission-plugin on`。它会向
  已经构建好的 registry 注册 `motion_*`、`mission_*`、`plan_mission` 工具；
  若 registry 已有 search/approach/confirm/verify，还会提供
  `semantic.service@1`（默认关闭，保持既有实验基线不变）。

任务 schema（`drone.mixed_mission/1`）在动作步之外支持
`search / approach / confirm / verify / revisit / goto`，以及
`takeoff / land / orbit / patrol`（只有 world 暴露对应方法时才执行，
否则明确报 `unsupported_capability`）。`mission_run` 每步后更新
`ctx.extra["subgoal"]` 与 `ctx.extra["mission.progress"]`，runtime 的
Structured state 会显示 `mission=... step=.../... done=...`。

## 插件迁移、worker 与 profile（M3）

- adapter plugins：`perception.core`（6 工具）、`navigation.core`（12 工具）、
  `task.tools`（2 工具）、`memory.store`、`model.provider`。
- `PluginContext.registry_view()` 让既有 `register_*_tools()` 直接接入插件
  作用域；停用后工具、服务、事件、worker 全部撤销。
- `plugins/worker.py` 提供 `ManagedWorker`/`WorkerPlugin`：独立进程、
  JSON-line RPC、启动握手、请求超时、stderr 日志、进程组 SIGTERM→SIGKILL。
  `plugins/worker_model.py` 把 VLM 后端放到 worker，并通过
  `model.provider@1` 暴露给 perception/navigation。
- profile：

```python
from vlm_nav.plugins.profiles import BUILTIN_PROFILES, activate_profile
# motion-only / semantic-nav / mixed-mission / 4090-local
```

`4090-local` 使用 `model.provider.worker`、4-bit 量化配置和本地 detector；
具体模型路径由 profile factory 提供。

## 长程矩阵与 4090（M4）

```bash
# 2/4/8 阶段 × action/semantic/mixed，CPU mock，可中断恢复
python scripts/run_mixed_missions.py \
  --groups action,semantic,mixed --stages 2,4,8 --seeds 1 \
  --interrupt-step 2 --out outputs/mixed_missions/matrix

# 4090 容量判定（无 CUDA 时返回 exit 2，不会伪造显存数字）
python scripts/check_hardware_profile.py --profile 4090-local --device cuda
```

`report.json` / `report.md` 记录每组每阶段：`success_rate`、
`false_completion`、`tool_calls`、`vlm_calls`、`wall_time_s`、峰值显存、
`resume_recovered`、`replayed_steps`，以及卸载后的资源稳定性
（工具/服务/worker 进程/residual）。语义组和混合组的 mock 服务只用于
CPU 矩阵与编排回归；真实语义能力由 `--semantic real` 或
`run_mixed_mission.py --semantic real` 验证。

## VLM 成本与吞吐

- **输入尺寸**：`OpenAIVLM` / `HFLocalVLM` 默认 `image_size=(320, 180)`，与
  传感器原生渲染一致；`--image-size 320 180` 是各驱动脚本的新默认值。此前的
  640×360 是把原图放大 4 倍再送入模型，视觉 token 翻倍但信息不增加。
- **scan contact sheet**：扫描/视角比较把 8/10 个视角拼成一张带编号的 contact
  sheet（`vlm_nav/vision.py`），prompt 直接要求返回格子编号；确认对象和最终
  identity 判断仍使用单张原图/裁剪图，不使用拼图。
- **客户端**：`OpenAIVLM` 使用 `requests.Session` keep-alive（无 requests 时
  回退 urllib）、JPEG quality 90、重复图片编码 bytes 的 LRU 缓存；历史图预算
  `max_history_images` 默认 1。`decide` 在 `json_mode` 下发送
  `response_format={"type":"json_object"}`，也接受 `guided_json`；单工具调用
  默认 `max_new_tokens=96`，plan turn 由 runtime 提高到
  `plan_max_new_tokens`。
- **连续批处理/共享端点**：批量评测复用同一个 `--base-url`，不要每个 episode
  起一个 server；vLLM 自己做连续批处理。离线 replay 可用
  `vlm_nav/offline_batch.py::VLLMBatchRunner`，它把整批 conversations 一次交给
  vLLM `LLM.chat`（需要本机 vLLM；无 vLLM 时给出明确错误）。
- **模型级联**：纯动作任务直接走 `motion_mission`/`mission_run`，不调用 VLM；
  候选定位优先使用 `--detector sam3_dino`，8B VLM 负责 plan、关系确认和最终
  identity。
- **可观测性**：vLLM `/metrics` 看 TTFT / inter-token latency / prompt 与
  generation tokens；客户端日志的 `usage_metrics` 记录
  `cached_tokens`，用于验证 prefix caching 和 contact sheet 的实际收益。
- **replay 精度验收**：`scripts/replay_accuracy.py` 用 replay 集的已知位姿
  投影出“目标在哪些视角可见”的真值，分别跑旧的逐帧 640×360 和新的 320×180
  contact sheet，输出 `exact / visible / within-one` 精度、图片数、像素数、
  prompt token 和 cached token：

```bash
python scripts/replay_accuracy.py \
  --replay-dir outputs/vlm_nav/replay/window_chair \
  --backend openai --model qwen3-vl-8b \
  --base-url http://127.0.0.1:8000/v1 --repeat 3 \
  --out outputs/vlm_nav/replay_accuracy
```

  `--dry-run` 不需要模型，只验证 replay 真值、拼图和真值投影；无模型/无
  CUDA 时不会伪造精度。

## 快速开始

```bash
git clone git@github.com:qiyuanchn/DroneNavAgent.git
cd DroneNavAgent
python -m venv .venv && . .venv/bin/activate
pip install -e '.[hf,mcp,test]'     # CPU 核心只需要 numpy/torch/Pillow

pytest tests -q
python -c "import vlm_nav"
python -m vlm_nav.mcp_server --service env --list
```

无模型权重的 CPU 冒烟（使用仓库内的小型 PLY fixture）：

```bash
python scripts/run_vlm_nav_tools.py \
  --world cpu --ply data/phase1/scenes/scenario_scene.ply \
  --target-name target --target-pos 0.48 1.404 0.442 --target-radius 0.2 \
  --start-pos -1.52 0.904 0.642 \
  --instruction "Find the red object and fly to it." \
  --backend mock --out outputs/vlm_nav/smoke
```

产物：`tool_trace.jsonl`（逐轮模型回复、工具调用、结果、证据）与
`summary.json`（成功 / 结束原因、工具统计、预算）。

## VLM 后端

### OpenAI-compatible 端点（推荐用于批量评测）

```bash
./serving/serve_qwen_vl.sh --model 8b --gpu 0 --port 8000

python scripts/run_vlm_nav_tools.py ... \
  --backend openai --model qwen3-vl-8b \
  --base-url http://127.0.0.1:8000/v1
```

`OpenAIVLM` 会请求 `<base-url>/v1/chat/completions`，因此 `base-url` 带不带
`/v1` 都可以。`serving/README.md` 覆盖 8B / 27B checkpoint、容器内启动、
API key、thinking 开关和 `hf` 后端的取舍。

### 本地 Hugging Face

```bash
python scripts/run_vlm_nav_3dgs.py ... \
  --backend hf --model-path /path/to/Qwen3-VL-8B --device cuda
```

`transformers` 只在 `HFLocalVLM` 构造时导入，属于 `hf` extra。

## MCP

`vlm_nav.mcp_server` 与脚本共用同一个 `ToolRegistry`、预算、日志和证据存储：

```bash
# 工具清单（不进入 stdio 循环）
python -m vlm_nav.mcp_server --service env --list

# 以 stdio MCP 服务运行
python -m vlm_nav.mcp_server --service env --frames-dir outputs/vlm_nav/run
python -m vlm_nav.mcp_server --service evidence
```

环境服务提供移动 / 观察 / 验证工具，evidence 服务提供
`ground_target` / `query_memory` / `compare_candidates` / `inspect_region`。
资源 URI（`nav://obs/...`、`nav://map/bev`、`nav://traj/...`）由 runtime 解析为
真实图像或 JSON，而不是裸路径。

## 与 diffaero 的关系

`vlm_nav` 是独立包：`import vlm_nav`、CPU 世界、核心决策循环和 agent 侧测试都不
需要 diffaero。以下验证能力通过 `third_party/diffaero` 子模块提供：

- HiGS GPU 渲染（`SplatWorldHiGS`）；
- 生产 3DGS 环境（`EnvWorld`、`ObstacleAvoidanceGaussianSplatting`）；
- SUPER 规划器（`SuperPlannerWorld`，及其 `super_planner_py` 扩展）；
- 场景组合 / 资产目录（`scripts/vlm_nav_make_scene.py`）；
- 矩阵评测所需的完整场景与 checkpoint 路径。

初始化并使用：

```bash
git submodule update --init third_party/diffaero
export PYTHONPATH="$PWD/third_party:$PYTHONPATH"
# 按 third_party/diffaero/README.md 安装其依赖并构建 SUPER / HiGS 扩展
```

`vlm_nav/world.py` 会自动在 `third_party/diffaero` 下查找 `cfg`，也支持
`DIFFAERO_ROOT=/path/to/diffaero` 指向独立 checkout。对 `diffaero.utils.*` 的
导入都位于方法内部（惰性），缺 diffaero 时 CPU 世界与核心循环仍可导入；只有真正
构造 HiGS / env / SUPER 世界或运行场景组合脚本时才会报缺少依赖。

## 测试

```bash
pytest tests -q
```

测试覆盖 core、perception、navigation、planning、memory、vlm、MCP、
confirmation、relation chain、detectors、插件生命周期、任务级动作和
scripts/drivers。`test_motion_mission.py` 固定了长距离分段、边界截断报告、
固定朝向、仿真时间悬停以及取消/恢复不重放；`test_plugin_lifecycle.py`
覆盖依赖解析、活跃任务保护、共享资源引用计数、失败回滚与清理重试；
`test_mixed_mission*.py` / `test_task_planner.py` 覆盖混合任务 schema、
planner、证据门控和插件工具；`test_semantic_*.py` 覆盖 search→approach→
confirm→verify 的 retry/next_action/不虚报；`test_worker_*.py` 覆盖 worker
启动/停止/超时/进程回收；`test_run_mixed_missions.py` 覆盖 2 阶段矩阵与
中断恢复；`test_hardware.py` 覆盖 4090 容量判定与无 CUDA 的诚实降级。源仓库中属于验证侧
的 `test_low_altitude_envelope.py`、`test_task_rewriter.py`、
`test_waypoint_*.py` 留在 diffaero，不随本仓库复制。
唯一需要 diffaero 的 agent 测试是 `test_vlm_nav_frames_contract.py` 中的
camera-mount 用例（`diffaero.utils.randomizer`），在缺少 diffaero 的环境中会
通过 `pytest.importorskip` 跳过；有 submodule / 已安装 diffaero 时正常执行。

`scripts/build_replay_window_chair.py` 从 SPZ 构建回放集时需要 diffaero 的 SPZ
解码器和它的 `scripts/spz_to_3dgs_ply.py`；本仓库不复制后者。直接使用已有 PLY
或 `scripts/replay_frames.py` 加载冻结回放不受影响，缺少转换器时脚本会给出明确
错误。

## 产物与数据

`outputs/`、checkpoint、PLY/SPZ 大文件均被 `.gitignore` 忽略。仓库只保留
`data/phase1/` 下的任务 JSON 和一个 4 KB 的冒烟 PLY；真实评测请用
`VLM_NAV_PHASE1_DIR`（见 `data/phase1/README.md`）或命令行参数指向数据集。

## 致谢

核心仿真、3DGS 渲染、HiGS 与 SUPER 相关工作来自
[DiffAero](https://github.com/zxh0916/diffaero)。本仓库只负责 VLM 导航 agent
层，并保持与验证侧的解耦。
