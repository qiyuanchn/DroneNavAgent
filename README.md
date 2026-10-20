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
        ├── plugins       插件生命周期宿主（manifest / service / cleanup / uninstall）
        │    └── plugins/motion.py  第一个完整动作插件（motion.service@1）
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
├── vlm_nav/                    # 核心 agent 包（21 个模块 + plugins 子包）
│   ├── runtime.py  tools.py  actions.py  candidates.py
│   ├── motion.py  observations.py  memory.py  world.py  cpu_renderer.py
│   ├── tool_impl_perception.py  tool_impl_navigation.py  tool_impl_task.py
│   ├── skills_approach.py  skills_search.py  skills_confirm.py
│   ├── detectors.py  vlm.py  prompts.py  harness.py
│   ├── plugins/__init__.py  plugins/motion.py
│   └── mcp_server.py
├── scripts/                    # 运行、对比、验证、诊断与资源画像驱动
│   ├── run_vlm_nav_tools.py  run_vlm_nav_m2_memory.py  run_vlm_nav_3dgs.py
│   ├── run_motion_mission.py  compare_agent_configs.py  compare_decision_loop.py
│   ├── verify_phase1.py  verify_long_horizon.py  diagnose_grounding.py
│   ├── profile_resources.py  replay_frames.py  build_replay_window_chair.py
│   ├── check_memory_identity.py  summarize_plan_ablation.py
│   └── vlm_nav_make_scene.py  vlm_nav_eval.py
├── docs/                       # 资源画像 / 4090 检查清单
├── serving/                    # Qwen-VL vLLM 启服脚本与说明
├── tests/                      # agent 侧测试（28 个文件）
├── data/phase1/                # book/synth 场景的任务 JSON 与小型冒烟 PLY
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
覆盖依赖解析、活跃任务保护、共享资源引用计数、失败回滚与清理重试。源仓库中属于验证侧
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
