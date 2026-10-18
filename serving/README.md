# 启动 Qwen-VL 服务（serving）

`serve_qwen_vl.sh` 用 vLLM 拉起一个 OpenAI-compatible 的 Qwen-VL 端点，供
navagent 的 `--backend openai` 使用。脚本不包含任何固定的容器名、工作目录或
checkpoint 路径，全部通过参数或环境变量传入。

## 依赖

- 一个安装了 `vllm` 的 Python 环境（宿主机或容器内均可）；
- Qwen-VL checkpoint 目录。默认占位：

| 别名 | 环境变量 | 默认路径 |
| --- | --- | --- |
| `8b` | `QWEN3_VL_8B_PATH` | `/models_ro/Qwen3-VL-8B` |
| `27b` | `QWEN38_27B_VL_PATH` | `/qwen27b` |

也可以直接 `--model /path/to/checkpoint`。`--model` 省略时使用
`QWEN_VL_MODEL`，再回退到 8B 路径。

## 启动

宿主机直接启动（默认后台，日志写到 `/tmp/qwen-vl-<port>.log`）：

```bash
./serving/serve_qwen_vl.sh --model 8b --gpu 0 --port 8000
```

Qwen3.8-27B-VL：

```bash
QWEN38_27B_VL_PATH=/path/to/Qwen3.8-27B-VL \
  ./serving/serve_qwen_vl.sh --model 27b --gpu 1 --port 8010 \
  --served-model-name qwen3.8-27b-vl --gpu-mem-util 0.92
```

在容器内启动（`CONTAINER` 或 `--container`）：

```bash
CONTAINER=vlm-nav \
  ./serving/serve_qwen_vl.sh --model 8b --gpu 0 --port 8000 \
  --vllm-python /opt/vllm/bin/python \
  --hf-home /models/.cache/huggingface
```

常用参数：

- `--served-model-name`：客户端请求里的 `model` 字段，必须与
  `--backend openai --model ...` 一致；
- `--language-model-only`：只加载语言塔（纯文本任务）；视觉导航不要加；
- `--thinking`：打开 Qwen thinking，默认关闭。导航 agent 直接解析工具调用
  JSON，建议保持关闭；
- `--limit-mm-image`：每个 prompt 允许的图片数（默认 4）。navagent 的
  `--context-frames` 超过这个值时需要同步调大；
- `--api-key` / `--api-key-file`：设置 `VLLM_API_KEY`。若设置了 key，
  navagent 侧必须提供相同的 `OPENAI_API_KEY`；
- `--extra`：透传任意 vLLM 参数，可重复；
- `--dry-run`：只打印最终命令，不启动；
- `--foreground`：前台运行，便于调试。

脚本默认带 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`，
这样 `OpenAIVLM` 的原生 tool calling 路径可用；`--reasoning-parser qwen3`
与 thinking 关闭的 chat-template 参数也由脚本统一处理。

检查端点：

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

停止：

```bash
pkill -f 'vllm serve.*--port 8000'
# 容器内
docker exec <container> pkill -f 'vllm serve.*--port 8000'
```

## 接到 navagent

`vlm_nav/vlm.py::OpenAIVLM` 会请求 `<base-url>/v1/chat/completions`，
`base-url` 写成 `http://127.0.0.1:8000` 或
`http://127.0.0.1:8000/v1` 都可以（脚本内部会规范化，避免出现 `/v1/v1`）。

```bash
export OPENAI_API_KEY=...   # 服务端未设 key 时可留空
python scripts/run_vlm_nav_tools.py \
  --world cpu --ply /path/scene.ply \
  --instruction "Find the book and fly to it." \
  --backend openai \
  --model qwen3-vl-8b \
  --base-url http://127.0.0.1:8000/v1 \
  --out outputs/vlm_nav/book_openai
```

其他驱动脚本（`run_vlm_nav_3dgs.py`、`run_vlm_nav_m2_memory.py`、
`compare_*.py`、`verify_*.py`）使用同一组 `--backend openai --model ...
--base-url ...` 参数，可直接透传。

## `--backend openai` vs `--backend hf`

| | `openai`（vLLM 服务） | `hf`（进程内加载） |
| --- | --- | --- |
| 部署 | 先起服务，再跑 agent | 不需要服务 |
| 显存 | 服务独占，可多客户端共享 | 与 agent 同进程，逐次前向 |
| 吞吐 | 连续批处理，适合批量评测 | 单请求，长跑较慢 |
| 工具调用 | 原生 `tools` / `tool_calls` | chat template 支持时原生，否则退回约束 JSON |
| 依赖 | 只需 HTTP 端点 | `transformers` + 本地权重 |
| 适用 | 正式评测、并行 episode | 快速调试、单次可复现实验 |

本地 HF 示例：

```bash
python scripts/run_vlm_nav_3dgs.py \
  --world cpu --ply /path/scene.ply --instruction "Find the book." \
  --backend hf --model-path /models_ro/Qwen3-VL-8B --device cuda \
  --out outputs/vlm_nav/book_hf
```

注意：`--language-model-only` 只适用于纯文本；视觉导航必须保留视觉塔。
