#!/usr/bin/env python
"""Run the M1 acceptance mission through the installable motion plugin.

The default mission is the one goal.md names:

    前进 5 米 -> 悬停 3 秒 -> 左转 90 度 -> 继续前进 1 米 -> 拍照

It runs on the CPU 3DGS world, through the same ``ToolRegistry`` the agent and
MCP server use. The script also exercises the full lifecycle: install, activate,
execute, deactivate, uninstall. Mission photos and state live outside the
plugin's private data directory, so the final report can show that the audit
trail survives the uninstall.

    python scripts/run_motion_mission.py

    python scripts/run_motion_mission.py \
        --mission-json mission.json --bound-radius 20 --out outputs/motion

    # continue an interrupted mission from its measured progress
    python scripts/run_motion_mission.py --resume-state outputs/motion/mission.state.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vlm_nav.motion import MissionState  # noqa: E402
from vlm_nav.plugins import PluginHost  # noqa: E402
from vlm_nav.plugins.motion import MotionPlugin  # noqa: E402
from vlm_nav.tools import ToolContext, ToolRegistry  # noqa: E402
from vlm_nav.world import SplatWorldCPU, TargetSpec  # noqa: E402

DEFAULT_MISSION = {
    "mission_id": "m1_acceptance",
    "instruction": "前进 5 米，悬停 3 秒，左转 90 度，继续前进 1 米并拍照",
    "steps": [
        {"kind": "move", "direction": "forward", "meters": 5.0},
        {"kind": "hover", "seconds": 3.0},
        {"kind": "turn", "direction": "left", "degrees": 90},
        {"kind": "move", "direction": "forward", "meters": 1.0},
        {"kind": "photo", "label": "after_turn"},
    ],
}


def _tuple3(values) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three numbers")
    return tuple(float(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install, run and uninstall the base motion plugin."
    )
    parser.add_argument(
        "--ply", default=str(REPO / "data" / "phase1" / "scenes" / "scenario_scene.ply"),
        help="CPU splat scene used for rendering (default: repository fixture)",
    )
    parser.add_argument(
        "--start-pos", nargs=3, type=float, default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--target-pos", nargs=3, type=float, default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--target-radius", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument(
        "--bound-radius", type=float, default=20.0,
        help="circular scene bound; keep it above the default 6 m path",
    )
    parser.add_argument(
        "--mission-json",
        help="mission JSON file (schema: {mission_id, instruction, steps}); "
             "defaults to the goal.md acceptance mission",
    )
    parser.add_argument("--mission-id", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument(
        "--resume-state",
        help="resume this stored mission state instead of starting a new mission",
    )
    parser.add_argument("--state-path", help="where to write the mission state")
    parser.add_argument(
        "--artifacts-dir", default=str(REPO / "outputs" / "motion_missions"),
        help="photos and state files (kept outside the plugin data dir)",
    )
    parser.add_argument(
        "--plugin-root", default=str(REPO / "outputs" / "motion_plugins"),
        help="lifecycle host root (private plugin data/cache)",
    )
    parser.add_argument(
        "--out", default=str(REPO / "outputs" / "motion_missions" / "run"),
        help="directory for summary.json",
    )
    parser.add_argument(
        "--keep-installed", action="store_true",
        help="leave the plugin installed (skip deactivate/uninstall)",
    )
    parser.add_argument(
        "--purge-data", action="store_true",
        help="also remove the plugin's private data directory on uninstall",
    )
    return parser


def load_mission(args: argparse.Namespace) -> dict:
    if args.mission_json:
        mission = json.loads(Path(args.mission_json).read_text(encoding="utf-8"))
    else:
        mission = json.loads(json.dumps(DEFAULT_MISSION))
    if args.mission_id:
        mission["mission_id"] = args.mission_id
    if args.instruction:
        mission["instruction"] = args.instruction
    return mission


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)
    state_path = Path(args.state_path) if args.state_path else artifacts_dir / "mission.state.json"

    world = SplatWorldCPU(
        args.ply,
        TargetSpec(
            name="target",
            position_flu=_tuple3(args.target_pos),
            radius_m=float(args.target_radius),
        ),
        start_flu=_tuple3(args.start_pos),
        start_yaw_deg=float(args.start_yaw_deg),
        width=int(args.width),
        height=int(args.height),
        bound_radius=float(args.bound_radius),
    )

    registry = ToolRegistry()
    host = PluginHost(root_dir=args.plugin_root, tool_registry=registry)
    plugin = MotionPlugin(artifacts_dir=artifacts_dir)
    reports = {
        "manifest": plugin.manifest.to_dict(),
        "install": host.install(plugin).to_dict(),
        "activate": host.activate(plugin.manifest.id).to_dict(),
    }
    context = ToolContext(
        world=world,
        episode_id="motion_demo",
        budget={"tool_calls_left": 32},
        extra={"motion.artifacts_dir": str(artifacts_dir)},
    )

    if args.resume_state:
        call_args = {
            "resume": True,
            "state_path": str(args.resume_state),
            "artifacts_dir": str(artifacts_dir),
        }
    else:
        mission = load_mission(args)
        call_args = {
            "mission": mission,
            "state_path": str(state_path),
            "artifacts_dir": str(artifacts_dir),
        }
    result = registry.call("motion_mission", call_args, context)
    state: MissionState | None = context.extra.get("motion.state")
    reports["mission"] = result.to_dict()
    if state is not None:
        reports["mission"]["state"] = state.to_dict()

    if not args.keep_installed:
        deactivate = host.deactivate(plugin.manifest.id)
        reports["deactivate"] = deactivate.to_dict()
        uninstall = host.uninstall(plugin.manifest.id, purge_data=bool(args.purge_data))
        reports["uninstall"] = uninstall.to_dict()
        reports["catalog_after_uninstall"] = host.catalog()
        reports["artifacts_preserved"] = [
            str(path) for path in sorted(artifacts_dir.glob("*")) if path.is_file()
        ]

    summary = {
        "result": result.status,
        "mission_status": state.status if state is not None else "",
        "position": [float(value) for value in world.position_flu],
        "yaw_deg": math.degrees(float(world.yaw_rad)),
        "reports": reports,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(result.prompt_text)
    print(f"position={summary['position']} yaw_deg={summary['yaw_deg']:.1f}")
    print(f"summary: {summary_path}")
    if not args.keep_installed:
        print(
            "lifecycle: "
            f"deactivate={reports['deactivate']['status']} "
            f"uninstall={reports['uninstall']['status']} "
            f"residual={len(reports['uninstall']['residual'])}"
        )
    ok = (
        result.status == "ok"
        and state is not None
        and state.status == "completed"
        and all(
            report.get("residual", []) == []
            for key, report in reports.items()
            if isinstance(report, dict) and key in ("deactivate", "uninstall")
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
