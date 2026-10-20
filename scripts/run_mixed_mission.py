#!/usr/bin/env python
"""Run a mixed mission (motion + semantic + control) through the plugin stack.

The default instruction is the goal.md M2 example:

    上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，找到窗边的木椅，
    飞到它旁边，再向左移动 1 米并拍照。

Modes:

* ``--motion-only``: install motion + planner + mixed-mission plugins only and
  execute a motion-only instruction (no semantic tools, no semantic service);
* ``--semantic mock`` (default): inject a deterministic fake semantic service
  so the whole chain runs on CPU without a VLM/GPU/network;
* ``--semantic real``: install the legacy perception/navigation plugins and the
  semantic navigation plugin when those modules are importable.

Every mode exercises the full lifecycle install -> activate -> run ->
deactivate -> uninstall and writes a JSON summary.  Photos, mission state and
the summary live outside the plugins' private data directories, so they remain
readable after uninstall.

    python scripts/run_mixed_mission.py --motion-only
    python scripts/run_mixed_mission.py --semantic mock
    python scripts/run_mixed_mission.py --semantic real
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

from vlm_nav.mixed_mission import (  # noqa: E402
    MOTION_STEP_KINDS,
    MixedMissionState,
)
from vlm_nav.plugins import PluginHost  # noqa: E402
from vlm_nav.plugins.mixed import MixedMissionPlugin  # noqa: E402
from vlm_nav.plugins.motion import MotionPlugin  # noqa: E402
from vlm_nav.plugins.task_planner import TaskPlannerPlugin  # noqa: E402
from vlm_nav.tools import ToolContext, ToolRegistry  # noqa: E402
from vlm_nav.world import SplatWorldCPU, TargetSpec  # noqa: E402

DEFAULT_INSTRUCTION = (
    "上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，"
    "找到窗边的木椅，飞到它旁边，再向左移动 1 米并拍照。"
)
DEFAULT_MOTION_INSTRUCTION = (
    "上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，"
    "向左移动 1 米并拍照。"
)


class MockSemanticService:
    """Deterministic semantic service for CPU smoke tests.

    It satisfies the contract of ``vlm_nav.semantic.SemanticNavigationService``
    without a VLM: search returns a fixed candidate, approach/goto/revisit move
    the world a little and report ``arrived``, confirm accepts, verify verifies.
    """

    def __init__(self, world=None) -> None:
        self.world = world
        self.calls: list[tuple[str, str]] = []

    def _reply(self, tool: str, **fields) -> dict:
        data = {
            "ok": True,
            "tool": tool,
            "status": "ok",
            "candidate_id": "",
            "task_id": "",
            "arrived": False,
            "accepted": False,
            "verified": False,
            "calls": 1,
            "error": "",
            "data": {},
        }
        data.update(fields)
        return data

    def _advance(self) -> None:
        position = getattr(self.world, "position_flu", None)
        if position is None:
            return
        try:
            position[0] += 0.5
        except Exception:  # noqa: BLE001 - mock movement is cosmetic
            pass

    def search(self, ctx, target, *, relation="", instruction=None,
               max_steps=8, max_turns=1) -> dict:
        self.calls.append(("search", str(target)))
        return self._reply(
            "search_target", candidate_id="cand_mock_1", status="unconfirmed"
        )

    def approach(self, ctx, candidate_id=None, *, standoff_m=1.2,
                 max_legs=8) -> dict:
        self.calls.append(("approach", str(candidate_id or "")))
        self._advance()
        return self._reply(
            "approach_candidate",
            candidate_id=str(candidate_id or ""),
            task_id="task_mock_1",
            arrived=True,
        )

    def confirm(self, ctx, candidate_id=None, *, instruction=None,
                relation=None, max_views=2) -> dict:
        self.calls.append(("confirm", str(candidate_id or "")))
        return self._reply(
            "confirm_candidate",
            candidate_id=str(candidate_id or ""),
            task_id="task_mock_1",
            arrived=True,
            accepted=True,
            status="accepted",
        )

    def verify(self, ctx, candidate_id=None, *, success_dist_m=None) -> dict:
        self.calls.append(("verify", str(candidate_id or "")))
        return self._reply(
            "verify_goal",
            candidate_id=str(candidate_id or ""),
            task_id="task_mock_1",
            arrived=True,
            accepted=True,
            verified=True,
            status="verified",
            data={"verified": True},
        )

    def revisit(self, ctx, *, position=None, observation_id=None,
                viewpoint_id=None, standoff_m=0.6) -> dict:
        self.calls.append(("revisit", str(position or observation_id or viewpoint_id)))
        self._advance()
        return self._reply(
            "return_to_observation", task_id="task_mock_1", arrived=True
        )

    def goto(self, ctx, position, *, standoff_m=0.6) -> dict:
        self.calls.append(("goto", str(position)))
        self._advance()
        return self._reply(
            "go_to_viewpoint", task_id="task_mock_1", arrived=True
        )


def _tuple3(values) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three numbers")
    return tuple(float(value) for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install, run and uninstall the mixed mission plugins."
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
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--height", type=int, default=54)
    parser.add_argument(
        "--bound-radius", type=float, default=20.0,
        help="circular scene bound; keep it above the default 8 m path",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--motion-only", action="store_true",
        help="no semantic tools/service; run a motion-only instruction",
    )
    mode.add_argument(
        "--semantic", choices=("mock", "real"), default=None,
        help="semantic backend: deterministic mock (default) or installed plugins",
    )
    parser.add_argument(
        "--instruction", default="",
        help="instruction to plan; defaults to the goal.md mixed example",
    )
    parser.add_argument(
        "--mission-json",
        help="run this mixed mission JSON file instead of planning an instruction",
    )
    parser.add_argument("--mission-id", default="")
    parser.add_argument(
        "--resume-state",
        help="resume this stored mixed mission state instead of starting a new mission",
    )
    parser.add_argument("--state-path", help="where to write the mixed mission state")
    parser.add_argument(
        "--artifacts-dir", default=str(REPO / "outputs" / "mixed_missions"),
        help="photos and state files (kept outside the plugin data dir)",
    )
    parser.add_argument(
        "--plugin-root", default=str(REPO / "outputs" / "mixed_plugins"),
        help="lifecycle host root (private plugin data/cache)",
    )
    parser.add_argument(
        "--out", default=str(REPO / "outputs" / "mixed_missions" / "run"),
        help="directory for summary.json",
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument(
        "--keep-installed", action="store_true",
        help="leave the plugins installed (skip deactivate/uninstall)",
    )
    parser.add_argument(
        "--purge-data", action="store_true",
        help="also remove the plugins' private data directories on uninstall",
    )
    return parser


def _load_mission_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _real_plugins(artifacts_dir: Path) -> list:
    """Build the legacy/semantic stack; raise ImportError when unavailable."""
    from vlm_nav.plugins.legacy import (  # noqa: PLC0415 - optional workstream
        NavigationPlugin,
        PerceptionPlugin,
        TaskToolsPlugin,
    )
    from vlm_nav.plugins.semantic import (  # noqa: PLC0415
        SemanticNavigationPlugin,
    )

    def construct(factory):
        try:
            return factory()
        except TypeError:
            return factory(vlm=None)

    return [
        construct(PerceptionPlugin),
        construct(NavigationPlugin),
        construct(TaskToolsPlugin),
        MotionPlugin(artifacts_dir=artifacts_dir),
        TaskPlannerPlugin(),
        SemanticNavigationPlugin(),
        MixedMissionPlugin(artifacts_dir=artifacts_dir),
    ]


def _motion_plugins(artifacts_dir: Path) -> list:
    return [
        MotionPlugin(artifacts_dir=artifacts_dir),
        TaskPlannerPlugin(),
        MixedMissionPlugin(artifacts_dir=artifacts_dir),
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_path) if args.state_path else (
        artifacts_dir / "mission.state.json"
    )

    motion_only = bool(args.motion_only)
    semantic_mode = "none" if motion_only else (args.semantic or "mock")
    if semantic_mode == "real":
        try:
            plugins = _real_plugins(artifacts_dir)
        except ImportError as exc:
            print(f"--semantic real unavailable: {exc}")
            return 1
    else:
        plugins = _motion_plugins(artifacts_dir)
    instruction = args.instruction or (
        DEFAULT_MOTION_INSTRUCTION if motion_only else DEFAULT_INSTRUCTION
    )

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
    context = ToolContext(
        world=world,
        episode_id="mixed_demo",
        budget={"tool_calls_left": int(args.budget)},
        extra={"mission.artifacts_dir": str(artifacts_dir)},
    )
    if semantic_mode == "mock":
        context.extra["mission.semantic_service"] = MockSemanticService(world)

    reports: dict = {
        "mode": semantic_mode,
        "manifests": [plugin.manifest.to_dict() for plugin in plugins],
        "install": [],
        "activate": [],
    }
    result = None
    state: MixedMissionState | None = None
    error = ""
    call_args = None
    try:
        for plugin in plugins:
            reports["install"].append(host.install(plugin).to_dict())
        for plugin in plugins:
            reports["activate"].append(host.activate(plugin.manifest.id).to_dict())

        if args.resume_state:
            call_args = {
                "resume": True,
                "state_path": str(args.resume_state),
                "artifacts_dir": str(artifacts_dir),
            }
        else:
            if args.mission_json:
                mission = _load_mission_json(args.mission_json)
            else:
                plan_result = registry.call(
                    "mission_plan", {"instruction": instruction}, context
                )
                reports["plan"] = {
                    "status": plan_result.status,
                    "data": plan_result.data,
                }
                if plan_result.status != "ok":
                    print(plan_result.prompt_text)
                    error = "planning failed"
                    mission = None
                else:
                    mission = plan_result.data
            if mission is not None:
                if motion_only:
                    step_list = (
                        mission.get("steps", [])
                        if isinstance(mission, dict)
                        else mission
                    )
                    semantic_steps = [
                        step.get("kind")
                        for step in step_list
                        if isinstance(step, dict)
                        and step.get("kind") not in MOTION_STEP_KINDS
                    ]
                    if semantic_steps:
                        error = (
                            "--motion-only instruction planned non-motion steps: "
                            + ", ".join(str(kind) for kind in semantic_steps)
                        )
                        print(error)
                        mission = None
            if mission is not None:
                if args.mission_id:
                    if isinstance(mission, dict):
                        mission["mission_id"] = args.mission_id
                    else:
                        mission = {
                            "steps": mission,
                            "mission_id": args.mission_id,
                        }
                call_args = {
                    "mission": mission,
                    "state_path": str(state_path),
                    "artifacts_dir": str(artifacts_dir),
                }
            else:
                call_args = None
        if call_args is not None:
            result = registry.call("mission_run", call_args, context)
            state = context.extra.get("mission.state")
            reports["mission"] = result.to_dict()
            if isinstance(state, MixedMissionState):
                reports["mission"]["state"] = state.to_dict()
    except Exception as exc:  # noqa: BLE001 - the script reports, never hides
        error = f"{type(exc).__name__}: {exc}"
        reports["error"] = error
        print(f"mixed mission failed: {error}")

    if not args.keep_installed:
        reports["deactivate"] = []
        reports["uninstall"] = []
        for plugin in reversed(plugins):
            plugin_id = plugin.manifest.id
            try:
                reports["deactivate"].append(
                    host.deactivate(plugin_id, force=True).to_dict()
                )
            except Exception as exc:  # noqa: BLE001
                reports["deactivate"].append({
                    "plugin_id": plugin_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            try:
                reports["uninstall"].append(
                    host.uninstall(plugin_id, purge_data=bool(args.purge_data)).to_dict()
                )
            except Exception as exc:  # noqa: BLE001
                reports["uninstall"].append({
                    "plugin_id": plugin_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        reports["catalog_after_uninstall"] = host.catalog()
        reports["artifacts_preserved"] = [
            str(path) for path in sorted(artifacts_dir.glob("*")) if path.is_file()
        ]

    summary = {
        "mode": semantic_mode,
        "instruction": instruction,
        "result": result.status if result is not None else "failed",
        "mission_status": state.status if state is not None else "",
        "position": [float(value) for value in world.position_flu],
        "yaw_deg": math.degrees(float(world.yaw_rad)),
        "error": error,
        "reports": reports,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if result is not None:
        print(result.prompt_text)
    else:
        print(error or "no mission result")
    print(f"position={summary['position']} yaw_deg={summary['yaw_deg']:.1f}")
    print(f"summary: {summary_path}")
    if not args.keep_installed:
        residual = sum(
            len(report.get("residual", []))
            for report in reports.get("uninstall", [])
            if isinstance(report, dict)
        )
        print(
            "lifecycle: "
            f"deactivated={len(reports.get('deactivate', []))} "
            f"uninstalled={len(reports.get('uninstall', []))} residual={residual}"
        )
    ok = (
        result is not None
        and result.status == "ok"
        and state is not None
        and state.status == "completed"
        and not error
    )
    if not args.keep_installed:
        ok = ok and all(
            report.get("residual", []) == []
            for report in reports.get("uninstall", [])
            if isinstance(report, dict)
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
