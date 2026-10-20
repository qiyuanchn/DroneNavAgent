"""Task planner plugin: deterministic instruction planning as a service/tool.

The plugin owns no mission execution.  It provides ``task.planner@1`` so the
mixed-mission plugin can turn an instruction into a mission through the
service registry, and registers the ``plan_mission`` tool for the model/runtime.
"""

from __future__ import annotations

from typing import Any

from ..task_planner import TaskPlannerError, plan_mission
from ..tools import STATUS_FAILED, STATUS_OK, ToolContext, ToolResult, ToolSpec
from . import Plugin, PluginContext, PluginManifest

TASK_PLANNER_SERVICE = "task.planner@1"


class TaskPlannerPlugin(Plugin):
    """Installable natural-language -> mixed-mission planner."""

    manifest = PluginManifest(
        id="task.planner",
        version="0.1.0",
        provides=(TASK_PLANNER_SERVICE,),
        tools=("plan_mission",),
    )

    def __init__(
        self,
        *,
        vlm: Any = None,
        default_target: str = "",
    ) -> None:
        self.vlm = vlm
        self.default_target = str(default_target or "")
        self._ctx: PluginContext | None = None

    # -- lifecycle -------------------------------------------------------- #

    def activate(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        ctx.services.provide(self.manifest.id, TASK_PLANNER_SERVICE, self)
        ctx.tools.register(ToolSpec(
            name="plan_mission",
            description=(
                "Plan a natural-language drone instruction into an ordered "
                "drone.mixed_mission/1 mission ({mission_id, instruction, steps}). "
                "Motion steps are move/turn/hover/photo; semantic steps are "
                "search/approach/confirm/verify/revisit/goto. The deterministic "
                "grammar supports Chinese and English; an unrecognized clause is "
                "reported instead of being guessed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "default_target": {"type": "string"},
                },
                "required": ["instruction"],
            },
            handler=self._tool_plan,
            side_effects=False,
            cacheable=False,
        ))

    def deactivate(self, ctx: PluginContext) -> None:
        self._ctx = None

    # -- task.planner@1 contract ------------------------------------------ #

    def plan_mission(
        self, instruction: str, *, default_target: str = ""
    ):
        """Return a :class:`MixedMissionSpec` for ``instruction``."""
        return plan_mission(
            instruction,
            vlm=self.vlm,
            default_target=default_target or self.default_target,
        )

    # ``plan`` is a convenience alias for consumers that prefer a short name.
    def plan(self, instruction: str, *, default_target: str = ""):
        return self.plan_mission(instruction, default_target=default_target)

    # -- tools ------------------------------------------------------------ #

    def _tool_plan(self, ctx: ToolContext, args: dict) -> ToolResult:
        instruction = str(args.get("instruction", "") or "")
        default_target = str(args.get("default_target", "") or "")
        try:
            spec = self.plan_mission(
                instruction, default_target=default_target
            )
        except TaskPlannerError as exc:
            return ToolResult(
                STATUS_FAILED,
                {"error": str(exc), "instruction": instruction},
            )
        data = spec.to_dict()
        lines = [
            f"[ok] mission {spec.mission_id}: {len(spec.steps)} steps",
            f"instruction: {spec.instruction}",
        ]
        for index, step in enumerate(spec.steps):
            detail = ", ".join(
                f"{key}={value!r}"
                for key, value in step.items()
                if key != "kind"
            )
            lines.append(f"  {index + 1}. {step['kind']} {detail}".rstrip())
        return ToolResult(STATUS_OK, data, prompt_text="\n".join(lines))
