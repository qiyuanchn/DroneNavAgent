"""Structured navigation actions and robust parsing of VLM text output.

The VLM answers with one JSON object per decision:

```json
{"action": "move", "forward_m": 1.0, "right_m": 0.0, "up_m": 0.0, "yaw_deg": 0.0, "reason": "..."}
{"action": "turn", "yaw_deg": 30.0, "reason": "..."}
{"action": "done", "reason": "book is directly ahead within reach"}
{"action": "hover", "reason": "..."}
```

Magnitudes are clamped to per-decision limits before execution; anything
unparseable degrades to ``hover`` so the episode never crashes on a malformed
model response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re

MAX_TRANSLATE_M = 2.0
MAX_YAW_DEG = 45.0
MAX_WAYPOINT_TRANSLATE_M = 8.0
MAX_WAYPOINT_YAW_DEG = 180.0

_ACTION_KINDS = ("move", "turn", "done", "hover", "scan", "waypoint")

# Categorical motion primitives (more reliable for small VLMs than signed floats).
# Each maps to (kind, forward_m, right_m, up_m, yaw_deg) per unit of
# "meters" (translation primitives) or "degrees" (turn primitives).
_PRIMITIVE_ACTIONS = {
    "forward": ("move", 1.0, 0.0, 0.0, 0.0),
    "backward": ("move", -1.0, 0.0, 0.0, 0.0),
    "back": ("move", -1.0, 0.0, 0.0, 0.0),
    "left": ("move", 0.0, -1.0, 0.0, 0.0),
    "right": ("move", 0.0, 1.0, 0.0, 0.0),
    "up": ("move", 0.0, 0.0, 1.0, 0.0),
    "down": ("move", 0.0, 0.0, -1.0, 0.0),
    "turn_left": ("turn", 0.0, 0.0, 0.0, 1.0),
    "turn_right": ("turn", 0.0, 0.0, 0.0, -1.0),
}
_DEFAULT_MOVE_M = 0.8
_DEFAULT_TURN_DEG = 30.0


@dataclass(frozen=True)
class NavAction:
    kind: str = "hover"
    forward_m: float = 0.0
    right_m: float = 0.0
    up_m: float = 0.0
    yaw_deg: float = 0.0
    reason: str = ""
    raw: str = field(default="", repr=False)
    parse_ok: bool = True
    # point-to-navigate: target pixel in normalized [0,1000] image coordinates
    # (Qwen-style) plus an optional rough distance estimate in meters
    pixel: tuple[float, float] | None = None
    distance_m: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.kind == "done"


def _clamp(value: float, limit: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, value))


def _number(data: dict, key: str) -> float:
    value = data.get(key, 0.0)
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_json_object(text: str) -> dict | None:
    """Return the first decodable JSON object containing an 'action' key."""
    for match in re.finditer(r"\{", text):
        depth = 0
        for pos in range(match.start(), len(text)):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[match.start():pos + 1]
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(data, dict):
                        return data
                    break
    return None


def _parse_pixel(data: dict) -> tuple[float, float] | None:
    pixel = data.get("pixel")
    if isinstance(pixel, (list, tuple)) and len(pixel) == 2:
        try:
            px, py = float(pixel[0]), float(pixel[1])
            if math.isfinite(px) and math.isfinite(py):
                return (px, py)
        except (TypeError, ValueError):
            pass
    return None


def parse_action(text: str) -> NavAction:
    """Parse VLM output into a NavAction, clamped to per-decision limits."""
    data = _extract_json_object(text or "")
    if data is None:
        return NavAction(kind="hover", reason="unparseable model output",
                         raw=text or "", parse_ok=False)
    kind = str(data.get("action", "hover")).strip().lower()
    if kind in _PRIMITIVE_ACTIONS:
        base_kind, fwd, right, up, yaw_sign = _PRIMITIVE_ACTIONS[kind]
        # Primitives carry the same point-to-navigate fields as move/waypoint:
        # dropping the pixel would silently disable waypoint-style navigation.
        pixel = _parse_pixel(data)
        distance_m = _clamp(_number(data, "distance_m"), MAX_WAYPOINT_TRANSLATE_M)
        if base_kind == "move":
            meters = _clamp(abs(_number(data, "meters") or _number(data, "amount")
                                or _DEFAULT_MOVE_M), MAX_TRANSLATE_M)
            return NavAction(
                kind="move",
                forward_m=fwd * meters,
                right_m=right * meters,
                up_m=up * meters,
                pixel=pixel,
                distance_m=distance_m,
                reason=str(data.get("reason", ""))[:400],
                raw=text or "",
                parse_ok=True,
            )
        degrees = _clamp(abs(_number(data, "degrees") or _number(data, "yaw_deg")
                               or _DEFAULT_TURN_DEG), MAX_YAW_DEG)
        return NavAction(
            kind="turn",
            yaw_deg=yaw_sign * degrees,
            pixel=pixel,
            distance_m=distance_m,
            reason=str(data.get("reason", ""))[:400],
            raw=text or "",
            parse_ok=True,
        )
    if kind not in _ACTION_KINDS:
        kind = "move" if any(
            abs(_number(data, key)) > 0
            for key in ("forward_m", "right_m", "up_m", "yaw_deg")
        ) else "hover"
    if kind == "waypoint":
        return NavAction(
            kind=kind,
            forward_m=_clamp(_number(data, "forward_m"), MAX_WAYPOINT_TRANSLATE_M),
            right_m=_clamp(_number(data, "right_m"), MAX_WAYPOINT_TRANSLATE_M),
            up_m=_clamp(_number(data, "up_m"), MAX_WAYPOINT_TRANSLATE_M),
            yaw_deg=_clamp(_number(data, "yaw_deg"), MAX_WAYPOINT_YAW_DEG),
            pixel=_parse_pixel(data),
            distance_m=_clamp(_number(data, "distance_m"), MAX_WAYPOINT_TRANSLATE_M),
            reason=str(data.get("reason", ""))[:400],
            raw=text or "",
            parse_ok=True,
        )
    return NavAction(
        kind=kind,
        forward_m=_clamp(_number(data, "forward_m"), MAX_TRANSLATE_M),
        right_m=_clamp(_number(data, "right_m"), MAX_TRANSLATE_M),
        up_m=_clamp(_number(data, "up_m"), MAX_TRANSLATE_M),
        yaw_deg=_clamp(_number(data, "yaw_deg"), MAX_YAW_DEG),
        pixel=_parse_pixel(data),
        distance_m=_clamp(_number(data, "distance_m"), MAX_WAYPOINT_TRANSLATE_M),
        reason=str(data.get("reason", ""))[:400],
        raw=text or "",
        parse_ok=True,
    )


def action_to_body_delta(action: NavAction) -> tuple[float, float, float, float]:
    """Displacement (dx, dy, dz) in the yaw-only body frame plus dyaw in rad.

    FLU body frame: +x forward, +y left, +z up, so a "right" command is -y.
    """
    dx = action.forward_m
    dy = -action.right_m
    dz = action.up_m
    dyaw = math.radians(action.yaw_deg)
    return dx, dy, dz, dyaw


def body_delta_to_world(
    dx: float, dy: float, dz: float, yaw_rad: float
) -> tuple[float, float, float]:
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    return (
        cos_yaw * dx - sin_yaw * dy,
        sin_yaw * dx + cos_yaw * dy,
        dz,
    )


def parse_scan_choice(text: str, n_views: int) -> tuple[int | None, bool, float]:
    """Parse the look-around reply: {"view": int, "target_visible": bool, "distance_m": float}.

    Returns (view_index or None, target_visible, estimated distance or 0).
    """
    data = _extract_json_object(text or "")
    if data is None:
        return None, False, 0.0
    visible = bool(data.get("target_visible", True))
    distance = _number(data, "distance_m")
    try:
        view = int(data.get("view"))
    except (TypeError, ValueError):
        return None, visible, distance
    if not 0 <= view < n_views:
        return None, visible, distance
    return view, visible, distance
