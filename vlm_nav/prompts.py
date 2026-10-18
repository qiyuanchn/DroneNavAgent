"""Prompt construction for the semantic-navigation VLM.

The model sees exactly one first-person RGB frame per decision plus a compact
text state (instruction, step budget, recent action history). It never sees
ground-truth target coordinates.
"""

from __future__ import annotations

from typing import Iterable

SYSTEM_PROMPT = """You are the navigation controller of a camera drone flying inside a \
photo-realistic 3D scene. At every step you receive ONE first-person RGB image \
from the drone's forward camera and a navigation instruction. Decide the next \
short motion that progresses toward the goal.

Reply with EXACTLY ONE JSON object and nothing else:
{"action": "<action>", "meters": <float>, "degrees": <float>, "reason": "<short>"}

Actions:
- "forward" / "backward": fly forward/backward by "meters" (0.2 to 2.0).
- "left" / "right": strafe left/right by "meters".
- "up" / "down": fly up/down by "meters".
- "turn_left" / "turn_right": rotate in place by "degrees" (10 to 45).
- "done": the target is clearly visible and within about 1-2 meters; you MUST \
include "pixel": [x, y] pointing at the target so the claim can be verified.
- WHATEVER action you choose, whenever the target is visible in the image, \
ALWAYS include "pixel": [x, y] of the target and "distance_m" - the autopilot \
locks onto the pointed point and steers precisely toward it.
- "hover": stay in place.
- "scan": rotate in place and look around again to re-locate the target; use it \
when you are lost, when the target left your view, or after entering a new area.
- "waypoint": fly all the way to a body-relative offset you estimate, \
{"action": "waypoint", "forward_m": <-8..8>, "right_m": <-8..8>, "up_m": <-8..8>, "yaw_deg": <-180..180>}. \
Prefer adding "pixel": [x, y] (the point you want to fly toward, in normalized \
0-1000 image coordinates) plus "distance_m" (rough distance) instead of \
guessing forward/right meters yourself - pointing at the image is more accurate. \
Use waypoint when you can estimate where the target (or a good exploration \
point) is; the lower-level planner flies the whole leg and avoids obstacles. \
One long "waypoint" is better than many small "forward" steps.

Examples:
{"action": "turn_left", "degrees": 30, "reason": "scanning the room for the table"}
{"action": "forward", "meters": 1.0, "reason": "book is straight ahead"}

Rules:
- The camera looks forward: use "forward" to approach what you see.
- Prefer one long "waypoint" over many small "forward" steps when the path ahead looks clear.
- If the target is not visible, keep turning in the SAME direction with 30-45 degrees per step until it appears; do NOT reverse the scan direction.
- If the target is visible but off-center, turn until it is near the image center, then move "forward".
- Prefer small moves (<= 1 m) when obstacles are near; never fly into walls, floors, or furniture.
- Reply "done" ONLY when the target is clearly visible and within about 1-2 meters.
- Never output anything except the JSON object."""


SCAN_SYSTEM_PROMPT = """You are the navigation controller of a camera drone. The drone has \
just rotated in place and you see the views it captured, in order, one per equal \
heading step (view 0 is the initial heading; headings increase by the same angle \
per view, covering a full circle).

Reply with EXACTLY ONE JSON object and nothing else:
{"view": <int>, "target_visible": <true|false>, "distance_m": <float>, "reason": "<short>"}

- "view": index of the view that shows the target best, or the view whose heading \
points closest to where the target is most likely to be.
- "target_visible": true only if you can actually see the target object in that view.
- "distance_m": if the target is visible, your estimate of how far away it is in \
meters; otherwise 0.
- Never output anything except the JSON object."""


VERIFY_SYSTEM_PROMPT = """You verify navigation targets for a camera drone. You receive one \
zoomed image crop around a point that was marked as the target, plus the \
original instruction. Reply with EXACTLY ONE JSON object and nothing else:
{"verified": true, "reason": "<short>"}  if the crop clearly shows the \
described target object,
{"verified": false, "reason": "<short>"} if it does not or you cannot tell."""


def build_scan_prompt(instruction: str, n_views: int, n_pitch: int = 0) -> str:
    text = (
        f"Navigation instruction: {instruction}\n\n"
        f"There are {n_views} views (index 0 to {n_views - 1}), taken while rotating "
        "in place over a full circle."
    )
    if n_pitch > 0:
        text += (
            f" The last {n_pitch} views (index {n_views - n_pitch} to {n_views - 1}) "
            "look downward: they show what is BELOW the drone's flight level."
        )
    return text + " Which view should the drone face next?"


def format_history(history: Iterable[dict], limit: int = 4) -> str:
    entries = list(history)[-limit:]
    if not entries:
        return "none"
    lines = []
    for item in entries:
        lines.append(
            f"step {item['step']}: {item['action_summary']} -> {item['outcome']}"
        )
    return "\n".join(lines)


def build_user_prompt(
    instruction: str,
    step: int,
    max_steps: int,
    history: Iterable[dict],
    last_feedback: str | None = None,
    odometry: str | None = None,
    n_frames: int = 1,
    bev: bool = False,
    candidates_text: str | None = None,
    keyframe: bool = False,
) -> str:
    parts = [
        f"Navigation instruction: {instruction}",
        f"Step {step} of {max_steps} (each step is one short motion).",
    ]
    if bev:
        parts.append(
            "Image 1 is a top-down map of your flight so far, oriented with your "
            "current heading pointing to the TOP of the image (like a minimap): "
            "gray = observed obstacles/walls, green = your flown path, blue tick "
            "= you at the center facing up. This map does NOT show where the "
            "target is: a red dot appears only at a target estimate you yourself "
            "marked earlier by pointing at it, so no red dot means you have not "
            "seen the target yet."
        )
        if n_frames > 1:
            parts.append(
                f"The remaining {n_frames} images are your recent first-person "
                "camera frames (oldest first, newest last)."
            )
    elif n_frames > 1:
        parts.append(
            f"You are looking at the last {n_frames} camera frames in order "
            "(oldest first, newest last). Use them to remember where you have looked."
        )
    if keyframe:
        parts.append(
            "The oldest camera image is pinned: it is the frame where you last "
            "pointed at the target."
        )
    if odometry:
        parts.append(odometry)
    if candidates_text:
        parts.append(f"Tracked target candidates (your past sightings): {candidates_text}")
    parts.append(f"Recent actions:\n{format_history(history)}")
    if last_feedback:
        parts.append(f"Note: {last_feedback}")
    parts.append("Output the next action as one JSON object.")
    return "\n\n".join(parts)


def summarize_action(action) -> str:
    if action.kind == "done":
        return f"done ({action.reason})"
    if action.kind == "hover":
        return f"hover ({action.reason})"
    if action.kind == "scan":
        return f"scan ({action.reason})"
    bits = []
    if action.forward_m:
        bits.append(f"fwd {action.forward_m:+.2f}m")
    if action.right_m:
        bits.append(f"right {action.right_m:+.2f}m")
    if action.up_m:
        bits.append(f"up {action.up_m:+.2f}m")
    if action.yaw_deg:
        bits.append(f"yaw {action.yaw_deg:+.0f}deg")
    return f"{action.kind} " + ", ".join(bits or ["0"])
