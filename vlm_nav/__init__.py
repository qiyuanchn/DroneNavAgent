"""Lightweight VLM-in-the-loop semantic navigation harness for 3DGS scenes.

Submodules:
- cpu_renderer: pure-torch CPU 3DGS splat renderer (no CUDA required), matching
  the camera conventions of the production HiGS sensor.
- actions: structured action schema + robust parsing of VLM text output.
- prompts: system/user prompt construction.
- vlm: pluggable VLM backends (local HF, OpenAI-compatible API, mock).
- world: sim worlds (CPU splat world / HiGS GPU world / production env world)
    with pose control and 360-degree scan support.
- harness: the perception-decision-action episode loop.
"""

from .actions import NavAction, parse_action

__all__ = ["NavAction", "parse_action"]
