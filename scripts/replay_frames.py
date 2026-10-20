"""Frozen observation replay sets for comparable resource measurements.

A replay set is a directory with a ``manifest.json`` and the PNG frames it
lists. It freezes exactly what a measurement must not let drift between hosts
or harness variants:

- the pixels (the frame file and its SHA-256),
- where each frame was taken (``pose_flu`` + ``yaw_deg`` + ``pitch_deg``),
- the camera model that maps its pixels to rays
  (``width``/``height``/``hfov_deg`` and the derived ``fx``/``fy``/``cx``/``cy``).

The loader validates all of that and raises :class:`ReplayError` instead of
degrading a benchmark silently: too few views, an unknown schema, a missing or
outside-the-root frame, a non-finite pose, or a checksum mismatch is a load
error. Nothing here imports torch/transformers at module import time, so the
contract is testable on a CPU-only host; :meth:`ReplayView.load_frame` imports
torch lazily.

Manifest (``schema: vlm_nav.replay.v1``)::

    {
      "schema": "vlm_nav.replay.v1",
      "name": "window_chair",
      "instruction": "在卧室窗边找到木椅",
      "camera": {"width": 320, "height": 180, "hfov_deg": 86.0},
      "scene": {...},
      "generator": {...},
      "views": [
        {"id": "view_00", "file": "frames/view_00.png",
         "pose_flu": [x, y, z], "yaw_deg": ..., "pitch_deg": ...,
         "phase": "scan", "sha256": "...",
         "intrinsics": {"width": 320, "height": 180, "hfov_deg": 86.0,
                        "fx": ..., "fy": ..., "cx": ..., "cy": ...}}
      ]
    }

A per-view ``intrinsics`` mapping is optional; when absent the top-level camera
is used. A view that was rendered with a different camera carries its own copy,
so a harness variant cannot silently resize/reproject it with the wrong model.

The benchmark-facing helpers are :func:`load_replay`,
:meth:`ReplaySet.frames` (a ``[N, 3, H, W]`` float tensor) and
:meth:`ReplaySet.pil_frames` (what a VLM backend's image path receives).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPLAY_SCHEMA = "vlm_nav.replay.v1"
MIN_VIEWS = 12
MANIFEST_NAME = "manifest.json"

_INTRINSICS_KEYS = ("width", "height", "hfov_deg", "fx", "fy", "cx", "cy")


class ReplayError(ValueError):
    """A replay set that cannot be used as a comparable measurement input."""


def _finite_float(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"{what} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise ReplayError(f"{what} is not finite: {value!r}")
    return number


def _positive_int(value: Any, what: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"{what} is not an integer: {value!r}") from exc
    if number <= 0:
        raise ReplayError(f"{what} must be positive, got {number}")
    return number


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayError(f"{what} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayCamera:
    """The pinhole model a replay frame's pixels were rendered with."""

    width: int
    height: int
    hfov_deg: float
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, what: str = "camera") -> "ReplayCamera":
        data = _mapping(value, what)
        width = _positive_int(data.get("width"), f"{what}.width")
        height = _positive_int(data.get("height"), f"{what}.height")
        hfov = _finite_float(data.get("hfov_deg"), f"{what}.hfov_deg")
        if not 0.0 < hfov < 180.0:
            raise ReplayError(f"{what}.hfov_deg must be in (0, 180), got {hfov}")
        fx = _finite_float(data.get("fx"), f"{what}.fx") if data.get("fx") is not None else (
            width / (2.0 * math.tan(math.radians(hfov) / 2.0))
        )
        fy = _finite_float(data.get("fy"), f"{what}.fy") if data.get("fy") is not None else fx
        cx = _finite_float(data.get("cx"), f"{what}.cx") if data.get("cx") is not None else width / 2.0
        cy = _finite_float(data.get("cy"), f"{what}.cy") if data.get("cy") is not None else height / 2.0
        if fx <= 0.0 or fy <= 0.0:
            raise ReplayError(f"{what} focal lengths must be positive, got fx={fx}, fy={fy}")
        return cls(
            width=int(width), height=int(height), hfov_deg=float(hfov),
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        )

    def intrinsics(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "hfov_deg": self.hfov_deg,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "hfov_deg": self.hfov_deg}


@dataclass(frozen=True)
class ReplayView:
    """One frozen frame plus the pose and intrinsics behind it."""

    view_id: str
    path: Path
    pose_flu: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    phase: str
    index: int
    camera: ReplayCamera
    sha256: str
    note: str = ""
    root: Path = field(default=Path("."), repr=False)

    @property
    def id(self) -> str:
        return self.view_id

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    @property
    def pitch_rad(self) -> float:
        """Body pitch, nose-down positive (the world/camera convention)."""
        return math.radians(self.pitch_deg)

    @property
    def intrinsics(self) -> dict:
        return self.camera.intrinsics()

    def pil_frame(self, image_size: Sequence[int] | None = None):
        """The frozen image as RGB PIL; optional ``(width, height)`` resize."""
        from PIL import Image

        with Image.open(self.path) as image:
            frame = image.convert("RGB")
        if image_size is not None:
            width, height = (int(v) for v in image_size)
            if width <= 0 or height <= 0:
                raise ReplayError(f"image_size must be positive, got {image_size!r}")
            if frame.size != (width, height):
                frame = frame.resize((width, height), Image.BILINEAR)
        return frame

    def load_frame(self, image_size: Sequence[int] | None = None):
        """``[3, H, W]`` float tensor in ``[0, 1]`` (CHW, the harness contract)."""
        import numpy as np
        import torch

        image = self.pil_frame(image_size=image_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def to_dict(self) -> dict:
        return {
            "id": self.view_id,
            "file": str(self.path.relative_to(self.root)) if self.path.is_relative_to(self.root) else str(self.path),
            "pose_flu": list(self.pose_flu),
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "phase": self.phase,
            "index": self.index,
            "sha256": self.sha256,
            "intrinsics": self.intrinsics,
        }


class ReplaySet:
    """A validated replay directory (frames + manifest)."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], views: list[ReplayView]):
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.views = list(views)
        self.name = str(manifest.get("name") or self.root.name)
        self.instruction = str(manifest.get("instruction") or "")
        self.scene = dict(manifest.get("scene") or {})
        self.generator = dict(manifest.get("generator") or {})
        self.benchmark = dict(manifest.get("benchmark") or {})
        self.camera = ReplayCamera.from_mapping(manifest["camera"])
        self.manifest_sha256 = _sha256(self.root / MANIFEST_NAME)

    def __len__(self) -> int:
        return len(self.views)

    @property
    def frame_count(self) -> int:
        return len(self.views)

    def view(self, view_id: str) -> ReplayView:
        for view in self.views:
            if view.view_id == view_id:
                return view
        raise KeyError(f"no replay view {view_id!r}")

    def views_in_phase(self, phase: str) -> list[ReplayView]:
        return [view for view in self.views if view.phase == phase]

    def phases(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for view in self.views:
            counts[view.phase] = counts.get(view.phase, 0) + 1
        return counts

    def frames(self, image_size: Sequence[int] | None = None):
        """Stack every view into one ``[N, 3, H, W]`` float tensor."""
        import torch

        loaded = [view.load_frame(image_size=image_size) for view in self.views]
        return torch.stack(loaded, dim=0)

    def pil_frames(self, image_size: Sequence[int] | None = None) -> list:
        return [view.pil_frame(image_size=image_size) for view in self.views]

    def pil_frame(self, view: str | ReplayView, image_size: Sequence[int] | None = None):
        """One view's RGB image; accepts an id or a :class:`ReplayView`."""
        resolved = self.view(view) if isinstance(view, str) else view
        return resolved.pil_frame(image_size=image_size)

    def load_frame(self, view: str | ReplayView, image_size: Sequence[int] | None = None):
        resolved = self.view(view) if isinstance(view, str) else view
        return resolved.load_frame(image_size=image_size)

    def summary(self) -> dict:
        return {
            "schema": REPLAY_SCHEMA,
            "name": self.name,
            "instruction": self.instruction,
            "frame_count": len(self.views),
            "phases": self.phases(),
            "camera": self.camera.intrinsics(),
            "scene": dict(self.scene),
            "generator": dict(self.generator),
            "benchmark": dict(self.benchmark),
            "manifest_sha256": self.manifest_sha256,
            "views": [view.view_id for view in self.views],
        }

    def to_manifest(self) -> dict:
        return {
            "schema": REPLAY_SCHEMA,
            "name": self.name,
            "instruction": self.instruction,
            "camera": self.camera.to_dict(),
            "scene": dict(self.scene),
            "generator": dict(self.generator),
            "benchmark": dict(self.benchmark),
            "views": [
                {key: value for key, value in view.to_dict().items()
                 if key not in ("index", "intrinsics")}
                for view in self.views
            ],
        }


def _resolve_frame(root: Path, relative: Any, view_id: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ReplayError(f"view {view_id!r} has no file")
    raw = Path(relative)
    if raw.is_absolute():
        raise ReplayError(f"view {view_id!r} file must be relative to the replay root: {relative!r}")
    root_resolved = root.resolve()
    candidate = (root / raw).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ReplayError(
            f"view {view_id!r} file resolves outside the replay root: {relative!r}"
        )
    if not candidate.is_file():
        raise ReplayError(f"view {view_id!r} frame is missing: {candidate}")
    return candidate


def load_replay(
    path: str | Path,
    *,
    verify_checksums: bool = True,
    min_views: int = MIN_VIEWS,
) -> ReplaySet:
    """Load and validate a replay directory (or its ``manifest.json``)."""
    source = Path(path).expanduser()
    if source.is_file() and source.name == MANIFEST_NAME:
        root = source.parent
        manifest_path = source
    else:
        root = source
        manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ReplayError(f"no replay manifest at {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read replay manifest {manifest_path}: {exc}") from exc
    manifest = _mapping(manifest, "manifest")
    schema = manifest.get("schema")
    if schema != REPLAY_SCHEMA:
        raise ReplayError(
            f"unknown replay schema {schema!r}; expected {REPLAY_SCHEMA!r}"
        )
    instruction = manifest.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ReplayError("replay.instruction must be a non-empty string")
    camera = ReplayCamera.from_mapping(_mapping(manifest.get("camera"), "camera"))
    entries = manifest.get("views")
    if not isinstance(entries, list):
        raise ReplayError("replay.views must be a list")
    if len(entries) < int(min_views):
        raise ReplayError(
            f"replay has {len(entries)} views; at least {int(min_views)} are required"
        )

    views: list[ReplayView] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        entry = _mapping(entry, f"views[{index}]")
        view_id = str(entry.get("id") or f"view_{index:02d}")
        if view_id in seen_ids:
            raise ReplayError(f"duplicate replay view id {view_id!r}")
        seen_ids.add(view_id)
        pose = entry.get("pose_flu")
        if not isinstance(pose, (list, tuple)) or len(pose) != 3:
            raise ReplayError(f"view {view_id!r} needs a 3-vector pose_flu")
        pose_flu = tuple(
            _finite_float(value, f"view {view_id!r} pose_flu[{axis}]")
            for axis, value in enumerate(pose)
        )
        yaw_deg = _finite_float(entry.get("yaw_deg", 0.0), f"view {view_id!r} yaw_deg")
        pitch_deg = _finite_float(entry.get("pitch_deg", 0.0), f"view {view_id!r} pitch_deg")
        if abs(pitch_deg) >= 90.0:
            raise ReplayError(
                f"view {view_id!r} pitch_deg must be in (-90, 90), got {pitch_deg}"
            )
        view_camera = camera
        if entry.get("intrinsics") is not None:
            view_camera = ReplayCamera.from_mapping(
                entry["intrinsics"], what=f"view {view_id!r} intrinsics"
            )
        file_path = _resolve_frame(root, entry.get("file"), view_id)
        digest = _sha256(file_path)
        expected = entry.get("sha256")
        if verify_checksums and expected:
            if str(expected).lower() != digest:
                raise ReplayError(
                    f"view {view_id!r} sha256 mismatch: manifest {expected}, file {digest}"
                )
        views.append(
            ReplayView(
                view_id=view_id,
                path=file_path,
                pose_flu=pose_flu,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                phase=str(entry.get("phase") or "unspecified"),
                index=int(entry.get("index", index)),
                camera=view_camera,
                sha256=digest,
                note=str(entry.get("note") or ""),
                root=root,
            )
        )
    return ReplaySet(root, manifest, views)


def replay_frame_paths(replay: ReplaySet) -> list[Path]:
    """Frame paths in manifest order (for tools that only need the files)."""
    return [view.path for view in replay.views]


def frame_list(replay: ReplaySet, *, image_size: Sequence[int] | None = None) -> list:
    """Every frame as a CHW float tensor (VLM ``decide`` accepts the list)."""
    return [view.load_frame(image_size=image_size) for view in replay.views]


def iter_view_batches(views: Iterable[ReplayView], size: int) -> Iterable[list[ReplayView]]:
    batch: list[ReplayView] = []
    for view in views:
        batch.append(view)
        if len(batch) >= int(size):
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = [
    "MANIFEST_NAME",
    "MIN_VIEWS",
    "REPLAY_SCHEMA",
    "ReplayCamera",
    "ReplayError",
    "ReplaySet",
    "ReplayView",
    "frame_list",
    "iter_view_batches",
    "load_replay",
    "replay_frame_paths",
]
