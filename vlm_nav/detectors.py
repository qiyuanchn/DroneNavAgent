"""Optional detection/segmentation backends for ``ground_target``.

The VLM grounds an instruction into a box; this module supplies an *independent*
instance proposal for the same frame. ``ground_target`` uses it as the authority
on instance identity when the backend is available, and falls back to the VLM
reply when it is not. The backend is intentionally optional: importing this
module never imports torch/transformers, and :class:`NullDetector` is the default.

The first backend is SAM3 (text-conditioned masks) + DINOv2 (crop features):

- SAM3 turns a short English object phrase into class-agnostic masks and boxes.
  The phrase is derived from the Chinese instruction by :func:`target_labels`,
  which keeps the *target* phrase of a relation ("书架旁的课桌" -> desk) and
  never the anchor.
- DINOv2 embeds the VLM box crop and every proposal crop; ``ground_target``
  uses the cosine similarity as the description/feature consistency half of the
  gate and the geometric IoU/containment as the other half.

Everything is lazy: weights load on the first real ``detect`` call on a CUDA
device, results are cached per (frame, labels), and a missing checkpoint,
missing dependency, absent GPU or a raised exception is reported as an
unavailable/error backend rather than a failed tool call.

Command-line forms (the detection bench and the tool runner share them)::

    off / none            no backend (the default, the VLM-only behaviour)
    sam3 / sam3_dino      :class:`Sam3DinoDetector`
    module:factory        any importable factory returning a detector backend

A backend implements ``available()``, ``detect(frame, instruction, labels=None)``
and optionally ``embed_boxes(frame, boxes)``. ``frame`` is the CHW float tensor
the perception tools already hold (0-1 or 0-255); boxes are canonical
``[x1, y1, x2, y2]`` normalized to 0-1000, the same contract as the candidates.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: checkpoint defaults (the container mounts ``/home/dataset-assist-0/models``
#: at ``/models_ro``; the second path is the repository host's own layout).
SAM3_DEFAULT_PATH = "/models_ro/sam3"
DINO_DEFAULT_PATH = "/models_ro/dinov2-base"
#: SAM3 score below which a mask is not a proposal at all. Real objects on the
#: 3DGS frames score ~0.7-0.9; a wall/floor frame scores ~0.01-0.1.
DEFAULT_MIN_SCORE = 0.2
#: A proposal covering more than this fraction of the frame is the backdrop,
#: not an instance (SAM3 emits a whole-frame mask on textureless walls).
DEFAULT_MAX_AREA_FRAC = 0.85
#: At most this many proposals per call, and this many text prompts.
DEFAULT_MAX_PROPOSALS = 8
DEFAULT_MAX_LABELS = 2
#: DINOv2 cosine similarity below which a geometric overlap is not the same
#: instance. Same-object crops score well above this; neighbouring objects and
#: background patches fall below it.
DEFAULT_DINO_MIN = 0.25


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """One detector proposal in the candidate box contract."""

    box: list[float]
    confidence: float | None = None
    label: str = ""
    mask: Any = None
    mask_area_frac: float | None = None
    dino_embedding: Any = None

    def to_dict(self) -> dict:
        return {
            "box": [round(float(value), 1) for value in self.box],
            "confidence": None if self.confidence is None else round(float(self.confidence), 3),
            "label": str(self.label),
            "has_mask": self.mask is not None,
            "mask_area_frac": (
                None
                if self.mask_area_frac is None
                else round(float(self.mask_area_frac), 3)
            ),
            "has_dino": self.dino_embedding is not None,
        }


class DetectorBackend:
    """Protocol/slot: ``available()`` then ``detect``; never raises into a tool."""

    name = "base"
    note = ""

    def available(self) -> bool:
        return False

    def detect(
        self,
        frame: Any,
        instruction: str,
        *,
        labels: Sequence[str] | None = None,
        context: Mapping | None = None,
    ) -> list[Detection]:
        raise NotImplementedError

    def embed_boxes(self, frame: Any, boxes: Sequence[Sequence[float]]):
        """Optional crop embeddings for the VLM boxes (``None`` when unsupported)."""
        return None

    def probe(self) -> dict:
        """CPU-only readiness report: files, imports, device. No weights loaded."""
        return {"name": self.name, "available": bool(self.available())}


class NullDetector(DetectorBackend):
    """The empty slot: the VLM-only path, unchanged."""

    name = "null"
    note = "no detection backend configured (VLM-only grounding)"

    def available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# instruction -> English object phrases
# ---------------------------------------------------------------------------

#: Specific phrases before generic ones: both may be tried when the budget
#: allows, and the generic phrase is the fallback for a look-alike object.
_LABEL_GLOSSARY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wooden chair", ("木椅", "木制椅", "木质椅")),
    ("office chair", ("办公椅", "电脑椅")),
    ("armchair", ("扶手椅", "沙发椅")),
    ("chair", ("椅子", "靠背椅", "椅")),
    ("bookshelf", ("书架", "书柜")),
    ("desk", ("课桌", "书桌")),
    ("table", ("桌子", "桌")),
    ("plastic bottle", ("塑料瓶", "水瓶", "瓶子")),
    ("bottle", ("瓶",)),
    ("cabinet", ("柜子", "柜")),
    ("lamp", ("台灯", "落地灯", "灯")),
    ("door", ("门",)),
    ("window", ("窗",)),
    ("bed", ("床",)),
    ("sofa", ("沙发",)),
    ("monitor", ("显示器", "屏幕")),
    ("cardboard box", ("纸箱", "纸盒")),
    ("trash can", ("垃圾桶",)),
    ("potted plant", ("盆栽", "花盆")),
)

_EN_LABELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wooden chair", ("wooden chair",)),
    ("office chair", ("office chair",)),
    ("chair", ("chair",)),
    ("bookshelf", ("bookshelf", "bookcase")),
    ("desk", ("desk",)),
    ("table", ("table",)),
    ("plastic bottle", ("plastic bottle",)),
    ("bottle", ("bottle",)),
    ("cabinet", ("cabinet",)),
    ("lamp", ("lamp", "light")),
    ("door", ("door",)),
    ("window", ("window",)),
    ("bed", ("bed",)),
    ("sofa", ("sofa", "couch")),
    ("monitor", ("monitor", "screen")),
    ("cardboard box", ("cardboard box", "box")),
)

#: The target of a relation instruction is the phrase *after* the relatum:
#: "书架旁的课桌" grounds the desk, not the bookshelf.
_TARGET_SPLIT = (
    "旁边", "旁", "边上", "边", "附近", "左侧", "右侧", "前面", "后面",
    "next to", "near", "beside", "left of", "right of",
)
_COLOR_WORDS = {
    "black": ("黑", "black"), "white": ("白", "white"), "brown": ("棕", "褐", "brown"),
    "red": ("红", "red"), "blue": ("蓝", "blue"), "green": ("绿", "green"),
    "gray": ("灰", "gray", "grey"), "yellow": ("黄", "yellow"),
}
_MATERIAL_WORDS = {
    "wood": ("木", "wood", "wooden"), "metal": ("金属", "metal"),
    "plastic": ("塑料", "plastic"), "glass": ("玻璃", "glass"),
    "leather": ("皮革", "leather"), "fabric": ("布", "fabric"),
}


def _target_phrase(instruction: str) -> str:
    text = str(instruction or "").strip()
    lowered = text.lower()
    cut = -1
    for token in _TARGET_SPLIT:
        position = lowered.rfind(token.lower())
        if position >= 0:
            cut = max(cut, position + len(token))
    phrase = text[cut:] if cut >= 0 else text
    return phrase.strip("的 地 得 （）()：:，,。.;；")


def _found_labels(text: str, glossary) -> list[str]:
    lowered = str(text or "").lower()
    found: list[str] = []
    for label, words in glossary:
        if any(word.lower() in lowered for word in words):
            found.append(label)
    return found


def _found_labels_latest_first(text: str, glossary) -> list[str]:
    """Glossary labels ordered by their last mention (the current subgoal)."""
    lowered = str(text or "").lower()
    ranked: list[tuple[int, str]] = []
    for label, words in glossary:
        positions = [lowered.rfind(word.lower()) for word in words]
        position = max(positions) if positions else -1
        if position >= 0:
            ranked.append((position, label))
    ranked.sort(reverse=True)
    return [label for _position, label in ranked]


def target_labels(
    instruction: str,
    candidates: Iterable[Mapping] | None = None,
    *,
    max_labels: int = DEFAULT_MAX_LABELS,
) -> list[str]:
    """Short English SAM3 prompts for an instruction (target phrase first).

    The relation anchor is dropped (``书架旁的课桌`` -> ``desk``). When the
    instruction names a colour/material and a candidate carries them, one
    attribute-qualified variant is added (``black office chair``), because the
    downstream instance gate treats a detector proposal conditioned on the
    attribute as verification of that attribute.
    """
    phrase = _target_phrase(instruction)
    labels = _found_labels(phrase, _LABEL_GLOSSARY)
    if not labels:
        # a recovery call may carry the *whole* multi-stage instruction, where
        # the last relation split is filler rather than the current target:
        # fall back to the known object phrase mentioned last (the current
        # subgoal), then the earlier ones
        labels = _found_labels_latest_first(str(instruction or ""), _LABEL_GLOSSARY)
    if not labels:
        # an English instruction: its noun is a usable prompt. A Chinese phrase
        # the glossary does not know is not translated by guessing - returning
        # nothing keeps the backend from proposing an unrelated object.
        english = _found_labels(str(instruction or ""), _EN_LABELS)
        if english:
            labels = english
        elif re.fullmatch(r"[A-Za-z0-9 ,'\-]+", phrase) and len(phrase) <= 48:
            labels = [phrase]
    attrs: list[str] = []
    # the instruction's own colour/material qualifies the prompt even when the
    # VLM reply is empty or omitted attributes
    from_phrase = [
        canonical
        for canonical, words in _COLOR_WORDS.items()
        if any(word in phrase.lower() for word in words)
    ]
    from_phrase += [
        canonical if canonical != "wood" else "wooden"
        for canonical, words in _MATERIAL_WORDS.items()
        if any(word in phrase.lower() for word in words)
    ]
    if labels and from_phrase:
        qualifier = [
            item for item in from_phrase if item.lower() not in labels[0].lower()
        ]
        if qualifier:
            attrs.append(f"{' '.join(qualifier)} {labels[0]}")
    for candidate in list(candidates or [])[:3]:
        values = candidate.get("attrs") if isinstance(candidate, Mapping) else None
        if not isinstance(values, Mapping):
            continue
        parts = []
        lowered = f"{phrase} {instruction}".lower()
        for key, value in values.items():
            text = str(value or "").strip().lower()
            if not text or len(text) > 24:
                continue
            if key in ("color", "colour") and any(
                word in lowered for word in _COLOR_WORDS.get(text, ())
            ):
                parts.append(text)
            elif key in ("material",) and any(
                word in lowered for word in _MATERIAL_WORDS.get(text, ())
            ):
                parts.append(text if text != "wood" else "wooden")
            elif key in ("shape", "type", "category") and text in labels:
                parts.append("")
        if parts and labels:
            qualifier = [
                part for part in parts if part and part.lower() not in labels[0].lower()
            ]
            if qualifier:
                attrs.append(f"{' '.join(qualifier)} {labels[0]}")
    # attribute-qualified prompts lead: "black office chair" is more specific
    # than the generic "office chair" and costs the same single forward
    labels = attrs + labels
    # dedupe while preserving order and cap the number of SAM3 forwards
    unique: list[str] = []
    for label in labels:
        label = " ".join(str(label).split())
        if label and label not in unique:
            unique.append(label)
    # "chair" adds nothing when "wooden chair" is already in the list; keeping
    # both would spend the second SAM3 forward on a broader duplicate
    generic = {
        label
        for label in unique
        if any(label != other and label in other.split() for other in unique)
    }
    unique = [label for label in unique if label not in generic]
    return unique[: max(1, int(max_labels))]


# ---------------------------------------------------------------------------
# frame helpers
# ---------------------------------------------------------------------------


def frame_to_numpy(frame: Any) -> np.ndarray:
    """CHW float tensor / HWC array -> uint8 HWC RGB (no torch import)."""
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    array = array[..., :3]
    if array.dtype != np.uint8:
        maximum = float(np.nanmax(array)) if array.size else 0.0
        scale = 255.0 if maximum <= 1.5 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _frame_to_pil(frame: Any):
    from PIL import Image

    return Image.fromarray(frame_to_numpy(frame), mode="RGB")


def _frame_hash(frame: Any) -> str:
    return hashlib.sha1(frame_to_numpy(frame).tobytes()).hexdigest()


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    ix1, iy1 = max(float(inner[0]), float(outer[0])), max(float(inner[1]), float(outer[1]))
    ix2, iy2 = min(float(inner[2]), float(outer[2])), min(float(inner[3]), float(outer[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = _box_area(inner)
    return inter / area if area > 0 else 0.0


# ---------------------------------------------------------------------------
# SAM3 + DINOv2
# ---------------------------------------------------------------------------


class Sam3DinoDetector(DetectorBackend):
    """SAM3 text-conditioned proposals, DINOv2 crop features.

    The backend is not loadable without a CUDA device: a 3.4 GB SAM3 forward is
    not a CPU fallback worth hiding. ``available()`` reports False in that case
    and ``ground_target`` keeps the VLM-only behaviour.
    """

    name = "sam3_dino"

    def __init__(
        self,
        sam3_path: str | Path = SAM3_DEFAULT_PATH,
        dino_path: str | Path = DINO_DEFAULT_PATH,
        device: str = "cuda:0",
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        dino_min: float = DEFAULT_DINO_MIN,
    ):
        self.sam3_path = str(sam3_path)
        self.dino_path = str(dino_path)
        self.device = str(device or "cuda:0")
        self.min_score = float(min_score)
        self.max_area_frac = float(max_area_frac)
        self.max_proposals = int(max_proposals)
        self.dino_min = float(dino_min)
        self._model = None
        self._processor = None
        self._dino = None
        self._dino_processor = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple, list[Detection]] = OrderedDict()
        self._cache_limit = 64
        self._feature_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._feature_limit = 256

    # -- readiness ---------------------------------------------------------

    def _imports_ok(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            from transformers import Sam3Model, Sam3Processor  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - a missing dependency is a finding
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""

    def _device_ok(self) -> bool:
        if not self.device.startswith("cuda"):
            return True
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    def available(self) -> bool:
        if not Path(self.sam3_path).exists():
            return False
        if not Path(self.dino_path).exists():
            return False
        ok, _ = self._imports_ok()
        return ok and self._device_ok()

    def probe(self) -> dict:
        imports_ok, import_error = self._imports_ok()
        return {
            "name": self.name,
            "available": bool(self.available()),
            "sam3_path": self.sam3_path,
            "sam3_exists": Path(self.sam3_path).exists(),
            "dino_path": self.dino_path,
            "dino_exists": Path(self.dino_path).exists(),
            "device": self.device,
            "cuda": self._device_ok(),
            "imports_ok": imports_ok,
            "import_error": import_error,
            "min_score": self.min_score,
            "max_area_frac": self.max_area_frac,
        }

    # -- loading -----------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available():
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                import torch
                from transformers import (
                    AutoImageProcessor,
                    Dinov2Model,
                    Sam3Model,
                    Sam3Processor,
                )

                self._model = Sam3Model.from_pretrained(self.sam3_path).to(self.device).eval()
                self._processor = Sam3Processor.from_pretrained(self.sam3_path)
                self._dino = Dinov2Model.from_pretrained(self.dino_path).to(self.device).eval()
                self._dino_processor = AutoImageProcessor.from_pretrained(self.dino_path)
                self._torch = torch
                return True
            except Exception as exc:  # noqa: BLE001 - report, never raise into a tool
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._model = None
                return False
        return False

    # -- detection ---------------------------------------------------------

    def detect(
        self,
        frame: Any,
        instruction: str,
        *,
        labels: Sequence[str] | None = None,
        context: Mapping | None = None,
    ) -> list[Detection]:
        if not self._ensure_loaded():
            return []
        prompts = [
            " ".join(str(label).split())
            for label in list(labels or target_labels(instruction))
            if str(label).strip()
        ][:2]
        if not prompts:
            return []
        key = (_frame_hash(frame), tuple(prompts))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return list(cached)
        image = _frame_to_pil(frame)
        proposals: list[Detection] = []
        torch = self._torch
        for prompt in prompts:
            try:
                inputs = self._processor(
                    images=image, text=prompt, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                processed = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.min_score,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]
            except Exception as exc:  # noqa: BLE001 - one failed prompt is not fatal
                self._load_error = f"detect: {type(exc).__name__}: {exc}"
                continue
            width = float(image.width)
            height = float(image.height)
            boxes = processed.get("boxes")
            scores = processed.get("scores")
            masks = processed.get("masks")
            if boxes is None or scores is None:
                continue
            try:
                boxes = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
                scores = scores.tolist() if hasattr(scores, "tolist") else list(scores)
            except (TypeError, ValueError):
                continue
            for index, (box, score) in enumerate(zip(boxes, scores)):
                try:
                    x1, y1, x2, y2 = (float(value) for value in list(box)[:4])
                except (TypeError, ValueError):
                    continue
                normalized = [
                    max(0.0, min(1000.0, x1 / width * 1000.0)),
                    max(0.0, min(1000.0, y1 / height * 1000.0)),
                    max(0.0, min(1000.0, x2 / width * 1000.0)),
                    max(0.0, min(1000.0, y2 / height * 1000.0)),
                ]
                area_frac = _box_area(normalized) / 1_000_000.0
                if area_frac <= 0.0 or area_frac > self.max_area_frac:
                    continue
                if normalized[2] - normalized[0] < 2 or normalized[3] - normalized[1] < 2:
                    continue
                mask = None
                mask_area = None
                if masks is not None:
                    try:
                        item = masks[index]
                        mask = item.detach().cpu().numpy().astype(bool)
                        mask_area = float(mask.mean())
                    except Exception:  # noqa: BLE001 - a mask is optional evidence
                        mask = None
                        mask_area = None
                proposals.append(
                    Detection(
                        box=normalized,
                        confidence=float(score),
                        label=prompt,
                        mask=mask,
                        mask_area_frac=mask_area,
                    )
                )
        # one object can answer two prompts: keep the highest-scoring box
        deduped: list[Detection] = []
        for proposal in sorted(proposals, key=lambda item: float(item.confidence or 0.0), reverse=True):
            if any(
                _box_iou(proposal.box, kept.box) >= 0.85
                or _containment(proposal.box, kept.box) >= 0.9
                for kept in deduped
            ):
                continue
            deduped.append(proposal)
            if len(deduped) >= self.max_proposals:
                break
        embeddings = self.embed_boxes(image, [proposal.box for proposal in deduped])
        if embeddings is not None:
            for proposal, embedding in zip(deduped, embeddings):
                proposal.dino_embedding = embedding
        self._cache[key] = deduped
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return list(deduped)

    # -- DINOv2 crop features ---------------------------------------------

    def embed_boxes(self, frame: Any, boxes: Sequence[Sequence[float]]):
        """L2-normalized DINOv2 CLS features of the box crops (None on failure)."""
        if not self._ensure_loaded() or not boxes:
            return None
        from PIL import Image

        image = frame if isinstance(frame, Image.Image) else _frame_to_pil(frame)
        width, height = image.width, image.height
        features: list[np.ndarray] = []
        torch = self._torch
        for box in boxes:
            key = (_frame_hash(image), tuple(round(float(v), 1) for v in box))
            cached = self._feature_cache.get(key)
            if cached is not None:
                self._feature_cache.move_to_end(key)
                features.append(cached)
                continue
            try:
                x1, y1, x2, y2 = (float(value) for value in list(box)[:4])
                pad_x = max(2.0, (x2 - x1) * 0.08)
                pad_y = max(2.0, (y2 - y1) * 0.08)
                left = max(0, int((x1 - pad_x) / 1000.0 * width))
                upper = max(0, int((y1 - pad_y) / 1000.0 * height))
                right = min(width, int((x2 + pad_x) / 1000.0 * width))
                lower = min(height, int((y2 + pad_y) / 1000.0 * height))
                if right - left < 4 or lower - upper < 4:
                    raise ValueError("empty_crop")
                crop = image.crop((left, upper, right, lower)).resize((224, 224), Image.BILINEAR)
                inputs = self._dino_processor(images=crop, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self._dino(**inputs)
                pooled = getattr(outputs, "pooler_output", None)
                vector = (
                    pooled[0]
                    if pooled is not None
                    else outputs.last_hidden_state[0, 0]
                ).detach().float().cpu().numpy()
                norm = float(np.linalg.norm(vector))
                vector = vector / norm if norm > 0 else vector
            except Exception:  # noqa: BLE001 - an embedding is optional
                vector = np.zeros(0, dtype=np.float32)
            self._feature_cache[key] = vector
            while len(self._feature_cache) > self._feature_limit:
                self._feature_cache.popitem(last=False)
            features.append(vector)
        return features


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

_INSTANCES: dict[tuple, DetectorBackend] = {}
_INSTANCES_LOCK = threading.Lock()


def sam3_detector(
    *,
    sam3_path: str | Path = SAM3_DEFAULT_PATH,
    dino_path: str | Path = DINO_DEFAULT_PATH,
    device: str = "cuda:0",
    **kwargs: Any,
) -> Sam3DinoDetector:
    """Process-wide SAM3+DINOv2 instance (weights load once, lazily)."""
    key = (str(sam3_path), str(dino_path), str(device))
    with _INSTANCES_LOCK:
        detector = _INSTANCES.get(key)
        if detector is None or not isinstance(detector, Sam3DinoDetector):
            detector = Sam3DinoDetector(
                sam3_path=sam3_path, dino_path=dino_path, device=device, **kwargs
            )
            _INSTANCES[key] = detector
        return detector


def resolve_detector(
    spec: str | None,
    *,
    device: str = "cuda:0",
    sam3_path: str | Path = SAM3_DEFAULT_PATH,
    dino_path: str | Path = DINO_DEFAULT_PATH,
    **kwargs: Any,
) -> DetectorBackend:
    """``off|sam3|sam3_dino|module:factory`` -> a backend (NullDetector for off)."""
    text = str(spec or "off").strip()
    lowered = text.lower()
    if lowered in ("", "off", "none", "null", "0", "false"):
        return NullDetector()
    if lowered in ("sam3", "sam3_dino", "sam3-dino", "sam3dino"):
        return sam3_detector(
            sam3_path=sam3_path, dino_path=dino_path, device=device, **kwargs
        )
    module_name, _, attribute = text.partition(":")
    if not attribute:
        raise ValueError(
            f"--detector {spec!r} must be 'off', 'sam3' or 'module:factory'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    detector = factory() if callable(factory) else factory
    if not hasattr(detector, "detect"):
        raise TypeError(f"{spec!r} does not return a detector backend")
    return detector


def detector_report(detector: DetectorBackend | None, status: str, **extra: Any) -> dict:
    """Uniform result block logged by ``ground_target``."""
    report = {
        "backend": getattr(detector, "name", "none") if detector is not None else "none",
        "status": str(status),
    }
    report.update(extra)
    return report


__all__ = [
    "DEFAULT_DINO_MIN",
    "DEFAULT_MAX_AREA_FRAC",
    "DEFAULT_MIN_SCORE",
    "Detection",
    "DetectorBackend",
    "NullDetector",
    "Sam3DinoDetector",
    "detector_report",
    "frame_to_numpy",
    "resolve_detector",
    "sam3_detector",
    "target_labels",
]
