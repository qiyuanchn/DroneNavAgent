"""Unified observation repository + the typed id contract (goal.md 3/4).

Every frame the model can address lives in one episode-scoped store, created and
owned by :class:`~vlm_nav.runtime.AgentRuntime` and reachable from every tool as
``ctx.extra["observation_store"]``. The runtime registers the frame it renders
each turn, the tools register the frames they capture, and a frame another tool
published is adopted on the way back through the runtime - so the image id the
model was shown and the id a tool resolves are the same ``obs_N`` (the failure
this module exists to remove: the model cites ``obs_10`` and a tool answers
"unknown observation" because the two sides kept separate id spaces).

An :class:`Observation` is evidence with provenance: the pose and heading it was
taken from, the intrinsics that map its pixels to rays, and the step/time it
belongs to. Positions are never fabricated here - a capture whose pose is
unknown is stored with ``NaN`` coordinates (``has_pose`` is then false and
:meth:`ObservationStore.poses` leaves it out), and an unknown field of view is
stored as ``hfov_deg: None``: pixel-to-ray math refuses to run instead of
silently assuming a default FOV.

Ids are typed and each type has exactly one home:

==============  ==================  ==========================================
id              store               meaning
==============  ==================  ==========================================
``obs_N``       this module         a recorded view: where it was taken, which
                                    way it faced, and the pixels it saw
``cand_N``      ``perception.candidates``  a target candidate inside an
                                    observation (``ground_target`` output)
``vp_N``        ``viewpoints_by_id``  a proposed observation *position*
==============  ==================  ==========================================

``require_observation_id`` / ``require_candidate_id`` / ``require_viewpoint_id``
enforce the separation at the tool boundary: an id of the wrong type raises
:class:`IdTypeError` carrying the expected type, the type actually passed and the
ids of the expected type that do exist - a model that hands ``obs_1`` to a
``candidate_id`` parameter gets told exactly that, instead of a bare "unknown".
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Iterable, Mapping

# ---------------------------------------------------------------------------
# ctx.extra keys: the one vocabulary every typed id resolves through
# ---------------------------------------------------------------------------

#: ``ctx.extra`` key of the episode's :class:`ObservationStore`.
OBSERVATION_STORE_KEY = "observation_store"
#: ``ctx.extra`` key of the ``cand_N`` -> candidate record store (the value is
#: unchanged from the perception tools, which re-export it).
CANDIDATE_STORE_KEY = "perception.candidates"
#: ``ctx.extra`` key of the navigation tools' ``obs_N`` -> pose records.
OBSERVATION_STATE_KEY = "observations"
#: ``ctx.extra`` key of the proposed viewpoints (``vp_N`` -> record).
VIEWPOINT_STORE_KEY = "viewpoints_by_id"
#: ``ctx.extra`` keys a caller may set to pin the render intrinsics.
INTRINSICS_KEYS = ("perception.intrinsics", "intrinsics")
#: ``ctx.extra`` keys a caller may set to pin the horizontal FOV alone.
HFOV_KEYS = ("perception.hfov_deg", "hfov_deg")

# ---------------------------------------------------------------------------
# id types (one prefix per kind; the id string is the only thing a model sees)
# ---------------------------------------------------------------------------

OBSERVATION_ID_PREFIX = "obs_"
CANDIDATE_ID_PREFIX = "cand_"
VIEWPOINT_ID_PREFIX = "vp_"
OBSERVATION_ID_TYPE = "observation_id"
CANDIDATE_ID_TYPE = "candidate_id"
VIEWPOINT_ID_TYPE = "viewpoint_id"

_ID_PREFIXES = {
    OBSERVATION_ID_TYPE: ("obs_", "obs#", "obs:", "observation_"),
    CANDIDATE_ID_TYPE: ("cand_", "cand#", "cand:", "candidate_"),
    VIEWPOINT_ID_TYPE: ("vp_", "vp#", "vp:", "viewpoint_"),
}

# ---------------------------------------------------------------------------
# observation kinds (the five deliberate reasons a frame is kept)
# ---------------------------------------------------------------------------

KIND_EVIDENCE = "evidence"  # the frame of a turn / a deliberate single view
KIND_SCAN = "scan"  # one view of a lookaround sweep
KIND_VERIFY = "verify"  # a frame captured to check a claim
KIND_INSPECT = "inspect"  # a magnified crop of another observation
KIND_REOBSERVE = "reobserve"  # a fresh frame of an earlier place
OBSERVATION_KINDS = (KIND_EVIDENCE, KIND_SCAN, KIND_VERIFY, KIND_INSPECT, KIND_REOBSERVE)

#: Bounded store: the oldest records are dropped first. The pixels of a record
#: may also live in the perception cache / memory archive, so eviction here
#: degrades in-process addressability without breaking the archive.
DEFAULT_MAX_OBSERVATIONS = 256

UNKNOWN_POSE: tuple[float, float, float] = (math.nan, math.nan, math.nan)
UNKNOWN_YAW_DEG = math.nan

_OBS_ID_RE = re.compile(r"^obs[_#:]?(\d+)$", re.IGNORECASE)
_CAND_ID_RE = re.compile(r"^cand[_#:]?(\d+)$", re.IGNORECASE)
_VP_ID_RE = re.compile(r"^vp[_#:]?(\d+)$", re.IGNORECASE)
_INDEX_RE = re.compile(r"^(\d+)$")


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _float_or_none(value: Any) -> float | None:
    """``value`` as a finite float, or None (bool/None/text are not numbers)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return None if number is None else int(round(number))


def is_known_pose(pose_flu: Any) -> bool:
    """True when ``pose_flu`` is a finite 3-vector (an unknown pose is NaN)."""
    try:
        values = [float(v) for v in list(pose_flu)[:3]]
    except (TypeError, ValueError):
        return False
    return len(values) == 3 and all(math.isfinite(v) for v in values)


def _pose_tuple(pose_flu: Any) -> tuple[float, float, float]:
    """``pose_flu`` as a plain float triple; unknown/invalid becomes NaN."""
    if pose_flu is None:
        return UNKNOWN_POSE
    try:
        values = [float(v) for v in list(pose_flu)[:3]]
    except (TypeError, ValueError):
        return UNKNOWN_POSE
    if len(values) != 3:
        return UNKNOWN_POSE
    return (values[0], values[1], values[2])


def observation_id(index: Any) -> str:
    """Canonical ``obs_N`` of an integer index."""
    return f"{OBSERVATION_ID_PREFIX}{int(index)}"


def classify_id(raw: Any) -> str | None:
    """Type of an id as written (``observation_id`` / ``candidate_id`` /
    ``viewpoint_id``), or None when it carries no recognisable prefix.

    A bare number carries no type (``"7"`` is ``obs_7`` to an observation
    argument and ``cand_7`` to a candidate argument); the caller decides.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or _INDEX_RE.match(text):
        return None
    for id_type, prefixes in _ID_PREFIXES.items():
        if text.startswith(prefixes):
            return id_type
    return None


def parse_id_index(raw: Any, id_type: str) -> str | None:
    """Canonical id of ``raw`` for ``id_type``, tolerating the bare-index alias.

    ``"obs_3"`` / ``"OBS_3"`` / ``"3"`` all resolve to ``"obs_3"`` for an
    observation; an id written with another type's prefix resolves to None (the
    caller must report a type error, not a lookup failure).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    patterns = {
        OBSERVATION_ID_TYPE: (_OBS_ID_RE, OBSERVATION_ID_PREFIX),
        CANDIDATE_ID_TYPE: (_CAND_ID_RE, CANDIDATE_ID_PREFIX),
        VIEWPOINT_ID_TYPE: (_VP_ID_RE, VIEWPOINT_ID_PREFIX),
    }
    pattern, prefix = patterns.get(id_type, (None, None))
    if pattern is None:
        return None
    match = pattern.match(text)
    if match is None and _INDEX_RE.match(text):
        return f"{prefix}{int(text)}"
    if match is None:
        return None
    return f"{prefix}{int(match.group(1))}"


# ---------------------------------------------------------------------------
# render intrinsics (never a hard-coded FOV)
# ---------------------------------------------------------------------------

_HFOV_ATTRS = ("hfov_deg", "camera_hfov_deg", "hfov", "horizontal_fov", "fov_deg", "fov")
_SIZE_ATTRS = (
    ("width", ("width", "W", "w", "width_px")),
    ("height", ("height", "H", "h", "height_px")),
)
_SENSOR_HOLDERS = ("camera", "rgb_camera", "sensor", "sensors", "cameras")
_SENSOR_KEYS = ("rgb_camera", "camera", "rgb", "exteroception_rgb_camera")


def _field(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _hfov_of(obj: Any) -> tuple[float | None, str | None]:
    """(hfov_deg, source) declared by one object; (None, None) when it declares
    no finite field of view."""
    for name in _HFOV_ATTRS:
        value = _field(obj, (name,))
        number = _float_or_none(value)
        if number is not None and 0.0 < number < 180.0:
            return number, name
    return None, None


def _size_of(obj: Any) -> tuple[int | None, int | None, str | None]:
    """(width, height, source) declared by one object."""
    width = _int_or_none(_field(obj, _SIZE_ATTRS[0][1]))
    height = _int_or_none(_field(obj, _SIZE_ATTRS[1][1]))
    if width is None or height is None or width <= 0 or height <= 0:
        return None, None, None
    return width, height, "size"


def _candidates_of(value: Any) -> list[tuple[str, Any]]:
    """(key, camera-like object) pairs inside a ``sensors`` attribute/dict."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        ordered = [(key, value[key]) for key in _SENSOR_KEYS if key in value]
        ordered += [(key, item) for key, item in value.items() if key not in _SENSOR_KEYS]
        return ordered
    if isinstance(value, (list, tuple)):
        return [(str(index), item) for index, item in enumerate(value)]
    return [("", value)]


def _sensor_intrinsics(world: Any) -> tuple[float | None, str, int | None, int | None, str]:
    """(hfov, hfov_source, width, height, holder) declared by the world's cameras.

    ``holder`` / ``hfov_source`` name where the value came from
    (``sensors.rgb_camera`` / ``sensors.rgb_camera.hfov``), so an audited
    observation can say which declaration its projection used.
    """
    for holder_name in _SENSOR_HOLDERS:
        for owner in (world, getattr(world, "env", None)):
            if owner is None:
                continue
            for key, candidate in _candidates_of(_field(owner, (holder_name,))):
                hfov, source = _hfov_of(candidate)
                width, height, _ = _size_of(candidate)
                if hfov is None and width is None:
                    continue
                holder = ".".join(part for part in (holder_name, key) if part)
                return hfov, (f"{holder}.{source}" if hfov is not None else "unknown"), width, height, holder
    return None, "unknown", None, None, "unknown"


def _frame_size(frame: Any) -> tuple[int | None, int | None]:
    """(width, height) of a CHW/HWC frame tensor or array."""
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None, None
    try:
        dims = [int(v) for v in shape]
    except (TypeError, ValueError):
        return None, None
    if len(dims) >= 3 and dims[0] in (1, 3, 4):
        return dims[-1], dims[-2]
    return dims[-1], dims[-2]


def resolve_intrinsics(
    ctx: Any = None,
    *,
    frame: Any = None,
    override: Mapping | None = None,
    world: Any = None,
) -> dict:
    """The render intrinsics of one observation: ``{width, height, hfov_deg}``.

    ``hfov_deg`` is never defaulted (the 86 deg assumption was one of the
    coordinate bugs): it is read from an explicit ``override``, then from
    ``ctx.extra["perception.intrinsics"]`` / ``["perception.hfov_deg"]``, then
    from the world or its camera/sensor objects. When nothing declares it the
    value is ``None`` and ``hfov_source`` says ``"unknown"`` - pixel-to-ray math
    then refuses to run instead of biasing every lateral offset.

    Width/height are the *rendered* pixel size: the frame's own shape first
    (the resolution that was actually rendered), then the world/sensor values.
    ``0`` means "unknown" for both.
    """
    ctx_world = world if world is not None else getattr(ctx, "world", None)
    extra = getattr(ctx, "extra", None)
    extra = extra if isinstance(extra, Mapping) else {}

    width = height = None
    source = "unknown"

    if isinstance(override, Mapping):
        hfov, hfov_source = _hfov_of(override)
        width = _int_or_none(_field(override, ("width", "W")))
        height = _int_or_none(_field(override, ("height", "H")))
        source = "override"
    else:
        hfov, hfov_source = (None, None)
        for key in INTRINSICS_KEYS:
            declared = extra.get(key)
            if isinstance(declared, Mapping):
                hfov, hfov_source = _hfov_of(declared)
                width = _int_or_none(_field(declared, ("width", "W")))
                height = _int_or_none(_field(declared, ("height", "H")))
                source = f"ctx.extra[{key}]"
                break
        if hfov is None:
            scalar = _float_or_none(_field(extra, HFOV_KEYS))
            if scalar is not None and 0.0 < scalar < 180.0:
                hfov, hfov_source = scalar, "ctx.extra"
                source = "ctx.extra"

    if hfov is None:
        hfov, hfov_source = _hfov_of(ctx_world)
        if hfov is not None:
            source = "world"
    sensor_hfov, sensor_source, sensor_w, sensor_h, sensor_holder = _sensor_intrinsics(ctx_world)
    if hfov is None and sensor_hfov is not None:
        hfov, hfov_source, source = sensor_hfov, sensor_source, sensor_holder
    # the declared render size is a fallback either way: a world that declares
    # its FOV directly may still keep width/height on its camera object
    width = width or sensor_w
    height = height or sensor_h

    if width is None or height is None:
        world_w, world_h, _ = _size_of(ctx_world)
        width = width or world_w
        height = height or world_h
    frame_w, frame_h = _frame_size(frame)
    width = frame_w or width
    height = frame_h or height

    return {
        "width": int(width or 0),
        "height": int(height or 0),
        "hfov_deg": hfov,
        "hfov_source": hfov_source or "unknown",
        "source": source,
    }


def kind_for_label(label: Any, default: str = KIND_EVIDENCE) -> str:
    """Store kind of a producer's own label (``"post_leg"``, ``"scan"``, ...).

    A frame carries the reason it was kept; the store vocabulary is the five
    canonical kinds, and a label that merely *contains* one of them (the
    navigation tools write ``"verify"``/``"scan"``/``"post_leg"``) maps onto it
    instead of inventing a sixth kind.
    """
    text = str(label or "").strip().lower()
    if not text:
        return default
    if text in OBSERVATION_KINDS:
        return text
    for kind in (KIND_VERIFY, KIND_SCAN, KIND_REOBSERVE, KIND_INSPECT, KIND_EVIDENCE):
        if kind in text:
            return kind
    return default


# ---------------------------------------------------------------------------
# observation record
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """One recorded frame with the provenance needed to reason about it.

    ``pose_flu`` / ``yaw_deg`` describe the capture, not the drone's current
    pose, and are NaN when the world exposed no pose (``has_pose`` /
    ``has_yaw``). ``intrinsics`` always carries the three documented keys
    (``width`` / ``height`` / ``hfov_deg``, plus the audit keys
    ``hfov_source`` / ``source``): ``hfov_deg`` is None when nothing declared
    the field of view.

    ``frame`` keeps the in-process pixels when the recorder still has them
    (``""``-path records may carry pixels only in the perception cache or the
    memory archive); ``image_path`` is the on-disk frame when one was written.
    """

    obs_id: str
    kind: str
    image_path: str
    pose_flu: tuple[float, float, float]
    yaw_deg: float
    intrinsics: dict
    timestamp: float
    step: int
    frame: Any = field(default=None, repr=False)
    parent_id: str | None = None
    note: str = ""
    episode_id: str | None = None
    #: Producer-specific provenance that is not part of the addressing contract
    #: (a pitched capture's ``pitch_down_deg``, a crop's
    #: ``crop = {source_observation_id, region_uvwh, upscale, ...}`` mapping).
    #: Kept verbatim so a record rebuilt from the store projects like the frame
    #: it was captured from.
    detail: dict = field(default_factory=dict)

    @property
    def has_pose(self) -> bool:
        return is_known_pose(self.pose_flu)

    @property
    def has_yaw(self) -> bool:
        return _float_or_none(self.yaw_deg) is not None

    @property
    def width(self) -> int:
        return int(self.intrinsics.get("width") or 0)

    @property
    def height(self) -> int:
        return int(self.intrinsics.get("height") or 0)

    @property
    def hfov_deg(self) -> float | None:
        return _float_or_none(self.intrinsics.get("hfov_deg"))

    def pose_dict(self) -> dict:
        """``{"position_flu": [x, y, z], "yaw_deg": float | None}``.

        The shape ``navigate`` and the target-error answers already speak, so a
        stored observation can be handed to them unchanged.
        """
        return {
            "position_flu": [float(v) for v in self.pose_flu],
            "yaw_deg": float(self.yaw_deg) if self.has_yaw else None,
        }


class ObservationStore:
    """Episode-scoped ``obs_N`` repository: everything the model can address.

    Ids are minted from one counter shared with ``ToolContext.next_obs_id``
    (which reserves its id here), so no producer can collide with another, and
    an id adopted from elsewhere (a memory hit, a frame a tool published under
    its own id) advances the counter past it.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_OBSERVATIONS,
        *,
        episode_id: str | None = None,
        start_seq: int = 0,
    ):
        self.max_entries = max(1, int(max_entries))
        self.episode_id = str(episode_id) if episode_id else None
        self._records: "OrderedDict[str, Observation]" = OrderedDict()
        self._next_index = max(0, int(start_seq)) + 1

    # ---------------- lifecycle ----------------

    def reset(self, *, episode_id: str | None = None, start_seq: int | None = None) -> None:
        """Start a new episode: drop every record and rebind the id counter.

        ``start_seq`` is the context's ``observation_seq`` (the ids already
        handed out this session), so a second episode continues the sequence
        instead of reusing ids the first one published.
        """
        self._records.clear()
        if episode_id is not None:
            self.episode_id = str(episode_id) if episode_id else None
        if start_seq is not None:
            self._next_index = max(0, int(start_seq)) + 1

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, raw: Any) -> bool:
        return self.resolve_id(raw) is not None

    def __iter__(self):
        return iter(self.ids())

    # ---------------- ids ----------------

    def resolve_id(self, raw: Any) -> str | None:
        """Canonical registered id of ``raw``, or None.

        Tolerates the bare index (``"3"`` for an observation parameter means
        ``obs_3``) and the case/separator spellings; an id written for another
        type never resolves here (that is a type error, not a miss).
        """
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        if text in self._records:
            return text
        canonical = parse_id_index(text, OBSERVATION_ID_TYPE)
        if canonical is not None and canonical in self._records:
            return canonical
        return None

    def forget(self, raw: Any) -> bool:
        """Drop one record (e.g. its pixels were released); True when dropped.

        The id stays reserved: a later registration with the same id fills the
        record again instead of a new one being minted for it.
        """
        obs_id = self.resolve_id(raw)
        if obs_id is None:
            return False
        self._records.pop(obs_id, None)
        return True

    def next_id(self) -> str:
        """Mint the next free ``obs_N`` (reserved ids are never reused)."""
        obs_id = observation_id(self._next_index)
        self._next_index += 1
        return obs_id

    def reserve_id(self, raw: Any) -> str:
        """Mark an id as handed out elsewhere (``ToolContext.next_obs_id``).

        Reserving does not create a record - the frame is registered later with
        that id - but it keeps :meth:`next_id` from minting it twice.
        """
        text = str(raw) if raw is not None else ""
        canonical = parse_id_index(text, OBSERVATION_ID_TYPE)
        if canonical is not None:
            self._next_index = max(self._next_index, int(canonical[len(OBSERVATION_ID_PREFIX):]) + 1)
        return text

    def ids(self, kind: str | None = None) -> list[str]:
        """Registered ids, newest first (optionally only one ``kind``)."""
        return [
            obs_id
            for obs_id, record in reversed(self._records.items())
            if kind is None or record.kind == kind
        ]

    def latest(self, kind: str | None = None) -> Observation | None:
        obs_ids = self.ids(kind)
        return self._records[obs_ids[0]] if obs_ids else None

    # ---------------- writing ----------------

    def register(
        self,
        image_path: Any = "",
        pose_flu: Any = None,
        yaw_deg: Any = None,
        intrinsics: Mapping | None = None,
        kind: str = KIND_EVIDENCE,
        step: int = 0,
        timestamp: float | None = None,
        *,
        obs_id: str | None = None,
        frame: Any = None,
        parent_id: str | None = None,
        note: str = "",
        episode_id: str | None = None,
        detail: Mapping | None = None,
    ) -> str:
        """Record one frame; returns its ``obs_id``.

        ``pose_flu`` / ``yaw_deg`` are the capture pose (NaN when unknown), never
        a guess. Re-registering an id updates the record in place (the recorder
        of a frame may fill in pixels or metadata later) while keeping its
        original position in the sequence; omitted pixels/path of the update
        keep the values already stored.
        """
        obs_id = self.reserve_id(obs_id) if obs_id else self.next_id()
        canonical = parse_id_index(obs_id, OBSERVATION_ID_TYPE) or obs_id
        obs_id = canonical
        existing = self._records.get(obs_id)
        declared = dict(intrinsics) if isinstance(intrinsics, Mapping) else {}
        record = Observation(
            obs_id=obs_id,
            kind=str(kind or KIND_EVIDENCE).strip().lower() or KIND_EVIDENCE,
            image_path=str(image_path or (existing.image_path if existing else "")),
            pose_flu=_pose_tuple(pose_flu if pose_flu is not None else (existing.pose_flu if existing else None)),
            yaw_deg=(
                float(yaw_deg)
                if _float_or_none(yaw_deg) is not None
                else (existing.yaw_deg if existing else UNKNOWN_YAW_DEG)
            ),
            intrinsics=declared or (dict(existing.intrinsics) if existing else {"width": 0, "height": 0, "hfov_deg": None}),
            timestamp=(
                float(timestamp)
                if _float_or_none(timestamp) is not None
                else (existing.timestamp if existing else time.time())
            ),
            step=_int_or_none(step) if _int_or_none(step) is not None else (existing.step if existing else 0),
            frame=frame if frame is not None else (existing.frame if existing else None),
            parent_id=str(parent_id) if parent_id else (existing.parent_id if existing else None),
            note=str(note or (existing.note if existing else "")),
            episode_id=str(episode_id) if episode_id else (existing.episode_id if existing else self.episode_id),
            detail=dict(detail) if isinstance(detail, Mapping) else (dict(existing.detail) if existing else {}),
        )
        self._records[obs_id] = record  # an existing key keeps its position
        while len(self._records) > self.max_entries:
            self._records.popitem(last=False)
        return obs_id

    # ---------------- reading ----------------

    def get(self, raw: Any) -> Observation | None:
        obs_id = self.resolve_id(raw)
        return None if obs_id is None else self._records.get(obs_id)

    def frame(self, raw: Any) -> Any:
        record = self.get(raw)
        return None if record is None else record.frame

    def image_path(self, raw: Any) -> str | None:
        record = self.get(raw)
        if record is None or not record.image_path:
            return None
        return record.image_path

    def poses(self) -> dict[str, dict]:
        """``obs_id -> {"position_flu": [...], "yaw_deg": float | None}``.

        Newest first, and only observations whose capture pose is known: a
        place nobody can point at is not a place to fly to.
        """
        out: dict[str, dict] = {}
        for obs_id in self.ids():
            record = self._records[obs_id]
            if record.has_pose:
                out[obs_id] = record.pose_dict()
        return out


# ---------------------------------------------------------------------------
# context wiring
# ---------------------------------------------------------------------------


def observation_store(ctx: Any, create: bool = False) -> ObservationStore | None:
    """The episode's :class:`ObservationStore`, optionally creating it.

    A value of an unexpected type under the key (a hand-built or deserialized
    ``extra``) is replaced rather than trusted. A freshly created store starts
    its counter above ``ctx.observation_seq``, so the ids of frames that were
    already published without it are never handed out twice.
    """
    if ctx is None:
        return None
    if not isinstance(getattr(ctx, "extra", None), dict):
        if not create:
            return None
        ctx.extra = {}
    store = ctx.extra.get(OBSERVATION_STORE_KEY)
    if isinstance(store, ObservationStore):
        return store
    if not create:
        return None
    store = ObservationStore(
        episode_id=getattr(ctx, "episode_id", None),
        start_seq=_int_or_none(getattr(ctx, "observation_seq", 0)) or 0,
    )
    ctx.extra[OBSERVATION_STORE_KEY] = store
    return store


def _current_pose(world: Any) -> tuple[Any, Any]:
    """(position_flu, yaw_deg) of a world; (None, None) when it has none."""
    if world is None:
        return None, None
    position = _field(world, ("position_flu", "position"))
    if not is_known_pose(position):
        state = getattr(world, "state", None)
        position = _field(state, ("position_flu", "position"))
    if not is_known_pose(position):
        position = None
    yaw_rad = _float_or_none(_field(world, ("yaw_rad", "yaw")))
    if yaw_rad is None:
        yaw_rad = _float_or_none(_field(getattr(world, "state", None), ("yaw_rad", "yaw")))
    return position, (None if yaw_rad is None else math.degrees(yaw_rad))


def register_observation(
    ctx: Any,
    frame: Any = None,
    *,
    kind: str = KIND_EVIDENCE,
    step: int | None = None,
    timestamp: float | None = None,
    image_path: str = "",
    obs_id: str | None = None,
    pose_flu: Any = None,
    yaw_deg: Any = None,
    intrinsics: Mapping | None = None,
    parent_id: str | None = None,
    note: str = "",
    detail: Mapping | None = None,
) -> Observation | None:
    """Register a frame the model can see; returns the :class:`Observation`.

    Pose, heading and intrinsics default to what the world declares *now* (the
    frame was just rendered), and an explicit value always wins - that is how a
    lookaround view records the heading it was rendered at rather than the
    restored one. ``obs_id`` adopts an id handed out elsewhere
    (``ctx.next_obs_id()``, a memory hit); without it the store mints one.
    Returns None when the context cannot hold a store.
    """
    store = observation_store(ctx, create=True)
    if store is None:
        return None
    world_pose, world_yaw_deg = _current_pose(getattr(ctx, "world", None))
    if pose_flu is None:
        pose_flu = world_pose
    if yaw_deg is None:
        yaw_deg = world_yaw_deg
    if intrinsics is None:
        intrinsics = resolve_intrinsics(ctx, frame=frame)
    if obs_id is None:
        mint = getattr(ctx, "next_obs_id", None)
        obs_id = mint() if callable(mint) else store.next_id()
    obs_id = store.register(
        image_path,
        pose_flu,
        yaw_deg if yaw_deg is not None else UNKNOWN_YAW_DEG,
        intrinsics,
        kind,
        0 if step is None else step,
        timestamp,
        obs_id=obs_id,
        frame=frame,
        parent_id=parent_id,
        note=note,
        detail=detail,
    )
    return store.get(obs_id)


def _pose_from_record(entry: Any) -> tuple[tuple[float, float, float] | None, float | None]:
    """(position_flu, yaw_rad) of a pose-bearing record (dict or object)."""
    position = _field(entry, ("position_flu", "pose", "position", "world_flu"))
    values = None
    if position is not None:
        try:
            numbers = [float(v) for v in list(position)[:3]]
        except (TypeError, ValueError):
            numbers = []
        if len(numbers) == 3 and all(math.isfinite(v) for v in numbers):
            values = (numbers[0], numbers[1], numbers[2])
    yaw_rad = _float_or_none(_field(entry, ("yaw_rad", "yaw")))
    if yaw_rad is None:
        yaw_deg = _float_or_none(_field(entry, ("yaw_deg",)))
        yaw_rad = None if yaw_deg is None else math.radians(yaw_deg)
    return values, yaw_rad


def lookup_pose(ctx: Any, obs_id: Any) -> tuple[tuple[float, float, float] | None, float | None]:
    """(position_flu, yaw_rad) of an observation id without decoding pixels.

    Reads the observation store first, then the navigation tools' pose records
    (``ctx.extra["observations"]``) - the two places a capture pose is written.
    """
    if ctx is None or obs_id in (None, ""):
        return None, None
    store = observation_store(ctx, create=False)
    if store is not None:
        record = store.get(obs_id)
        if record is not None and record.has_pose:
            yaw_rad = math.radians(float(record.yaw_deg)) if record.has_yaw else None
            return (float(record.pose_flu[0]), float(record.pose_flu[1]), float(record.pose_flu[2])), yaw_rad
    text = str(obs_id).strip()
    state = getattr(ctx, "extra", None)
    state = state.get(OBSERVATION_STATE_KEY) if isinstance(state, dict) else None
    if isinstance(state, dict):
        canonical = parse_id_index(text, OBSERVATION_ID_TYPE)
        for key in (text, canonical):
            if key and key in state:
                return _pose_from_record(state[key])
    return None, None


def published_frames(ctx: Any) -> dict:
    """``ctx.extra["images"]``: frames published for the next model turn."""
    extra = getattr(ctx, "extra", None)
    published = extra.get("images") if isinstance(extra, dict) else None
    return published if isinstance(published, dict) else {}


# ---------------------------------------------------------------------------
# typed id checks
# ---------------------------------------------------------------------------


class IdTypeError(ValueError):
    """A typed id was passed where another type (or a known id) was required.

    ``reason`` is ``missing_id`` / ``id_type_mismatch`` / ``unknown_<type>``, and
    every instance carries the ids of the **expected** type that do exist, so a
    tool answer can hand the model a usable one instead of only a complaint.
    """

    def __init__(
        self,
        raw: Any,
        expected: str,
        available: Iterable[str] | None = None,
        *,
        reason: str = "id_type_mismatch",
        actual: str | None = None,
        hint: str = "",
    ):
        self.raw = "" if raw is None else str(raw)
        self.expected = expected
        self.available = [str(item) for item in (available or [])]
        self.reason = reason
        self.actual = actual
        self.hint = hint
        super().__init__(self.message())

    def message(self) -> str:
        what = f"'{self.raw}'" if self.raw else "no id"
        if self.reason == "id_type_mismatch" and self.actual:
            head = (
                f"{what} is a {self.actual}, but this argument takes a {self.expected} "
                f"({self.id_examples()})"
            )
        elif self.reason == "unknown_observation_id":
            head = f"unknown {self.expected} {what}"
        elif self.reason == "missing_id":
            head = f"missing {self.expected}"
        else:
            head = f"unknown {self.expected} {what}"
        tail = f"; available {self.expected}s: {', '.join(self.available)}" if self.available else ""
        note = f"; {self.hint}" if self.hint else ""
        return head + tail + note

    def id_examples(self) -> str:
        return {
            OBSERVATION_ID_TYPE: "obs_N, an observation id",
            CANDIDATE_ID_TYPE: "cand_N, a target candidate id",
            VIEWPOINT_ID_TYPE: "vp_N, a viewpoint id",
        }.get(self.expected, self.expected)

    def to_data(self) -> dict:
        """Tool-result payload of the error (structured, id list included)."""
        data: dict[str, Any] = {
            "error": self.error_name(),
            "expected": self.expected,
            "got": self.raw or None,
            "available": list(self.available),
            f"available_{self.expected}s": list(self.available),
        }
        if self.actual:
            data["actual_type"] = self.actual
        if self.reason != "missing_id" and self.raw:
            data[self.expected] = self.raw
        if self.hint:
            data["hint"] = self.hint
        return data

    def error_name(self) -> str:
        """``data["error"]`` of this failure (the tool-level spelling).

        A missing argument keeps the tool-specific name the schemas and the
        earlier answers used (``missing_observation_id``), so callers that keyed
        on it keep working.
        """
        if self.reason == "missing_id":
            return _MISSING_ID_ERRORS.get(self.expected, "missing_id")
        return self.reason


def _available_from_mapping(ctx: Any, key: str, limit: int) -> list[str]:
    extra = getattr(ctx, "extra", None)
    container = extra.get(key) if isinstance(extra, dict) else None
    if not isinstance(container, Mapping):
        return []
    return [str(item) for item in list(container)][-max(1, int(limit)):]


def available_observation_ids(ctx: Any, limit: int = 8) -> list[str]:
    """Ids of the observations that exist right now, newest first.

    The store (every frame the model was shown) plus the pose records the
    navigation tools write, de-duplicated and capped.
    """
    found: list[str] = []
    store = observation_store(ctx, create=False)
    if store is not None:
        found.extend(store.ids())
    extra = getattr(ctx, "extra", None)
    state = extra.get(OBSERVATION_STATE_KEY) if isinstance(extra, dict) else None
    if isinstance(state, Mapping):
        for key in reversed(list(state)):
            if str(key) not in found:
                found.append(str(key))
    published = published_frames(ctx)
    for key in reversed(list(published)):
        if str(key) not in found:
            found.append(str(key))
    return found[: max(1, int(limit))]


def available_candidate_ids(ctx: Any, limit: int = 8) -> list[str]:
    return _available_from_mapping(ctx, CANDIDATE_STORE_KEY, limit)


def available_viewpoint_ids(ctx: Any, limit: int = 8) -> list[str]:
    return _available_from_mapping(ctx, VIEWPOINT_STORE_KEY, limit)


def known_observation_id(ctx: Any, raw: Any) -> str | None:
    """Canonical id of an existing observation, or None.

    Looks in the store first, then at the ids other producers published (pose
    records / published frames). Memory is deliberately not probed here: the
    perception tools adopt a recalled record into the store when it is used, so
    the store stays the single answer to "does this observation exist?".
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    canonical = parse_id_index(text, OBSERVATION_ID_TYPE)
    candidates = [text] + ([canonical] if canonical and canonical != text else [])
    store = observation_store(ctx, create=False)
    if store is not None:
        found = store.resolve_id(text)
        if found:
            return found
    extra = getattr(ctx, "extra", None)
    state = extra.get(OBSERVATION_STATE_KEY) if isinstance(extra, dict) else None
    published = published_frames(ctx)
    for key in candidates:
        if isinstance(state, Mapping) and key in state:
            return key
        if key in published:
            return key
    return None


def require_observation_id(raw: Any, ctx: Any) -> str:
    """An existing ``obs_N`` (bare ``"3"`` means ``obs_3``); raises on misuse."""
    available = available_observation_ids(ctx)
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise IdTypeError(
            raw, OBSERVATION_ID_TYPE, available, reason="missing_id",
            hint="pass an observation id a tool returned (obs_N)",
        )
    actual = classify_id(text)
    if actual is not None and actual != OBSERVATION_ID_TYPE:
        raise IdTypeError(
            text, OBSERVATION_ID_TYPE, available, reason="id_type_mismatch", actual=actual,
            hint=_MISMATCH_HINTS[actual],
        )
    found = known_observation_id(ctx, text)
    if found is None:
        raise IdTypeError(
            text, OBSERVATION_ID_TYPE, available, reason="unknown_observation_id",
            hint="call observe (or use an id a tool returned) before referencing a frame",
        )
    return found


def require_candidate_id(raw: Any, ctx: Any) -> str:
    """An existing ``cand_N`` (bare ``"3"`` means ``cand_3``); raises on misuse."""
    available = available_candidate_ids(ctx)
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise IdTypeError(
            raw, CANDIDATE_ID_TYPE, available, reason="missing_id",
            hint="pass a candidate id from ground_target (cand_N)",
        )
    actual = classify_id(text)
    if actual is not None and actual != CANDIDATE_ID_TYPE:
        raise IdTypeError(
            text, CANDIDATE_ID_TYPE, available, reason="id_type_mismatch", actual=actual,
            hint=_MISMATCH_HINTS[actual],
        )
    extra = getattr(ctx, "extra", None)
    store = extra.get(CANDIDATE_STORE_KEY) if isinstance(extra, dict) else None
    store = store if isinstance(store, Mapping) else {}
    canonical = parse_id_index(text, CANDIDATE_ID_TYPE) or text
    for key in (text, canonical):
        if key in store:
            return key
    raise IdTypeError(
        text, CANDIDATE_ID_TYPE, available, reason="unknown_candidate_id",
        hint="call ground_target first; only candidate ids it returned can be used",
    )


def require_viewpoint_id(raw: Any, ctx: Any) -> str:
    """An existing ``vp_N`` (bare ``"3"`` means ``vp_3``); raises on misuse."""
    available = available_viewpoint_ids(ctx)
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise IdTypeError(
            raw, VIEWPOINT_ID_TYPE, available, reason="missing_id",
            hint="pass a viewpoint id from propose_viewpoints (vp_N)",
        )
    actual = classify_id(text)
    if actual is not None and actual != VIEWPOINT_ID_TYPE:
        raise IdTypeError(
            text, VIEWPOINT_ID_TYPE, available, reason="id_type_mismatch", actual=actual,
            hint=_MISMATCH_HINTS[actual],
        )
    extra = getattr(ctx, "extra", None)
    store = extra.get(VIEWPOINT_STORE_KEY) if isinstance(extra, dict) else None
    store = store if isinstance(store, Mapping) else {}
    canonical = parse_id_index(text, VIEWPOINT_ID_TYPE) or text
    for key in (text, canonical):
        if key in store:
            return key
    raise IdTypeError(
        text, VIEWPOINT_ID_TYPE, available, reason="unknown_viewpoint_id",
        hint="call propose_viewpoints first; only viewpoint ids it returned can be used",
    )


_MISMATCH_HINTS = {
    OBSERVATION_ID_TYPE: (
        "obs_N is a recorded view (a place that was photographed); it is a target "
        "for approach_candidate/return_to_observation, not a candidate id"
    ),
    CANDIDATE_ID_TYPE: (
        "cand_N is a target candidate inside an observation (from ground_target), "
        "not an observation or viewpoint id"
    ),
    VIEWPOINT_ID_TYPE: (
        "vp_N is a proposed observation position (from propose_viewpoints); use "
        "go_to_viewpoint for it"
    ),
}

#: ``data["error"]`` of a missing argument, per id type (the tool-level spelling
#: the schemas and the earlier answers used).
_MISSING_ID_ERRORS = {
    OBSERVATION_ID_TYPE: "missing_observation_id",
    CANDIDATE_ID_TYPE: "missing_candidate_id",
    VIEWPOINT_ID_TYPE: "missing_viewpoint_id",
}
