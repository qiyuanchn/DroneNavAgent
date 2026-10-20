"""Deterministic Chinese/English instruction -> mixed mission planner (M2).

The planner is deliberately a grammar, not a guesser: it recognizes a bounded
vocabulary and raises :class:`TaskPlannerError` naming the clause it could not
understand.  If the caller supplies a ``vlm``, exactly one strict-JSON fallback
is attempted for the whole instruction; the returned mission is validated with
:class:`~vlm_nav.mixed_mission.MixedMissionSpec` before being accepted.

The goal.md example must parse exactly::

    上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，找到窗边的木椅，
    飞到它旁边，再向左移动 1 米并拍照。

English equivalents ("ascend 1 meter ... turn right 90 degrees ... find the
wooden chair by the window ... fly to it, then move left 1 meter and take a
photo") produce the same step chain.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .mixed_mission import MixedMissionError, MixedMissionSpec

_LEN = r"(?:米|m|meters?|metres?)"
_TIME = r"(?:秒|s|sec(?:onds?)?)"
_ANGLE = r"(?:度|°|deg(?:rees?)?)"
_NUM = r"(\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百点]+)"

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
}

_TOP_SPLIT = re.compile(
    r"[，,。；;！!？?\n]+"
    r"|\s*(?:然后|接着|随后|再(?!次)|之后)\s*"
    r"|\s+(?:and\s+then|then|after\s+that|afterwards)\s+",
    re.IGNORECASE,
)
_INNER_SPLIT = re.compile(
    r"\s+and\s+|\s*并(?:且)?\s*|\s*并且\s*|\s*然后\s*|\s*;\s*",
    re.IGNORECASE,
)
_LEADING = re.compile(
    r"^(?:再|然后|接着|随后|之后|又|并且|并|also|and|then)\s*",
    re.IGNORECASE,
)
_TRAILING = " \t\r\n。.!！?？,，;；"

_MOVE_CN = (
    (r"(?:上升|升高|向上(?:飞|移动)?|爬升)", "up"),
    (r"(?:下降|降低|向下(?:飞|移动)?)", "down"),
    (r"(?:向前(?:飞|移动)?|前进|往前(?:飞|移动)?)", "forward"),
    (r"(?:后退|向后(?:飞|移动)?|往后(?:飞|移动)?)", "backward"),
    (r"(?:向左(?:飞|移动)?|左移|往左(?:飞|移动)?)", "left"),
    (r"(?:向右(?:飞|移动)?|右移|往右(?:飞|移动)?)", "right"),
)
_MOVE_EN = (
    (r"(?:ascend|rise|climb|fly\s+up|go\s+up|move\s+up)", "up"),
    (r"(?:descend|sink|fly\s+down|go\s+down|move\s+down)", "down"),
    (r"(?:(?:fly|move|go|continue)\s+)?forward", "forward"),
    (r"(?:(?:fly|move|go)\s+)?back(?:ward)?", "backward"),
    (r"(?:(?:fly|move|go|strafe)\s+)?(?:to\s+the\s+)?left", "left"),
    (r"(?:(?:fly|move|go|strafe)\s+)?(?:to\s+the\s+)?right", "right"),
)

_APPROACH_CN = re.compile(
    r"^(?:飞到|飞向|飞近|接近|靠近|前往|移动到|移到)"
    r"\s*(?P<obj>它|目标|该目标|那个)?\s*(?:旁边|附近)?$",
    re.IGNORECASE,
)
_APPROACH_EN = re.compile(
    r"^(?:approach|fly\s+to|go\s+to|get\s+close\s+to|move\s+to)"
    r"\s*(?P<obj>it|the\s+target|that|cand_[A-Za-z0-9_-]+)?$",
    re.IGNORECASE,
)
_CONFIRM_CN = re.compile(
    r"^(?:确认|核实)(?:一下)?(?:\s*(?:它|目标|该目标|那个))?$", re.IGNORECASE
)
_CONFIRM_EN = re.compile(
    r"^confirm(?:\s+(?:it|the\s+target|that|the\s+object))?$", re.IGNORECASE
)
_VERIFY_CN = re.compile(
    r"^(?:验证|核验|最终验证)(?:一下)?(?:\s*(?:它|目标|该目标|那个))?$",
    re.IGNORECASE,
)
_VERIFY_EN = re.compile(
    r"^verify(?:\s+(?:it|the\s+target|the\s+goal|that))?$", re.IGNORECASE
)
_PHOTO_CN = re.compile(
    r"^(?:拍照|拍张照|拍个照|拍摄|照相|拍一张照片|拍张照片|拍个照片|拍一张照)$",
    re.IGNORECASE,
)
_PHOTO_EN = re.compile(
    r"^(?:take\s+(?:a\s+)?(?:photo|picture|snapshot|pic)"
    r"|snapshot|photograph)(?:\s+of\s+(?P<label>.+))?$",
    re.IGNORECASE,
)
_SEARCH_CN = re.compile(
    r"^(?:找到|寻找|搜寻|搜索|查找|找(?!到))\s*(?P<target>.+)$", re.IGNORECASE
)
_SEARCH_EN = re.compile(
    r"^(?:search\s+for|look\s+for|find|locate|search)\s+(?P<target>.+)$",
    re.IGNORECASE,
)
_REVISIT_CN = re.compile(
    r"^(?:返回|回到|重返|回访)\s*(?:到)?\s*(?P<place>.+)$", re.IGNORECASE
)
_REVISIT_EN = re.compile(
    r"^(?:return|go\s+back|come\s+back|revisit)"
    r"\s*(?:to\s+)?(?:the\s+)?(?P<place>.+)$",
    re.IGNORECASE,
)
_GOTO_CN = re.compile(
    r"^(?:飞到|前往|到达|移动到|去往)\s*(?:坐标)?\s*\(?\s*"
    + _NUM
    + r"\s*[,，]\s*"
    + _NUM
    + r"\s*[,，]\s*"
    + _NUM
    + r"\s*\)?$",
    re.IGNORECASE,
)
_GOTO_EN = re.compile(
    r"^(?:go|fly|move)\s+to\s*\(?\s*"
    + _NUM
    + r"\s*[,，]\s*"
    + _NUM
    + r"\s*[,，]\s*"
    + _NUM
    + r"\s*\)?$",
    re.IGNORECASE,
)
_CAPABILITY = {
    "takeoff": re.compile(r"^(?:起飞|升空|take\s*off|takeoff|lift\s*off)$", re.I),
    "land": re.compile(r"^(?:降落|着陆|land|touch\s*down)$", re.I),
    "orbit": re.compile(r"^(?:环绕|绕飞|orbit)$", re.I),
    "patrol": re.compile(r"^(?:巡检|巡逻|patrol)$", re.I),
}


class TaskPlannerError(ValueError):
    """An instruction clause the deterministic grammar does not understand."""


def _parse_number(token: str) -> float:
    text = str(token or "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if not text:
        raise TaskPlannerError("missing number")
    if "点" in text:
        whole, _, fraction = text.partition("点")
        whole_value = _parse_chinese_int(whole) if whole else 0
        digits = "".join(str(_CN_DIGITS[ch]) for ch in fraction if ch in _CN_DIGITS)
        return float(f"{whole_value}.{digits}") if digits else float(whole_value)
    return float(_parse_chinese_int(text))


def _parse_chinese_int(text: str) -> int:
    section = 0
    number = 0
    for char in text:
        if char in ("零", "〇"):
            continue
        if char not in _CN_DIGITS:
            raise TaskPlannerError(f"invalid number: {text!r}")
        value = _CN_DIGITS[char]
        if value >= 10:
            if number == 0:
                number = 1
            section += number * value
            number = 0
        else:
            number = number * 10 + value
    return section + number


def _clean_clause(clause: str) -> str:
    text = clause.strip().strip(_TRAILING)
    while True:
        cleaned = _LEADING.sub("", text).strip().strip(_TRAILING)
        if cleaned == text:
            return cleaned
        text = cleaned


def _strip_article(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)


def _search_step(target: str, default_target: str) -> dict:
    target = target.strip().strip(_TRAILING)
    if target.lower() in ("it", "the target") or target in ("它", "目标", "该目标"):
        if default_target:
            target = default_target
    target = _strip_article(target) if re.search(r"[A-Za-z]", target) else target
    relation = ""
    match = re.match(
        r"^(?P<anchor>.+?)(?:旁边|附近|旁)(?:的)?(?P<object>.+)$", target
    )
    if match:
        relation = match.group("anchor").strip()
        target = match.group("object").strip()
    else:
        match = re.match(r"^(?P<anchor>.+?边)的(?P<object>.+)$", target)
        if match:
            # Keep the full locative phrase as the target (schema example:
            # target="窗边的木椅", relation="窗边").
            relation = match.group("anchor").strip()
            object_name = match.group("object").strip()
            if not object_name:
                raise TaskPlannerError(f"search target is empty in {target!r}")
        else:
            match = re.match(
                r"^(?P<object>.+?)\s+(?:by|near|next\s+to|close\s+to)\s+"
                r"(?:the\s+)?(?P<anchor>.+)$",
                target,
                re.IGNORECASE,
            )
            if match:
                target = match.group("object").strip()
                relation = match.group("anchor").strip()
    if not target:
        raise TaskPlannerError("search target is empty")
    step: dict[str, Any] = {"kind": "search", "target": target}
    if relation:
        step["relation"] = relation
    return step


def _revisit_step(place: str) -> dict:
    place = place.strip().strip(_TRAILING)
    observation = re.fullmatch(
        r"(?:observation|观测)\s*[:#]?\s*(obs_[A-Za-z0-9_.-]+)", place, re.IGNORECASE
    )
    if observation:
        return {"kind": "revisit", "observation_id": observation.group(1)}
    viewpoint = re.fullmatch(
        r"(?:viewpoint|视角)\s*[:#]?\s*(vp_[A-Za-z0-9_.-]+)", place, re.IGNORECASE
    )
    if viewpoint:
        return {"kind": "revisit", "viewpoint_id": viewpoint.group(1)}
    if re.fullmatch(r"obs_[A-Za-z0-9_.-]+", place):
        return {"kind": "revisit", "observation_id": place}
    if re.fullmatch(r"vp_[A-Za-z0-9_.-]+", place):
        return {"kind": "revisit", "viewpoint_id": place}
    if not place:
        raise TaskPlannerError("revisit place is empty")
    return {"kind": "revisit", "place": place}


def _parse_single(clause: str, default_target: str) -> list[dict] | None:
    clause = _clean_clause(clause)
    if not clause:
        return None
    lowered = clause.lower()

    # Numeric goto first: "go to 1, 2, 1" must not be read as an approach.
    match = _GOTO_CN.fullmatch(clause) or _GOTO_EN.fullmatch(clause)
    if match:
        position = [
            _parse_number(match.group(index)) for index in (1, 2, 3)
        ]
        return [{"kind": "goto", "position": position}]

    for kind, pattern in _CAPABILITY.items():
        if pattern.fullmatch(clause):
            return [{"kind": kind}]

    # motion: relative moves, hover, turn
    for verbs in (_MOVE_CN, _MOVE_EN):
        for verb, direction in verbs:
            match = re.fullmatch(
                rf"{verb}(?:\s+(?:by|for))?\s*{_NUM}\s*{_LEN}", clause, re.IGNORECASE
            )
            if match:
                return [
                    {
                        "kind": "move",
                        "direction": direction,
                        "meters": _parse_number(match.group(1)),
                    }
                ]

    match = re.fullmatch(
        rf"(?:悬停|停留|保持悬停)\s*{_NUM}\s*{_TIME}", clause, re.IGNORECASE
    ) or re.fullmatch(
        rf"(?:hover|hold(?:\s+still)?)(?:\s+(?:for|in\s+place\s+for))?\s*"
        rf"{_NUM}\s*{_TIME}",
        clause,
        re.IGNORECASE,
    )
    if match:
        return [{"kind": "hover", "seconds": _parse_number(match.group(1))}]

    match = re.fullmatch(
        rf"(?P<dir>左转|右转|向左转|向右转)\s*(?P<amount>{_NUM})\s*{_ANGLE}",
        clause,
        re.IGNORECASE,
    )
    if match:
        direction = "left" if "左" in match.group("dir") else "right"
        return [
            {
                "kind": "turn",
                "direction": direction,
                "degrees": _parse_number(match.group("amount")),
            }
        ]
    match = re.fullmatch(
        rf"turn\s+(?P<dir>left|right)(?:\s+by)?\s*(?P<amount>{_NUM})\s*{_ANGLE}",
        clause,
        re.IGNORECASE,
    ) or re.fullmatch(
        rf"turn\s*(?P<amount>{_NUM})\s*{_ANGLE}\s*(?P<dir>left|right)",
        clause,
        re.IGNORECASE,
    )
    if match:
        return [
            {
                "kind": "turn",
                "direction": match.group("dir").lower(),
                "degrees": _parse_number(match.group("amount")),
            }
        ]

    # semantic steps
    if _VERIFY_CN.fullmatch(clause) or _VERIFY_EN.fullmatch(clause):
        return [{"kind": "verify", "target": "last"}]
    if _CONFIRM_CN.fullmatch(clause) or _CONFIRM_EN.fullmatch(clause):
        return [{"kind": "confirm", "target": "last"}]

    match = _SEARCH_CN.fullmatch(clause) or _SEARCH_EN.fullmatch(clause)
    if match:
        return [_search_step(match.group("target"), default_target)]

    match = _APPROACH_CN.fullmatch(clause) or _APPROACH_EN.fullmatch(clause)
    if match:
        obj = (match.groupdict().get("obj") or "").strip()
        target = obj if re.fullmatch(r"cand_[A-Za-z0-9_.-]+", obj) else "last"
        return [{"kind": "approach", "target": target}]

    match = _REVISIT_CN.fullmatch(clause) or _REVISIT_EN.fullmatch(clause)
    if match:
        return [_revisit_step(match.group("place"))]

    match = _PHOTO_CN.fullmatch(clause) or _PHOTO_EN.fullmatch(clause)
    if match:
        label = (match.groupdict().get("label") or "").strip()
        return [{"kind": "photo", "label": _strip_article(label) or "photo"}]

    # "take a photo" appears inside "and take a photo"; keep the phrasal verb
    # recognizable without swallowing a preceding action.
    return None


def _parse_clause(clause: str, default_target: str) -> list[dict] | None:
    # Try the conjunction split first: a search target would otherwise swallow
    # "and confirm it" greedily.  A split is accepted only when *both* sides are
    # complete actions, so "the chair and table" stays one target.
    for match in _INNER_SPLIT.finditer(clause):
        left = clause[: match.start()].strip()
        right = clause[match.end() :].strip()
        if not left or not right:
            continue
        left_steps = _parse_clause(left, default_target)
        if left_steps is None:
            continue
        right_steps = _parse_clause(right, default_target)
        if right_steps is None:
            continue
        return left_steps + right_steps
    return _parse_single(clause, default_target)


_POSITION_TRIPLE = re.compile(
    r"(\d+(?:\.\d+)?)(\s*)[,，](\s*)(\d+(?:\.\d+)?)(\s*)[,，]"
    r"(\s*)(\d+(?:\.\d+)?)"
)
_POSITION_MARK = "\x00\x01"


def _protect_positions(text: str) -> str:
    """Keep the commas of ``goto`` triples away from the clause splitter."""

    def replace(match: re.Match) -> str:
        return (
            match.group(1) + match.group(2) + _POSITION_MARK
            + match.group(3) + match.group(4) + match.group(5)
            + _POSITION_MARK + match.group(6) + match.group(7)
        )

    return _POSITION_TRIPLE.sub(replace, text)


def _split_clauses(instruction: str) -> list[str]:
    protected = _protect_positions(instruction)
    return [
        part.replace(_POSITION_MARK, ",")
        for part in _TOP_SPLIT.split(protected)
        if part and part.strip()
    ]


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise TaskPlannerError("VLM fallback did not return a JSON object")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise TaskPlannerError(f"VLM fallback JSON is invalid: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TaskPlannerError("VLM fallback JSON must be an object")
    return dict(data)


def _invoke_vlm(vlm: Any, prompt: str) -> str:
    if callable(vlm) and not hasattr(vlm, "generate"):
        result = vlm(prompt)
    else:
        method = None
        for name in ("generate", "complete", "chat", "ask"):
            candidate = getattr(vlm, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None:
            raise TaskPlannerError("vlm object is not callable and has no generate/complete method")
        result = method(prompt)
    if isinstance(result, Mapping):
        for key in ("text", "content", "response"):
            if isinstance(result.get(key), str):
                return result[key]
    return str(result)


def _vlm_fallback(instruction: str, vlm: Any) -> MixedMissionSpec:
    prompt = (
        "Convert this drone instruction into strict JSON for the "
        "drone.mixed_mission/1 schema. Return only JSON of the form "
        '{"steps": [...]} with ordered steps. Allowed step kinds and fields: '
        '{"kind":"move","direction":"forward|backward|left|right|up|down","meters":N}, '
        '{"kind":"turn","direction":"left|right","degrees":N}, '
        '{"kind":"hover","seconds":N}, {"kind":"photo","label":"..."}, '
        '{"kind":"search","target":"...","relation":"..."}, '
        '{"kind":"approach","target":"last"}, {"kind":"confirm","target":"last"}, '
        '{"kind":"verify","target":"last"}, '
        '{"kind":"revisit","place":"..."}, {"kind":"goto","position":[x,y,z]}, '
        '{"kind":"takeoff"}, {"kind":"land"}, {"kind":"orbit"}, {"kind":"patrol"}. '
        f"Instruction: {instruction}"
    )
    text = _invoke_vlm(vlm, prompt)
    data = _extract_json_object(text)
    try:
        spec = MixedMissionSpec.from_dict(data)
    except MixedMissionError as exc:
        raise TaskPlannerError(f"VLM fallback mission is invalid: {exc}") from exc
    return MixedMissionSpec(
        steps=spec.steps,
        mission_id=spec.mission_id,
        instruction=instruction,
    )


def plan_mission(
    instruction: str,
    *,
    vlm: Any = None,
    default_target: str = "",
) -> MixedMissionSpec:
    """Plan ``instruction`` into a validated :class:`MixedMissionSpec`.

    Deterministic grammar first.  On an unrecognized clause, one VLM fallback
    (when ``vlm`` is provided) may return a strict JSON mission; otherwise
    :class:`TaskPlannerError` names the clause.  The planner never silently
    drops or reorders an instruction.
    """
    text = str(instruction or "").strip()
    if not text:
        raise TaskPlannerError("instruction is empty")
    steps: list[dict] = []
    for clause in _split_clauses(text):
        parsed = _parse_clause(clause, default_target)
        if parsed is None:
            if vlm is not None:
                try:
                    return _vlm_fallback(text, vlm)
                except TaskPlannerError as exc:
                    raise TaskPlannerError(
                        f"unrecognized clause: {clause!r}; "
                        f"VLM fallback failed: {exc}"
                    ) from exc
            raise TaskPlannerError(f"unrecognized clause: {clause!r}")
        steps.extend(parsed)
    if not steps:
        raise TaskPlannerError(f"no actions found in instruction: {text!r}")
    try:
        return MixedMissionSpec(steps=tuple(steps), instruction=text)
    except MixedMissionError as exc:
        raise TaskPlannerError(f"planned mission is invalid: {exc}") from exc
