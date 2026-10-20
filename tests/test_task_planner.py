"""Deterministic Chinese/English mixed-mission planner (goal.md M2)."""

from __future__ import annotations

import unittest

from vlm_nav.mixed_mission import MixedMissionSpec
from vlm_nav.task_planner import TaskPlannerError, plan_mission

GOAL_CN = (
    "上升 1 米，向前飞 5 米，悬停 3 秒，右转 90 度，"
    "找到窗边的木椅，飞到它旁边，再向左移动 1 米并拍照。"
)
GOAL_EN = (
    "Ascend 1 meter, fly forward 5 meters, hover for 3 seconds, "
    "turn right 90 degrees, find the wooden chair by the window, "
    "fly to it, then move left 1 meter and take a photo."
)


class FakeVLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class GoalExampleTest(unittest.TestCase):
    def test_goal_md_chinese_example(self):
        spec = plan_mission(GOAL_CN)
        self.assertIsInstance(spec, MixedMissionSpec)
        self.assertEqual(spec.instruction, GOAL_CN)
        self.assertEqual(list(spec.steps), [
            {"kind": "move", "direction": "up", "meters": 1.0},
            {"kind": "move", "direction": "forward", "meters": 5.0},
            {"kind": "hover", "seconds": 3.0},
            {"kind": "turn", "direction": "right", "degrees": 90.0},
            {"kind": "search", "target": "窗边的木椅", "relation": "窗边"},
            {"kind": "approach", "target": "last"},
            {"kind": "move", "direction": "left", "meters": 1.0},
            {"kind": "photo", "label": "photo"},
        ])

    def test_english_equivalent(self):
        spec = plan_mission(GOAL_EN)
        self.assertEqual(list(spec.steps), [
            {"kind": "move", "direction": "up", "meters": 1.0},
            {"kind": "move", "direction": "forward", "meters": 5.0},
            {"kind": "hover", "seconds": 3.0},
            {"kind": "turn", "direction": "right", "degrees": 90.0},
            {"kind": "search", "target": "wooden chair", "relation": "window"},
            {"kind": "approach", "target": "last"},
            {"kind": "move", "direction": "left", "meters": 1.0},
            {"kind": "photo", "label": "photo"},
        ])

    def test_plan_roundtrips_as_mission_spec(self):
        spec = plan_mission(GOAL_CN)
        restored = MixedMissionSpec.from_dict(spec.to_dict())
        self.assertEqual(restored.to_dict(), spec.to_dict())


class VocabularyTest(unittest.TestCase):
    def _steps(self, instruction):
        return list(plan_mission(instruction).steps)

    def test_motion_vocabulary(self):
        self.assertEqual(
            self._steps("下降 2 米")[0],
            {"kind": "move", "direction": "down", "meters": 2.0},
        )
        self.assertEqual(
            self._steps("后退 1 米")[0],
            {"kind": "move", "direction": "backward", "meters": 1.0},
        )
        self.assertEqual(
            self._steps("向右移动 3 米")[0],
            {"kind": "move", "direction": "right", "meters": 3.0},
        )
        self.assertEqual(
            self._steps("turn left by 45 degrees")[0],
            {"kind": "turn", "direction": "left", "degrees": 45.0},
        )
        self.assertEqual(
            self._steps("turn 30 degrees right")[0],
            {"kind": "turn", "direction": "right", "degrees": 30.0},
        )

    def test_chinese_numerals(self):
        steps = self._steps("上升一米，向左移动两米，悬停三秒，左转九十度")
        self.assertEqual(steps[0]["meters"], 1.0)
        self.assertEqual(steps[1]["direction"], "left")
        self.assertEqual(steps[1]["meters"], 2.0)
        self.assertEqual(steps[2]["seconds"], 3.0)
        self.assertEqual(steps[3]["direction"], "left")
        self.assertEqual(steps[3]["degrees"], 90.0)

    def test_separators_then_and_inner_and(self):
        steps = self._steps("向前飞 2 米然后悬停 1 秒，再右转 45 度")
        self.assertEqual([step["kind"] for step in steps], ["move", "hover", "turn"])
        steps = self._steps(
            "move forward 2 meters and then hover for 1 second and take a photo"
        )
        self.assertEqual(
            [step["kind"] for step in steps], ["move", "hover", "photo"]
        )

    def test_semantic_vocabulary(self):
        steps = self._steps(
            "找到木椅，接近它，确认它，验证它，返回厨房，"
            "前往 1, 2, 1，起飞"
        )
        self.assertEqual(
            [step["kind"] for step in steps],
            ["search", "approach", "confirm", "verify", "revisit", "goto",
             "takeoff"],
        )
        self.assertEqual(steps[4], {"kind": "revisit", "place": "厨房"})
        self.assertEqual(steps[5]["position"], [1.0, 2.0, 1.0])

    def test_capability_vocabulary(self):
        for text, kind in (
            ("take off", "takeoff"),
            ("land", "land"),
            ("orbit", "orbit"),
            ("patrol", "patrol"),
            ("降落", "land"),
            ("环绕", "orbit"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self._steps(text)[0], {"kind": kind})

    def test_relation_extraction(self):
        steps = self._steps(
            "find the red chair next to the window"
        )
        self.assertEqual(steps[0]["target"], "red chair")
        self.assertEqual(steps[0]["relation"], "window")
        steps = self._steps("找到木椅旁边的窗户")
        self.assertEqual(steps[0]["target"], "窗户")
        self.assertEqual(steps[0]["relation"], "木椅")

    def test_default_target_used_for_pronoun(self):
        steps = self._steps_target("find it", "木椅")
        self.assertEqual(steps[0]["target"], "木椅")

    def _steps_target(self, instruction, target):
        return list(
            plan_mission(instruction, default_target=target).steps
        )

    def test_photo_label_from_english_phrase(self):
        steps = self._steps("take a photo of the window chair")
        self.assertEqual(steps[0], {"kind": "photo", "label": "window chair"})


class ErrorAndFallbackTest(unittest.TestCase):
    def test_unrecognized_clause_raises_and_names_it(self):
        with self.assertRaises(TaskPlannerError) as caught:
            plan_mission("向前飞 1 米，倒立三秒")
        self.assertIn("倒立三秒", str(caught.exception))

    def test_empty_instruction_raises(self):
        with self.assertRaises(TaskPlannerError):
            plan_mission("   ")

    def test_partial_vocabulary_is_not_guessed(self):
        for text in ("向左平移五米", "go to the moon", "找到"):
            with self.subTest(text=text):
                with self.assertRaises(TaskPlannerError):
                    plan_mission(text)

    def test_vlm_fallback_used_once_for_whole_instruction(self):
        reply = (
            '```json\n{"steps": ['
            '{"kind": "move", "direction": "forward", "meters": 2},'
            '{"kind": "photo", "label": "fallback"}'
            ']}\n```'
        )
        vlm = FakeVLM(reply)
        spec = plan_mission("perform the thing", vlm=vlm)
        self.assertEqual(len(vlm.prompts), 1)
        self.assertEqual(spec.instruction, "perform the thing")
        self.assertEqual(list(spec.steps), [
            {"kind": "move", "direction": "forward", "meters": 2.0},
            {"kind": "photo", "label": "fallback"},
        ])

    def test_vlm_fallback_invalid_json_is_reported(self):
        vlm = FakeVLM("I cannot do that")
        with self.assertRaises(TaskPlannerError) as caught:
            plan_mission("perform the thing", vlm=vlm)
        message = str(caught.exception)
        self.assertIn("unrecognized clause", message)
        self.assertIn("VLM fallback failed", message)

    def test_vlm_fallback_validates_step_schema(self):
        vlm = FakeVLM('{"steps": [{"kind": "walk", "meters": 2}]}')
        with self.assertRaises(TaskPlannerError):
            plan_mission("perform the thing", vlm=vlm)


if __name__ == "__main__":
    unittest.main()
