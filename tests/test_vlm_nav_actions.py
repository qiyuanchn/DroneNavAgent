"""Regression tests for primitive-action parsing in vlm_nav.actions.

The categorical primitives (forward / up / turn_left / ...) used to drop the
point-to-navigate fields ("pixel", "distance_m") that the move and waypoint
branches keep; the harness upgrades a move-with-pixel into a waypoint, so the
loss silently disabled waypoint navigation. These tests pin the fix and guard
the move / waypoint behaviour against regression.
"""

import unittest

from vlm_nav.actions import (
    MAX_TRANSLATE_M,
    MAX_WAYPOINT_TRANSLATE_M,
    action_to_body_delta,
    parse_action,
)

_MOVE_PRIMITIVES = ("forward", "backward", "back", "left", "right", "up", "down")
_TURN_PRIMITIVES = ("turn_left", "turn_right")
_NUMERIC_FIELDS = ("forward_m", "right_m", "up_m", "yaw_deg", "distance_m")


def _numeric_tuple(action):
    return tuple(getattr(action, field) for field in _NUMERIC_FIELDS)


class PrimitivePixelRetentionTest(unittest.TestCase):
    def test_forward_keeps_pixel_and_distance(self):
        action = parse_action(
            '{"action": "forward", "meters": 1.0, "pixel": [500, 320], '
            '"distance_m": 3.0, "reason": "door ahead"}'
        )
        self.assertTrue(action.parse_ok)
        self.assertEqual(action.kind, "move")
        self.assertAlmostEqual(action.forward_m, 1.0)
        self.assertAlmostEqual(action.right_m, 0.0)
        self.assertAlmostEqual(action.up_m, 0.0)
        self.assertEqual(action.pixel, (500.0, 320.0))
        self.assertAlmostEqual(action.distance_m, 3.0)
        self.assertEqual(action.reason, "door ahead")

    def test_turn_left_keeps_pixel(self):
        action = parse_action(
            '{"action": "turn_left", "degrees": 45, "pixel": [100, 950]}'
        )
        self.assertTrue(action.parse_ok)
        self.assertEqual(action.kind, "turn")
        self.assertAlmostEqual(action.yaw_deg, 45.0)
        self.assertEqual(action.pixel, (100.0, 950.0))

    def test_every_primitive_keeps_pixel_and_distance(self):
        for primitive in _MOVE_PRIMITIVES + _TURN_PRIMITIVES:
            with self.subTest(primitive=primitive):
                action = parse_action(
                    '{"action": "%s", "pixel": [500, 320], "distance_m": 2.5}'
                    % primitive
                )
                self.assertTrue(action.parse_ok)
                self.assertEqual(action.pixel, (500.0, 320.0))
                self.assertAlmostEqual(action.distance_m, 2.5)

    def test_primitive_without_pixel_is_unchanged(self):
        # no point-to-navigate fields -> defaults, and the motion values still
        # match the pre-fix behaviour
        action = parse_action('{"action": "forward", "meters": 1.0}')
        self.assertIsNone(action.pixel)
        self.assertAlmostEqual(action.distance_m, 0.0)
        self.assertAlmostEqual(action.forward_m, 1.0)
        action = parse_action('{"action": "up"}')
        self.assertIsNone(action.pixel)
        self.assertAlmostEqual(action.up_m, 0.8)  # default meters
        action = parse_action('{"action": "turn_right"}')
        self.assertIsNone(action.pixel)
        self.assertAlmostEqual(action.yaw_deg, -30.0)  # default degrees


class PrimitiveClampTest(unittest.TestCase):
    def test_distance_clamped_to_waypoint_limit(self):
        action = parse_action(
            '{"action": "forward", "meters": 1.0, "pixel": [500, 320], '
            '"distance_m": 99.0}'
        )
        self.assertAlmostEqual(action.distance_m, MAX_WAYPOINT_TRANSLATE_M)

    def test_turn_distance_clamped_to_waypoint_limit(self):
        action = parse_action(
            '{"action": "turn_left", "pixel": [10, 20], "distance_m": -42.0}'
        )
        self.assertAlmostEqual(action.distance_m, -MAX_WAYPOINT_TRANSLATE_M)

    def test_meters_still_clamped_to_translate_limit(self):
        action = parse_action(
            '{"action": "forward", "meters": 99.0, "pixel": [500, 320], '
            '"distance_m": 99.0}'
        )
        self.assertAlmostEqual(action.forward_m, MAX_TRANSLATE_M)
        self.assertAlmostEqual(action.distance_m, MAX_WAYPOINT_TRANSLATE_M)

    def test_malformed_pixel_degrades_to_none(self):
        for pixel in ('"center"', "[1, 2, 3]", "[1]", "[null, 2]", "null"):
            with self.subTest(pixel=pixel):
                action = parse_action(
                    '{"action": "forward", "pixel": %s, "distance_m": 4.0}' % pixel
                )
                self.assertIsNone(action.pixel)
                self.assertAlmostEqual(action.distance_m, 4.0)


class MoveAndWaypointRegressionTest(unittest.TestCase):
    def test_move_unchanged(self):
        action = parse_action(
            '{"action": "move", "forward_m": 1.5, "right_m": -0.5, "yaw_deg": 10, '
            '"pixel": [480, 300], "distance_m": 3.0, "reason": "target ahead"}'
        )
        self.assertTrue(action.parse_ok)
        self.assertEqual(action.kind, "move")
        self.assertAlmostEqual(action.forward_m, 1.5)
        self.assertAlmostEqual(action.right_m, -0.5)
        self.assertAlmostEqual(action.yaw_deg, 10.0)
        self.assertEqual(action.pixel, (480.0, 300.0))
        self.assertAlmostEqual(action.distance_m, 3.0)

    def test_move_clamping_unchanged(self):
        action = parse_action(
            '{"action": "move", "forward_m": 99.0, "yaw_deg": -180.0, '
            '"distance_m": 99.0}'
        )
        self.assertAlmostEqual(action.forward_m, MAX_TRANSLATE_M)
        self.assertAlmostEqual(action.yaw_deg, -45.0)
        self.assertAlmostEqual(action.distance_m, MAX_WAYPOINT_TRANSLATE_M)

    def test_waypoint_unchanged(self):
        action = parse_action(
            '{"action": "waypoint", "forward_m": 6.0, "right_m": -3.0, '
            '"yaw_deg": 120, "pixel": [500, 320], "distance_m": 3.0}'
        )
        self.assertEqual(action.kind, "waypoint")
        self.assertAlmostEqual(action.forward_m, 6.0)
        self.assertAlmostEqual(action.right_m, -3.0)
        self.assertAlmostEqual(action.yaw_deg, 120.0)
        self.assertEqual(action.pixel, (500.0, 320.0))
        self.assertAlmostEqual(action.distance_m, 3.0)
        action = parse_action(
            '{"action": "waypoint", "forward_m": 99.0, "right_m": -99.0, '
            '"up_m": 99.0, "yaw_deg": 999.0, "distance_m": 99.0}'
        )
        self.assertAlmostEqual(action.forward_m, MAX_WAYPOINT_TRANSLATE_M)
        self.assertAlmostEqual(action.right_m, -MAX_WAYPOINT_TRANSLATE_M)
        self.assertAlmostEqual(action.up_m, MAX_WAYPOINT_TRANSLATE_M)
        self.assertAlmostEqual(action.yaw_deg, 180.0)
        self.assertAlmostEqual(action.distance_m, MAX_WAYPOINT_TRANSLATE_M)

    def test_unparseable_still_hovers(self):
        action = parse_action("I think the book is over there!")
        self.assertFalse(action.parse_ok)
        self.assertEqual(action.kind, "hover")
        self.assertIsNone(action.pixel)
        self.assertAlmostEqual(action.distance_m, 0.0)

    def test_pixel_only_forward_is_waypoint_ready(self):
        # the harness turns a move-with-pixel into a waypoint; the parsed
        # action must therefore expose both fields unchanged
        action = parse_action('{"action": "forward", "pixel": [640, 360]}')
        self.assertEqual(action.kind, "move")
        self.assertEqual(action.pixel, (640.0, 360.0))
        dx, dy, dz, dyaw = action_to_body_delta(action)
        self.assertAlmostEqual(dx, 0.8)  # default meters, forward
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dz, 0.0)
        self.assertAlmostEqual(dyaw, 0.0)
        self.assertAlmostEqual(_numeric_tuple(action)[0], 0.8)


if __name__ == "__main__":
    unittest.main()
