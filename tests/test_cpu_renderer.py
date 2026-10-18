"""Depth-order regressions for the pure-torch CPU splat renderer.

``CpuSplatRenderer._splat`` emits its contributions bin by bin, in *radius*
order, and ``_composite`` used to sort them by pixel id only. The front-to-back
order inside a pixel therefore held only within one radius bin: a Gaussian that
projects a large splat (a near wall) landed in a later bin than a small one (a
distant target) and was composited *after* it, so the distance was lost and the
target behind the wall stayed visible. The M2 real-model experiment hit this -
64x64 frames from the loss pose showed the walled-off target (83 px at yaw 45,
52 px at yaw 90) although the analytic slab test says it is occluded.
"""

import math
import struct
import unittest

import numpy as np
import torch

from vlm_nav.cpu_renderer import (
    CameraSpec,
    CpuGaussianScene,
    CpuSplatRenderer,
    yaw_to_quat_xyzw,
)

WIDTH = HEIGHT = 64
HFOV_DEG = 60.0
# FLU -> 3DGS, the convention the loader/renderer use (see cpu_renderer header)
FLU_TO_3DGS = np.asarray([[0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0]])


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _write_scene_ply(path, gaussians_flu):
    """Write authored-FLU gaussians the way ``CpuGaussianScene`` reads them.

    The loader takes PLY vertices as absolute 3DGS coordinates and puts the FLU
    origin at their centroid, so the authored geometry is rotated into the 3DGS
    frame around its own FLU centroid. The centroid is returned: a pose authored
    at FLU ``p`` is rendered by passing ``p - centroid`` (a rigid translation,
    nothing about the scene's geometry changes).
    """
    positions = np.asarray([xyz for xyz, *_ in gaussians_flu], dtype=np.float64)
    origin = positions.mean(axis=0)
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(gaussians_flu)}",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    with open(path, "wb") as handle:
        handle.write((header + "\n").encode("ascii"))
        for xyz, sh, opacity, scale, rot in gaussians_flu:
            local = (np.asarray(xyz, dtype=np.float64) - origin) @ FLU_TO_3DGS.T
            handle.write(struct.pack("<14f", *local.tolist(), *sh, opacity,
                                     *scale, *rot))
    return origin


def _wall_splats(x_m: float = 1.0, half: float = 0.9, spacing: float = 0.15):
    """Dense opaque slab facing the camera at FLU x = ``x_m``."""
    out = []
    steps = np.linspace(-half, half, int(round(2 * half / spacing)) + 1)
    for y in steps:
        for z in steps:
            out.append(((x_m, float(y), float(z)), (0.0, 0.0, 0.0),
                        _logit(0.95), (math.log(0.22),) * 3, (1.0, 0.0, 0.0, 0.0)))
    return out


def _target_splats(x_m: float = 3.0):
    """Small red cluster far behind the wall (SH0 red, green/blue off)."""
    return [
        ((x_m + dx, dy, dz), (2.0, -1.0, -1.0), _logit(0.9),
         (math.log(0.15),) * 3, (1.0, 0.0, 0.0, 0.0))
        for dx, dy, dz in ((0.0, 0.0, 0.0), (0.05, 0.03, 0.02), (-0.03, -0.04, 0.03))
    ]


def _shift_splats(splats, offset):
    """Translate authored-FLU gaussians by an FLU offset."""
    ox, oy, oz = offset
    return [
        ((x + ox, y + oy, z + oz), sh, opacity, scale, rot)
        for (x, y, z), sh, opacity, scale, rot in splats
    ]


def _red_pixels(image: torch.Tensor) -> np.ndarray:
    """Mask of pixels whose colour can only come from the target cluster."""
    arr = image.permute(1, 2, 0).numpy()
    red, green, blue = arr[..., 0], arr[..., 1], arr[..., 2]
    return (red > 0.30) & (red - green > 0.18) & (red - blue > 0.18)


class CpuRendererDepthOrderTest(unittest.TestCase):
    """What is behind the wall must not be visible from the wall's far side."""

    def setUp(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ply_path = f"{tmp.name}/wall_target.ply"

    def _renderer(self, gaussians_flu):
        self.origin = _write_scene_ply(self.ply_path, gaussians_flu)
        scene = CpuGaussianScene(self.ply_path)
        renderer = CpuSplatRenderer(scene, CameraSpec(WIDTH, HEIGHT, HFOV_DEG))
        return renderer

    def _render(self, renderer, cam_flu, yaw_deg=0.0):
        pose = np.asarray(cam_flu, dtype=np.float64) - self.origin
        return renderer.render(pose, yaw_to_quat_xyzw(math.radians(yaw_deg)))

    def test_target_behind_opaque_wall_is_occluded(self):
        # camera 1 m in front of the wall (radius bin 24), target 3 m away
        # (bin 12): radius order alone would composite the target first
        renderer = self._renderer(_wall_splats() + _target_splats())
        image = self._render(renderer, (0.0, 0.0, 0.0))
        # the wall fills the frame (a black frame would pass a leak test for free)
        self.assertGreater(float(image.mean()), 0.05)
        self.assertEqual(int(_red_pixels(image).sum()), 0)

    def test_target_visible_when_the_wall_is_behind_the_camera(self):
        # same scene, camera between the wall and the target: the wall is behind
        # the camera, so nothing hides the target any more
        renderer = self._renderer(_wall_splats() + _target_splats())
        image = self._render(renderer, (2.0, 0.0, 0.0))
        self.assertGreater(int(_red_pixels(image).sum()), 20)

    def test_target_renders_without_any_occluder(self):
        renderer = self._renderer(_target_splats())
        image = self._render(renderer, (0.0, 0.0, 0.0))
        self.assertGreater(int(_red_pixels(image).sum()), 20)

    def test_far_small_splat_does_not_paint_over_the_near_one(self):
        # minimal case: two splats on the optical axis, both landing on the
        # centre pixel. The near one projects a 24 px splat (radius bin 24), the
        # far red one a ~2 px splat (bin 2). Radius order alone composites the
        # far one first and lets it paint over the near grey; in depth order the
        # near splat (SH0 grey 0.5) is composited first and the far target colour
        # reaches the pixel only through the 5% it transmits.
        near = [((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), _logit(0.95),
                 (math.log(1.0),) * 3, (1.0, 0.0, 0.0, 0.0))]
        far = [((3.0, 0.0, 0.0), (2.0, -1.0, -1.0), _logit(0.95),
                (math.log(0.02),) * 3, (1.0, 0.0, 0.0, 0.0))]
        renderer = self._renderer(near + far)
        image = self._render(renderer, (0.0, 0.0, 0.0))
        center = image[:, HEIGHT // 2, WIDTH // 2]
        # the far cluster's red-dominance survives only as a chroma leak
        self.assertLess(float(center[0] - center[1]), 0.1)


class CpuRendererDepthTest(unittest.TestCase):
    """``render_with_depth`` must report the depth of what the pixel *shows*.

    Downstream (``SplatWorldCPU.depth_at_pixel`` -> ``ground_target``) turns a
    normalized pixel plus this depth into a world position, so an occluded pixel
    that reported the hidden target's range instead of the occluder's would send
    the drone through the occluder.
    """

    def setUp(self):
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ply_path = f"{tmp.name}/scene.ply"

    def _render(self, gaussians_flu, cam_flu=(0.0, 0.0, 0.0), yaw_deg=0.0):
        """Render one view of an authored-FLU scene: (rgb [3,H,W], depth [H,W])."""
        origin = _write_scene_ply(self.ply_path, gaussians_flu)
        scene = CpuGaussianScene(self.ply_path)
        renderer = CpuSplatRenderer(scene, CameraSpec(WIDTH, HEIGHT, HFOV_DEG))
        pose = np.asarray(cam_flu, dtype=np.float64) - origin
        return renderer.render_with_depth(pose, yaw_to_quat_xyzw(math.radians(yaw_deg)))

    def test_depth_is_the_axial_distance_not_the_euclidean_range(self):
        # 3 m ahead and 1 m to the left: axial 3.00 m, range 3.16 m. A reader
        # that returned the range would overshoot a pixel-aimed waypoint.
        _, depth = self._render(_shift_splats(_target_splats(3.0), (0.0, 1.0, 0.0)))

        hit = depth > 0.0
        self.assertGreater(int(hit.sum()), 4)
        self.assertAlmostEqual(float(depth[hit].median()), 3.0, delta=0.05)

    def test_pixels_that_no_gaussian_reaches_have_no_return(self):
        _, depth = self._render(_target_splats())

        self.assertGreater(int((depth > 0.0).sum()), 4)
        self.assertEqual(float(depth[0, 0]), 0.0)
        # the cluster is small: most of the frame is empty space, not a surface
        self.assertGreater(float((depth == 0.0).float().mean()), 0.5)

    def test_an_occluded_pixel_reads_the_occluder(self):
        # the target's own footprint in exactly this camera (same pose)
        _, target_depth = self._render(_target_splats())
        target_px = target_depth > 0.0
        self.assertGreater(int(target_px.sum()), 4)

        image, depth = self._render(_wall_splats() + _target_splats())
        behind_the_wall = depth[target_px]
        # the wall stands at x = 1 m, the target at 3 m: the hidden pixels read
        # the wall, and never the target behind it
        self.assertAlmostEqual(float(behind_the_wall.median()), 1.0, delta=0.05)
        self.assertLess(float(behind_the_wall.max()), 1.1)
        self.assertEqual(int(_red_pixels(image).sum()), 0)

    def test_the_same_pixels_read_the_target_once_the_wall_is_behind(self):
        _, target_depth = self._render(_target_splats())
        target_px = target_depth > 0.0

        _, depth = self._render(
            _wall_splats() + _target_splats(), cam_flu=(2.0, 0.0, 0.0)
        )
        # camera at 2 m: 1 m to the target, the wall 1 m behind the camera
        self.assertAlmostEqual(float(depth[target_px].median()), 1.0, delta=0.05)


class CpuMirrorSceneTest(unittest.TestCase):
    """``CpuGaussianScene.from_gaussian_model`` mirrors an activated model.

    ``SplatWorldHiGS`` renders on the GPU without a depth output, so its depth
    reads run through this mirror; it has to project the same geometry as the
    PLY-loaded scene does.
    """

    def test_the_mirror_renders_the_same_view_as_the_loaded_scene(self):
        import tempfile
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            ply_path = f"{tmp}/scene.ply"
            _write_scene_ply(ply_path, _wall_splats(x_m=2.0) + _target_splats(4.0))
            scene = CpuGaussianScene(ply_path)
            # GaussianModelLite carries exactly these activated attributes
            model = SimpleNamespace(
                center=scene.center,
                xyz=scene.xyz,
                scale=scene.scale,
                rot=scene.rot,
                opacity=scene.opacity,
                sh=((scene.color - 0.5) / 0.28209479177387814).unsqueeze(1),
            )
            mirror = CpuGaussianScene.from_gaussian_model(model)

            camera = CameraSpec(WIDTH, HEIGHT, HFOV_DEG)
            pose, quat = torch.zeros(3), yaw_to_quat_xyzw(math.radians(20.0))
            image, depth = CpuSplatRenderer(scene, camera).render_with_depth(pose, quat)
            mirror_image, mirror_depth = CpuSplatRenderer(mirror, camera).render_with_depth(
                pose, quat
            )

        self.assertGreater(float(depth.sum()), 0.0)
        torch.testing.assert_close(mirror_depth, depth)
        torch.testing.assert_close(mirror_image, image)


if __name__ == "__main__":
    unittest.main()
