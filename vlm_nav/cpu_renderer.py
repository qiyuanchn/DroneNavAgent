"""Pure-torch CPU 3D Gaussian Splatting renderer.

The production sensor (`sensor/gs_cameras.py`, HiGS backend) requires CUDA.
This module reproduces its camera conventions and SH0 RGB-only rendering model
on CPU so the VLM navigation harness can run on machines without a GPU.

Conventions (identical to `utils/gaussian_splatting/runtime.py` and the C17
notes in `docs/3dgs_renderer_experiments_20260805.md`):
- World/simulation frame is FLU (+x forward, +y left, +z up), origin at the
  PLY vertex centroid.
- Absolute 3DGS coordinates: `flu_to_3dgs(p) + center` with
  `flu_to_3dgs(x, y, z) = (-y, z, -x)`.
- Camera rotation in the 3DGS frame for a yaw-only drone pose is
  `Ry(yaw) @ diag(1, -1, -1)`; the general quaternion path mirrors
  `GaussianSplattingCamera.update_transformation`.
- Color is SH degree 0 only: `rgb = sh_dc * 0.28209479177387814 + 0.5`.

Rendering model: EWA splatting with a 3-sigma support, per-pixel k-buffer
filled in exact front-to-back order (every contribution carries the depth rank
of its Gaussian, and the contributions of a pixel are re-ordered by it before
compositing), then composited with the standard over operator. PLY attributes
are activated exactly like `GaussianModelLite` (sigmoid opacity, exp scale,
normalized wxyz quaternion).

``render_with_depth`` additionally returns the per-pixel axial depth (the
camera-frame ``z`` of the composited Gaussians, in scene units - metres for
metric scenes): the same traversal that produces the RGB produces the depth, so
a pixel whose view is blocked by a near surface reads the near surface's depth,
not the depth of whatever it hides.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SH_C0 = 0.28209479177387814
ZNEAR = 0.01
ZFAR = 100.0
EPS_2D = 0.3
MIN_ALPHA = 1.0 / 255.0
MAX_CONIC_POWER = 8.0
K_BUFFER = 24
MAX_RADIUS_PX = 24
RADIUS_BINS = (1, 2, 3, 5, 8, 12, 16, 24)


def flu_to_3dgs(pos_flu: Tensor) -> Tensor:
    x_flu, y_flu, z_flu = pos_flu.unbind(dim=-1)
    return torch.stack([-y_flu, z_flu, -x_flu], dim=-1)


def _axis_rotmat(axis: str, angle: Tensor) -> Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return torch.stack(flat, dim=-1).reshape(*angle.shape, 3, 3)


def _quat_xyzw_to_euler(quat_xyzw: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    quat = quat_xyzw / quat_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    pitch = torch.asin((2.0 * (w * y - x * z)).clamp(-1.0 + 1e-8, 1.0 - 1e-8))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    return roll, pitch, yaw


def _quat_wxyz_to_matrix(rot_wxyz: Tensor) -> Tensor:
    q = rot_wxyz / rot_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def yaw_to_quat_xyzw(yaw_rad: float) -> np.ndarray:
    return np.asarray([0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)])


def camera_rotation_3dgs(quat_xyzw: Tensor) -> Tensor:
    """Camera rotation in the 3DGS frame, mirroring update_transformation."""
    roll, pitch, yaw = _quat_xyzw_to_euler(quat_xyzw)
    ypr = torch.stack([yaw, -pitch, -roll], dim=-1)
    R = (
        _axis_rotmat("Y", ypr[..., 0])
        @ _axis_rotmat("X", ypr[..., 1])
        @ _axis_rotmat("Z", ypr[..., 2])
    )
    R[..., :, 1] = -R[..., :, 1]
    R[..., :, 2] = -R[..., :, 2]
    return R


@dataclass
class CameraSpec:
    width: int = 320
    height: int = 180
    hfov_deg: float = 86.0

    @property
    def tan_x(self) -> float:
        return math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def tan_y(self) -> float:
        return self.tan_x * self.height / self.width

    @property
    def fx(self) -> float:
        return self.width / (2.0 * self.tan_x)

    @property
    def fy(self) -> float:
        return self.height / (2.0 * self.tan_y)


class CpuGaussianScene:
    """Activated Gaussian attributes loaded from a 3DGS PLY (SH0).

    Loading is memory-lean (memmap + chunked activation) so multi-million
    Gaussian scenes fit in small containers: peak usage stays close to the
    final float32 tensors (~56 bytes per kept Gaussian).
    """

    def __init__(self, ply_path: str | Path, scale: float = 1.0,
                 min_opacity: float = 0.0, chunk_rows: int = 1_000_000):
        ply_path = Path(ply_path)
        names, count, offset = _read_ply_header(ply_path)
        dtype = np.dtype([(name, "<f4") for name in names])
        mm = np.memmap(ply_path, dtype=dtype, mode="r", offset=offset, shape=(count,))

        xyz_parts: list[np.ndarray] = []
        sh_parts: list[np.ndarray] = []
        opacity_parts: list[np.ndarray] = []
        scale_parts: list[np.ndarray] = []
        rot_parts: list[np.ndarray] = []
        for lo in range(0, count, chunk_rows):
            hi = min(lo + chunk_rows, count)
            block = mm[lo:hi]
            xyz = np.stack([block["x"], block["y"], block["z"]], axis=1)
            rot = np.stack(
                [block["rot_0"], block["rot_1"], block["rot_2"], block["rot_3"]],
                axis=1,
            )
            gscale = np.stack(
                [block["scale_0"], block["scale_1"], block["scale_2"]], axis=1
            )
            opacity = np.asarray(block["opacity"])
            keep = (
                np.isfinite(xyz).all(axis=1)
                & np.isfinite(rot).all(axis=1)
                & np.isfinite(gscale).all(axis=1)
                & np.isfinite(opacity)
            )
            if min_opacity > 0.0:
                keep &= opacity >= _logit_np(min_opacity)
            if not bool(keep.any()):
                continue
            sh = np.stack(
                [block["f_dc_0"], block["f_dc_1"], block["f_dc_2"]], axis=1
            )
            xyz_parts.append(np.ascontiguousarray(xyz[keep]))
            sh_parts.append(np.ascontiguousarray(sh[keep]))
            opacity_parts.append(np.ascontiguousarray(opacity[keep]))
            scale_parts.append(np.ascontiguousarray(gscale[keep]))
            rot_parts.append(np.ascontiguousarray(rot[keep]))
        if not xyz_parts:
            raise ValueError(f"no usable Gaussians in {ply_path}")

        xyz_np = np.concatenate(xyz_parts)
        self.center = torch.from_numpy(xyz_np.mean(axis=0, dtype=np.float64).astype(np.float32))
        self.xyz = torch.from_numpy(xyz_np)
        self.xyz.sub_(self.center).mul_(float(scale)).add_(self.center)
        self.color = torch.clamp_min(torch.from_numpy(np.concatenate(sh_parts)) * SH_C0 + 0.5, 0.0)
        self.opacity = torch.sigmoid(torch.from_numpy(np.concatenate(opacity_parts)))
        self.scale = torch.exp(torch.from_numpy(np.concatenate(scale_parts))) * float(scale)
        rot_t = torch.from_numpy(np.concatenate(rot_parts))
        self.rot = rot_t / rot_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.radius = float((self.xyz - self.center).norm(dim=-1).max())

    def __len__(self) -> int:
        return self.xyz.shape[0]

    @classmethod
    def from_gaussian_model(cls, model) -> "CpuGaussianScene":
        """CPU mirror of an already-activated Gaussian model (``GaussianModelLite``).

        The attributes are read with the meaning that model gives them
        (absolute 3DGS ``xyz`` already scaled about the scene centre, activated
        ``scale``/``opacity``, normalized ``wxyz`` rotation, SH DC term), so the
        mirror rasterises the same geometry as the CUDA backend. Backends whose
        renderer exposes no depth use it to answer depth reads on CPU.
        """
        scene = cls.__new__(cls)
        scene.center = model.center.detach().to("cpu", torch.float32).clone()
        scene.xyz = model.xyz.detach().to("cpu", torch.float32).clone()
        scene.color = torch.clamp_min(
            model.sh[:, 0].detach().to("cpu", torch.float32) * SH_C0 + 0.5, 0.0
        )
        scene.opacity = (
            model.opacity.detach().to("cpu", torch.float32).reshape(-1).clone()
        )
        scene.scale = model.scale.detach().to("cpu", torch.float32).clone()
        rot = model.rot.detach().to("cpu", torch.float32)
        scene.rot = rot / rot.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scene.radius = float((scene.xyz - scene.center).norm(dim=-1).max())
        return scene


def _logit_np(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def _read_ply_header(path: Path) -> tuple[list[str], int, int]:
    names: list[str] = []
    count = 0
    with open(path, "rb") as handle:
        magic = handle.readline().decode("ascii", "replace").strip()
        if magic != "ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            line = handle.readline().decode("ascii", "replace").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                raise ValueError(f"unsupported PLY format: {line}")
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("element") and not line.startswith("element vertex"):
                raise ValueError(f"unsupported extra element in PLY: {line}")
            elif line.startswith("property"):
                parts = line.split()
                if parts[1] != "float":
                    raise ValueError(f"unsupported PLY property type: {line}")
                names.append(parts[2])
            elif line.startswith("end_header"):
                break
        return names, count, handle.tell()


class CpuSplatRenderer:
    """Single-view SH0 EWA splatter with an exact-order per-pixel k-buffer."""

    def __init__(self, scene: CpuGaussianScene, camera: CameraSpec | None = None):
        self.scene = scene
        self.camera = camera or CameraSpec()

    @torch.no_grad()
    def render(self, pos_flu: Tensor | np.ndarray, quat_xyzw: Tensor | np.ndarray) -> Tensor:
        """Render one view. Returns float32 RGB [3, H, W] in [0, 1]."""
        return self.render_with_depth(pos_flu, quat_xyzw)[0]

    @torch.no_grad()
    def render_with_depth(
        self, pos_flu: Tensor | np.ndarray, quat_xyzw: Tensor | np.ndarray
    ) -> tuple[Tensor, Tensor]:
        """Render one view as ``(rgb, depth)``.

        ``depth`` is float32 ``[H, W]`` in scene units (metres for a metric
        scene): the transmittance-weighted mean axial depth of the composited
        Gaussians, i.e. the distance along the camera's forward axis - the same
        axial convention ``world.pixel_to_body_offset`` uses - and 0 for pixels
        no Gaussian reached ("no return"). Occlusion is already resolved by the
        k-buffer, so a pixel behind an opaque surface reads that surface.
        """
        pos_flu = torch.as_tensor(pos_flu, dtype=torch.float32).reshape(3)
        quat = torch.as_tensor(quat_xyzw, dtype=torch.float32).reshape(4)
        cam = self.camera
        scene = self.scene

        R = camera_rotation_3dgs(quat)
        cam_center = flu_to_3dgs(pos_flu) + scene.center
        view = (scene.xyz - cam_center) @ R
        z = view[:, 2]

        fx, fy = cam.fx, cam.fy
        keep = (z > ZNEAR) & (z < ZFAR)
        # conservative frustum cull with a 3-sigma margin
        sigma_max = scene.scale.max(dim=-1).values
        margin = 3.0 * sigma_max
        keep &= view[:, 0].abs() < z * cam.tan_x + margin
        keep &= view[:, 1].abs() < z * cam.tan_y + margin
        keep &= scene.opacity >= MIN_ALPHA
        idx = keep.nonzero(as_tuple=True)[0]
        image = torch.zeros(3, cam.height, cam.width)
        depth = torch.zeros(cam.height, cam.width)
        if idx.numel() == 0:
            return image, depth

        v = view[idx]
        z = v[:, 2].clamp_min(ZNEAR)
        Rg = _quat_wxyz_to_matrix(scene.rot[idx])
        M = Rg * scene.scale[idx].unsqueeze(-1)  # rows scaled: M M^T = R S^2 R^T
        Mv = M @ R  # view-space Cholesky: cov_v = Mv Mv^T
        c00 = (Mv[:, 0] ** 2).sum(-1)
        c01 = (Mv[:, 0] * Mv[:, 1]).sum(-1)
        c02 = (Mv[:, 0] * Mv[:, 2]).sum(-1)
        c11 = (Mv[:, 1] ** 2).sum(-1)
        c12 = (Mv[:, 1] * Mv[:, 2]).sum(-1)
        c22 = (Mv[:, 2] ** 2).sum(-1)

        x, y = v[:, 0], v[:, 1]
        j00, j02 = fx / z, -fx * x / z**2
        j11, j12 = fy / z, -fy * y / z**2
        s00 = j00**2 * c00 + 2 * j00 * j02 * c02 + j02**2 * c22 + EPS_2D
        s01 = j00 * j11 * c01 + j00 * j12 * c02 + j02 * j11 * c12 + j02 * j12 * c22
        s11 = j11**2 * c11 + 2 * j11 * j12 * c12 + j12**2 * c22 + EPS_2D

        det = (s00 * s11 - s01**2).clamp_min(1e-12)
        conic00, conic01, conic11 = s11 / det, -s01 / det, s00 / det
        mid = 0.5 * (s00 + s11)
        disc = (mid**2 - det).clamp_min(0.0).sqrt()
        lam_max = mid + disc
        radius = 3.0 * lam_max.clamp_min(0.0).sqrt()

        valid = radius >= 0.5
        idx, z, radius = idx[valid], z[valid], radius[valid]
        if idx.numel() == 0:
            return image, depth
        conic00, conic01, conic11 = conic00[valid], conic01[valid], conic11[valid]

        mx = fx * x[valid] / z + cam.width / 2.0
        my = fy * y[valid] / z + cam.height / 2.0
        in_img = (
            (mx > -MAX_RADIUS_PX) & (mx < cam.width + MAX_RADIUS_PX)
            & (my > -MAX_RADIUS_PX) & (my < cam.height + MAX_RADIUS_PX)
        )
        idx, z, radius = idx[in_img], z[in_img], radius[in_img]
        if idx.numel() == 0:
            return image, depth
        mx, my = mx[in_img], my[in_img]
        conic00, conic01, conic11 = conic00[in_img], conic01[in_img], conic11[in_img]

        order = torch.argsort(z, stable=True)
        idx, z, radius, mx, my = idx[order], z[order], radius[order], mx[order], my[order]
        conic00, conic01, conic11 = conic00[order], conic01[order], conic11[order]

        px_all, alpha_all, col_all, rank_all, sample_all = self._splat(
            idx, z, radius, mx, my, conic00, conic01, conic11
        )
        if px_all is None:
            return image, depth
        return self._composite(px_all, alpha_all, col_all, rank_all, sample_all)

    def _splat(
        self,
        idx: Tensor,
        z: Tensor,
        radius: Tensor,
        mx: Tensor,
        my: Tensor,
        conic00: Tensor,
        conic01: Tensor,
        conic11: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Contributions of one view: (pixel, alpha, colour, depth rank, depth).

        ``idx``/``z``/``mx``/``my``/``conic*`` are the depth-sorted per-Gaussian
        arrays, so a contribution's ``gcol`` is both its index into them and its
        front-to-back rank; the sample depth is ``z[gcol]``.
        """
        cam = self.camera
        scene = self.scene
        px_chunks: list[Tensor] = []
        alpha_chunks: list[Tensor] = []
        col_chunks: list[Tensor] = []
        rank_chunks: list[Tensor] = []
        z_chunks: list[Tensor] = []
        r_int = radius.ceil().clamp(min=1.0, max=float(MAX_RADIUS_PX)).to(torch.int64)
        for bin_hi in RADIUS_BINS:
            lo = 0 if bin_hi == RADIUS_BINS[0] else RADIUS_BINS[RADIUS_BINS.index(bin_hi) - 1]
            sel = (r_int > lo) & (r_int <= bin_hi)
            if not bool(sel.any()):
                continue
            gidx = sel.nonzero(as_tuple=True)[0]
            offs = torch.arange(-bin_hi, bin_hi + 1)
            dy, dx = torch.meshgrid(offs, offs, indexing="ij")
            dy, dx = dy.reshape(-1), dx.reshape(-1)
            pxx = (mx[gidx].round().unsqueeze(1) + dx).reshape(-1)
            pyy = (my[gidx].round().unsqueeze(1) + dy).reshape(-1)
            ddx = (pxx - mx[gidx].unsqueeze(1).expand(-1, dx.numel()).reshape(-1))
            ddy = (pyy - my[gidx].unsqueeze(1).expand(-1, dy.numel()).reshape(-1))
            gcol = gidx.unsqueeze(1).expand(-1, dx.numel()).reshape(-1)
            in_img = (
                (pxx >= 0) & (pxx < cam.width) & (pyy >= 0) & (pyy < cam.height)
            )
            pxx, pyy, ddx, ddy, gcol = (
                pxx[in_img], pyy[in_img], ddx[in_img], ddy[in_img], gcol[in_img],
            )
            pxx = pxx.to(torch.int64)
            pyy = pyy.to(torch.int64)
            power = (
                0.5 * (conic00[gcol] * ddx**2 + conic11[gcol] * ddy**2)
                + conic01[gcol] * ddx * ddy
            )
            keep = (power < MAX_CONIC_POWER) & (power >= 0.0)
            pxx, pyy, gcol, power = pxx[keep], pyy[keep], gcol[keep], power[keep]
            if pxx.numel() == 0:
                continue
            alpha = (scene.opacity[idx[gcol]] * torch.exp(-power)).clamp(max=0.99)
            keep = alpha >= MIN_ALPHA
            pxx, pyy, gcol, alpha = pxx[keep], pyy[keep], gcol[keep], alpha[keep]
            if pxx.numel() == 0:
                continue
            px_chunks.append(pyy * cam.width + pxx)
            alpha_chunks.append(alpha)
            col_chunks.append(scene.color[idx[gcol]])
            # gcol indexes the depth-sorted arrays: it is the front-to-back rank
            # of the Gaussian this contribution belongs to
            rank_chunks.append(gcol)
            z_chunks.append(z[gcol])
        if not px_chunks:
            return None, None, None, None, None
        return (
            torch.cat(px_chunks),
            torch.cat(alpha_chunks),
            torch.cat(col_chunks),
            torch.cat(rank_chunks),
            torch.cat(z_chunks),
        )

    def _composite(
        self, px: Tensor, alpha: Tensor, color: Tensor, rank_key: Tensor,
        sample: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Composited view: (RGB [3, H, W], axial depth [H, W])."""
        cam = self.camera
        n_px = cam.width * cam.height
        # ties in px are broken by the Gaussian's depth rank, so the stable sort
        # below leaves every pixel's k-buffer in exact front-to-back order even
        # though the contributions arrive bin by bin (radius order)
        key = px * (int(rank_key.max()) + 1) + rank_key
        order = torch.argsort(key, stable=True)
        px, alpha, color, sample = px[order], alpha[order], color[order], sample[order]
        counts = torch.bincount(px, minlength=n_px)
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(px.numel()) - starts[px]
        keep = rank < K_BUFFER
        px, rank, alpha, color, sample = (
            px[keep], rank[keep], alpha[keep], color[keep], sample[keep],
        )

        alpha_buf = torch.zeros(n_px, K_BUFFER)
        color_buf = torch.zeros(n_px, K_BUFFER, 3)
        sample_buf = torch.zeros(n_px, K_BUFFER)
        alpha_buf[px, rank] = alpha
        color_buf[px, rank] = color
        sample_buf[px, rank] = sample
        trans = torch.ones(n_px)
        out = torch.zeros(n_px, 3)
        # depth: the transmittance-weighted mean of the composited samples, so a
        # pixel reads the surface that actually contributes most of its colour
        depth_sum = torch.zeros(n_px)
        depth_weight = torch.zeros(n_px)
        for k in range(K_BUFFER):
            a = alpha_buf[:, k]
            out += trans.unsqueeze(-1) * a.unsqueeze(-1) * color_buf[:, k]
            weight = trans * a
            depth_sum += weight * sample_buf[:, k]
            depth_weight += weight
            trans = trans * (1.0 - a)
        out = out.clamp(0.0, 1.0)
        depth = depth_sum / depth_weight.clamp_min(1e-6)  # 0 where nothing landed
        return (
            out.reshape(cam.height, cam.width, 3).permute(2, 0, 1).contiguous(),
            depth.reshape(cam.height, cam.width),
        )
