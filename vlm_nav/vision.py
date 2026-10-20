"""Image tiling for scan/comparison calls (goal.md optimization).

A scan asks one question - "which heading should the drone face?" - and does
not need one full-resolution frame per heading. Tiling the views into a single
numbered contact sheet sends one image, keeps the per-view ordering, and cuts
the visual-token count roughly by the number of tiles.

Identity/confirmation paths keep using single frames or crops; a contact sheet
is never used for a final identity judgement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


@dataclass(frozen=True)
class ContactSheet:
    image: Image.Image
    columns: int
    rows: int
    tile_width: int
    tile_height: int
    count: int
    padding: int = 2
    label_height: int = 16

    def tile_box(self, index: int) -> tuple[int, int, int, int]:
        """Pixel box of one tile's image (without the label strip)."""
        if not 0 <= index < self.count:
            raise IndexError(f"tile {index} outside 0..{self.count - 1}")
        row, column = divmod(index, self.columns)
        left = self.padding + column * (self.tile_width + self.padding)
        top = (
            self.padding
            + row * (self.tile_height + self.label_height + self.padding)
            + self.label_height
        )
        return (left, top, left + self.tile_width, top + self.tile_height)

    def map_box(
        self, index: int, box: Sequence[float]
    ) -> list[float]:
        """Map a normalized sheet-coordinate box onto one tile (0-1000)."""
        left, top, right, bottom = self.tile_box(index)
        width = max(1.0, float(right - left))
        height = max(1.0, float(bottom - top))
        sheet_w, sheet_h = self.image.size
        x1 = float(box[0]) / 1000.0 * sheet_w
        y1 = float(box[1]) / 1000.0 * sheet_h
        x2 = float(box[2]) / 1000.0 * sheet_w
        y2 = float(box[3]) / 1000.0 * sheet_h
        mapped = [
            (x1 - left) / width * 1000.0,
            (y1 - top) / height * 1000.0,
            (x2 - left) / width * 1000.0,
            (y2 - top) / height * 1000.0,
        ]
        return [
            round(min(1000.0, max(0.0, value)), 3) for value in mapped
        ]

    def tensor(self) -> torch.Tensor:
        array = np.asarray(self.image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if hasattr(frame, "detach"):
        array = frame.detach().clamp(0, 1).cpu().numpy()
    else:
        array = np.asarray(frame)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.dtype != np.uint8:
        array = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=14)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(
    frames: Sequence[Any],
    *,
    columns: int | None = None,
    padding: int = 2,
    label_height: int = 16,
    label: bool = True,
    start_index: int = 0,
    tile_size: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> ContactSheet:
    """Tile ``frames`` into one numbered sheet.

    ``start_index=0`` matches the scan view-index convention; the comparison
    call uses ``start_index=1`` because its prompt has always numbered views
    from 1. ``scale`` shrinks each tile before tiling (0.5 halves both sides,
    cutting the sheet's pixel area by 4 while keeping one image/token budget
    for the whole scan).
    """
    images = [_to_pil(frame) for frame in frames]
    if not images:
        raise ValueError("contact sheet needs at least one frame")
    if tile_size is None:
        base_w, base_h = images[0].size
        factor = max(0.05, min(1.0, float(scale)))
        tile_size = (
            max(1, int(round(base_w * factor))),
            max(1, int(round(base_h * factor))),
        )
    width, height = max(1, int(tile_size[0])), max(1, int(tile_size[1]))
    count = len(images)
    if columns is None:
        columns = 4 if count <= 8 else 5
    columns = max(1, min(int(columns), count))
    rows = int(math.ceil(count / columns))
    label_h = max(0, int(label_height)) if label else 0
    pad = max(0, int(padding))
    sheet = Image.new(
        "RGB",
        (
            columns * width + (columns + 1) * pad,
            rows * (height + label_h) + (rows + 1) * pad,
        ),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)
    font = _label_font() if label else None
    for index, image in enumerate(images):
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        row, column = divmod(index, columns)
        left = pad + column * (width + pad)
        top = pad + row * (height + label_h + pad)
        if label:
            draw.rectangle(
                (left, top, left + width - 1, top + label_h - 1),
                fill=(0, 0, 0),
            )
            draw.text(
                (left + 4, top + 1),
                str(index + start_index),
                fill=(255, 255, 255),
                font=font,
            )
        sheet.paste(image, (left, top + label_h))
    return ContactSheet(
        image=sheet,
        columns=columns,
        rows=rows,
        tile_width=width,
        tile_height=height,
        count=count,
        padding=pad,
        label_height=label_h,
    )


def contact_sheet_tensor(frames: Sequence[Any], **kwargs: Any) -> torch.Tensor:
    return contact_sheet(frames, **kwargs).tensor()
