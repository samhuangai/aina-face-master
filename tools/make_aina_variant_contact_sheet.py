#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

VIEWS = ("FRONT", "THREEQ", "PROFILE")


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = sorted({path.name.rsplit("_", 1)[0] for path in args.preview.glob("*_FRONT.png")})
    if not labels:
        raise RuntimeError(f"No variant FRONT images in {args.preview}")

    cell = 540
    title_h = 70
    row_label_w = 250
    canvas = Image.new(
        "RGB",
        (row_label_w + cell * len(VIEWS), title_h + cell * len(labels)),
        (226, 229, 235),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(30)
    label_font = font(24)

    draw.rectangle((0, 0, canvas.width, title_h), fill=(35, 40, 52))
    draw.text((20, 18), "AINA MAKEHUMAN IDENTITY VARIANT REVIEW", fill=(245, 247, 252), font=title_font)
    for column, view in enumerate(VIEWS):
        x = row_label_w + column * cell + 18
        draw.text((x, 20), view, fill=(186, 210, 255), font=label_font)

    for row, label in enumerate(labels):
        y0 = title_h + row * cell
        draw.rectangle((0, y0, row_label_w, y0 + cell), fill=(49, 55, 70) if row % 2 == 0 else (57, 64, 81))
        wrapped = label.replace("_", "\n")
        draw.multiline_text((18, y0 + 28), wrapped, fill=(245, 247, 252), font=label_font, spacing=8)
        for column, view in enumerate(VIEWS):
            source = args.preview / f"{label}_{view}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            image = Image.open(source).convert("RGB")
            fitted = ImageOps.contain(image, (cell - 16, cell - 16))
            x0 = row_label_w + column * cell + (cell - fitted.width) // 2
            y = y0 + (cell - fitted.height) // 2
            canvas.paste(fitted, (x0, y))
            draw.rectangle((row_label_w + column * cell, y0, row_label_w + (column + 1) * cell, y0 + cell), outline=(160, 165, 178), width=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92, subsampling=0)
    print(args.output)


if __name__ == "__main__":
    main()
