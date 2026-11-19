#!/usr/bin/env python3
"""Textless cover: a scattered ggml graph converging into one TPU chip.

Draws at 2x and downsamples. Writes devto-cover.<hash>.jpg (1376x578) and builder-cover.<hash>.jpg
(1200x675) beside this file, each named by the first 8 hex digits of its own SHA-256.
Palette from publishing-kit make-cover.py (surface, blue, orange).
"""

import hashlib
import io
import math
import pathlib
import random

from PIL import Image, ImageDraw

SURFACE = (26, 26, 25)
RULE = (58, 58, 55)
INK_3 = (128, 127, 120)
BLUE = (57, 135, 229)
ORANGE = (217, 89, 38)
HERE = pathlib.Path(__file__).resolve().parent


def draw(w, h):
    s = 2
    W, H = w * s, h * s
    im = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    rnd = random.Random(7)

    # the chip, right of centre
    cs = int(H * 0.50)
    cx, cy = int(W * 0.74), H // 2
    x0, y0 = cx - cs // 2, cy - cs // 2
    pin_n, pin_l, pin_w = 9, int(cs * 0.09), max(4, int(cs * 0.03))
    for i in range(pin_n):
        t = x0 + int(cs * (i + 1) / (pin_n + 1))
        for (ax, ay, bx, by) in (
            (t - pin_w // 2, y0 - pin_l, t + pin_w // 2, y0),
            (t - pin_w // 2, y0 + cs, t + pin_w // 2, y0 + cs + pin_l),
        ):
            d.rectangle((ax, ay, bx, by), fill=INK_3)
        u = y0 + int(cs * (i + 1) / (pin_n + 1))
        d.rectangle((x0 + cs, u - pin_w // 2, x0 + cs + pin_l, u + pin_w // 2), fill=INK_3)
        d.rectangle((x0 - pin_l, u - pin_w // 2, x0, u + pin_w // 2), fill=INK_3)
    d.rounded_rectangle((x0, y0, x0 + cs, y0 + cs), radius=int(cs * 0.06), fill=(40, 40, 38), outline=RULE, width=s * 3)
    # cores: a 4x4 grid, lit orange
    g, pad = 4, int(cs * 0.14)
    cell = (cs - 2 * pad) / g
    for i in range(g):
        for j in range(g):
            ax = x0 + pad + i * cell + cell * 0.12
            ay = y0 + pad + j * cell + cell * 0.12
            d.rounded_rectangle((ax, ay, ax + cell * 0.76, ay + cell * 0.76), radius=int(cell * 0.12), fill=ORANGE)

    # the graph: layered nodes on the left, edges between layers, then everything into the chip's left edge
    layers = [7, 9, 8, 6, 4]
    lx0, lx1 = int(W * 0.07), int(W * 0.48)
    pts = []
    for li, n in enumerate(layers):
        x = lx0 + (lx1 - lx0) * li / (len(layers) - 1)
        span = H * (0.78 - 0.1 * li)
        col = []
        for k in range(n):
            y = H / 2 - span / 2 + span * (k + 0.5) / n + rnd.uniform(-H * 0.02, H * 0.02)
            col.append((x + rnd.uniform(-W * 0.012, W * 0.012), y))
        pts.append(col)
    for a, b in zip(pts, pts[1:]):
        for p in a:
            for q in rnd.sample(b, min(2, len(b))):
                d.line((p, q), fill=RULE, width=s * 2)
    entry_x = x0 - pin_l
    for k, p in enumerate(pts[-1]):
        ty = y0 + int(cs * (2 * k + 2) / (pin_n + 1))
        # a gentle curve into the chip
        steps = 24
        prev = p
        for i in range(1, steps + 1):
            t = i / steps
            x = p[0] + (entry_x - p[0]) * t
            y = p[1] + (ty - p[1]) * (1 - math.cos(math.pi * t)) / 2
            d.line((prev, (x, y)), fill=BLUE, width=s * 3)
            prev = (x, y)
    r = int(H * 0.022)
    for col in pts:
        for (x, y) in col:
            d.ellipse((x - r, y - r, x + r, y + r), fill=BLUE)
    return im.resize((w, h), Image.LANCZOS)


def save(im, stem):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90, optimize=True)
    data = buf.getvalue()
    name = f"{stem}.{hashlib.sha256(data).hexdigest()[:8]}.jpg"
    (HERE / name).write_bytes(data)
    print(f"wrote {name}  {im.width}x{im.height}  {len(data) // 1024} KB")


if __name__ == "__main__":
    save(draw(1376, 578), "devto-cover")
    save(draw(1200, 675), "builder-cover")
