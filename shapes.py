"""Dotted shapes for Trace & Practise mode (coordinates are 0..1 of the canvas)."""
import math
import numpy as np
from PIL import Image, ImageDraw


def circle(cx, cy, r, n=48):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n + 1)]


def _star():
    pts = [(0.5 + 0.38 * math.cos(math.radians(-90 + 144 * k)),
            0.55 + 0.38 * math.sin(math.radians(-90 + 144 * k))) for k in range(6)]
    return pts


def _heart(n=60):
    out = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        out.append((0.5 + x / 40, 0.45 - y / 40))
    return out


SHAPES = {
    "circle": [circle(0.5, 0.5, 0.32)],
    "square": [[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]],
    "triangle": [[(0.5, 0.15), (0.85, 0.8), (0.15, 0.8), (0.5, 0.15)]],
    "star": [_star()],
    "heart": [_heart()],
    "house": [[(0.15, 0.52), (0.5, 0.18), (0.85, 0.52), (0.15, 0.52)],
              [(0.25, 0.52), (0.25, 0.82), (0.75, 0.82), (0.75, 0.52)],
              [(0.45, 0.82), (0.45, 0.62), (0.55, 0.62), (0.55, 0.82)]],
    "sun": [circle(0.5, 0.5, 0.17)] + [
        [(0.5 + 0.27 * math.cos(a), 0.5 + 0.27 * math.sin(a)),
         (0.5 + 0.40 * math.cos(a), 0.5 + 0.40 * math.sin(a))]
        for a in [k * math.pi / 4 for k in range(8)]],
    "fish": [[(0.45 + 0.28 * math.cos(2 * math.pi * i / 40), 0.5 + 0.17 * math.sin(2 * math.pi * i / 40))
              for i in range(41)],
             [(0.72, 0.5), (0.9, 0.32), (0.9, 0.68), (0.72, 0.5)]],
    "tree": [circle(0.5, 0.35, 0.22), [(0.44, 0.56), (0.44, 0.85)], [(0.56, 0.56), (0.56, 0.85)]],
    "wave": [[(0.1 + 0.8 * i / 60, 0.5 + 0.15 * math.sin(2 * math.pi * 2 * i / 60)) for i in range(61)]],
    "zigzag": [[(0.1, 0.65), (0.26, 0.35), (0.42, 0.65), (0.58, 0.35), (0.74, 0.65), (0.9, 0.35)]],
}


def trace_score(ink, strokes, size, tol=22):
    """ink: bool array (size x size). Returns 0..1 = how much of the dotted line was covered,
    reduced if the child scribbled far away from the line."""
    pts = []
    for st in strokes:
        for (x0, y0), (x1, y1) in zip(st, st[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) * size // 6))
            pts += [((x0 + (x1 - x0) * t / n) * size, (y0 + (y1 - y0) * t / n) * size) for t in range(n + 1)]
    hit = sum(1 for x, y in pts
              if ink[max(0, int(y) - 12):int(y) + 13, max(0, int(x) - 12):int(x) + 13].any())
    cover = hit / len(pts)
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    for st in strokes:
        d.line([(x * size, y * size) for x, y in st], fill=255, width=2 * tol)
    precision = (ink & (np.array(mask) > 0)).sum() / max(int(ink.sum()), 1)
    return cover * (0.1 + 0.9 * precision)