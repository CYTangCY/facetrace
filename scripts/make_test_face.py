"""Generate scripts/test_face.jpg — a synthetic face drawn with OpenCV only.

Used as the built-in test image for scripts/verify_env.py. No network, no
third-party model; just cv2 + numpy primitives. Deterministic output.

Usage:
    python scripts/make_test_face.py [--out scripts/test_face.jpg]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "scripts" / "test_face.jpg"

W, H = 640, 800


def _shade_ellipse(img: np.ndarray, center: tuple[int, int], axes: tuple[int, int],
                   color: tuple[int, int, int], dark_scale: float = 0.55) -> None:
    """Fill an ellipse with a soft radial gradient (lighter centre, darker rim)."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dx = (xx - center[0]) / axes[0]
    dy = (yy - center[1]) / axes[1]
    r = np.sqrt(dx * dx + dy * dy).clip(0, 1)
    scale = 1.0 - (1.0 - dark_scale) * (r ** 2)
    base = np.array(color, np.float32)
    layer = (scale[..., None] * base).clip(0, 255).astype(np.uint8)
    m = mask > 0
    img[m] = layer[m]


def draw_face() -> np.ndarray:
    img = np.full((H, W, 3), (70, 75, 80), np.uint8)  # plain dark background (BGR)

    # --- neck & shoulders (gives the detector body context) ---
    cv2.rectangle(img, (240, 560), (400, 700), (120, 160, 205), -1)
    cv2.ellipse(img, (320, 760), (260, 120), 0, 180, 360, (60, 90, 140), -1)

    # --- head with hair mass behind ---
    cv2.ellipse(img, (320, 330), (215, 285), 0, 0, 360, (30, 30, 40), -1)   # hair
    _shade_ellipse(img, (320, 360), (190, 250), (150, 185, 225), 0.7)       # skin

    # --- hairline (cover top of head) ---
    cv2.ellipse(img, (320, 200), (200, 110), 0, 180, 360, (30, 30, 40), -1)

    # --- ears ---
    for cx in (130, 510):
        _shade_ellipse(img, (cx, 370), (24, 42), (140, 170, 210), 0.7)

    # --- eyebrows (thick, slightly arched) ---
    for cx, sgn in ((240, -1), (400, 1)):
        pts = np.array([
            (cx - 55 * sgn, 268), (cx - 20 * sgn, 252), (cx + 25 * sgn, 250),
            (cx + 55 * sgn, 262), (cx + 55 * sgn, 274), (cx + 25 * sgn, 266),
            (cx - 20 * sgn, 268), (cx - 55 * sgn, 280),
        ], np.int32)
        cv2.fillPoly(img, [pts], (35, 30, 30))

    # --- eyes: white, iris, pupil, highlight, upper lid line ---
    for cx in (240, 400):
        cv2.ellipse(img, (cx, 320), (42, 22), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(img, (cx, 322), 18, (90, 60, 30), -1)      # iris (brown)
        cv2.circle(img, (cx, 322), 9, (10, 10, 10), -1)       # pupil
        cv2.circle(img, (cx - 6, 316), 4, (255, 255, 255), -1)  # catch light
        cv2.ellipse(img, (cx, 320), (42, 22), 0, 180, 360, (60, 40, 30), 3)  # upper lid
        cv2.ellipse(img, (cx, 320), (42, 22), 0, 0, 180, (110, 90, 80), 1)   # lower lid

    # --- nose: bridge shading + nostrils ---
    nose = np.array([(320, 330), (295, 430), (345, 430)], np.int32)
    cv2.fillPoly(img, [nose], (130, 165, 205))
    cv2.ellipse(img, (320, 432), (30, 16), 0, 0, 180, (120, 150, 190), -1)
    cv2.ellipse(img, (302, 436), (9, 5), 0, 0, 360, (70, 80, 110), -1)
    cv2.ellipse(img, (338, 436), (9, 5), 0, 0, 360, (70, 80, 110), -1)

    # --- mouth: lips with a dark centre line ---
    cv2.ellipse(img, (320, 505), (62, 20), 0, 0, 180, (95, 95, 190), -1)   # lower lip
    cv2.ellipse(img, (320, 500), (62, 12), 0, 180, 360, (80, 80, 170), -1)  # upper lip
    cv2.ellipse(img, (320, 500), (62, 6), 0, 0, 180, (40, 30, 70), 3)      # mouth line

    # --- chin / cheek shading ---
    cv2.ellipse(img, (320, 560), (110, 40), 0, 0, 180, (135, 165, 205), -1)

    # soften everything a bit so edges look photographic rather than vector
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    img = draw_face()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[{'ok' if ok else 'fail'}] wrote {args.out} ({img.shape[1]}x{img.shape[0]})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
