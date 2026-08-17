"""verify_env.py — offline environment check for FaceTrace (M0-T01).

What it does (no network, ever):
  1. Loads models/face_landmarker.task with MediaPipe Face Landmarker
     (blendshapes + facial transformation matrix enabled).
  2. Runs detection on scripts/test_face.jpg.
  3. Prints: number of faces detected, top-10 blendshapes by score
     (MediaPipe native names, e.g. mouthSmileLeft), and an FPS estimate from
     20 back-to-back inferences.
  4. Exits non-zero with a clear message if the model or test image is
     missing, or if no face is detected.

A socket guard is installed before importing MediaPipe so that any accidental
network attempt inside this process raises immediately — the run log therefore
proves "zero network at runtime" rather than merely claiming it.

Usage (from anywhere; paths are resolved relative to the repo root):
    python scripts/verify_env.py
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Hard-disable outbound network for this process (before any heavy import).
# ---------------------------------------------------------------------------
class _NetworkDisabled(RuntimeError):
    pass


def _blocked(*_args, **_kwargs):
    raise _NetworkDisabled("verify_env.py: outbound network is disabled by design")


socket.socket.connect = _blocked          # type: ignore[assignment]
socket.socket.connect_ex = _blocked       # type: ignore[assignment]
socket.create_connection = _blocked       # type: ignore[assignment]
socket.getaddrinfo = _blocked             # type: ignore[assignment]

# Quieter native logs (cosmetic; harmless if ignored by the installed build).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"
IMAGE_PATH = REPO_ROOT / "scripts" / "test_face.jpg"
TIMED_RUNS = 20


def fail(msg: str, code: int = 1) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return code


def main() -> int:
    # Line-buffer stdout so [..] lines interleave sanely with stderr when piped.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    print("[guard] outbound network sockets disabled for this process")

    # ---- 1. imports (report versions; any ImportError is a hard failure) ----
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as exc:
        return fail(f"import error: {exc}. Run: pip install -r requirements.txt")

    print(f"[env]   python     {sys.version.split()[0]}  ({sys.executable})")
    print(f"[env]   mediapipe  {mp.__version__}")
    print(f"[env]   opencv     {cv2.__version__}")
    print(f"[env]   numpy      {np.__version__}")

    # ---- 2. model file ------------------------------------------------------
    if not MODEL_PATH.is_file():
        return fail(
            f"model not found: {MODEL_PATH}\n"
            "       expected models/face_landmarker.task in the repo. "
            "If it is missing, run scripts/download_model.py once (needs network)."
        )
    print(f"[model] {MODEL_PATH}  ({MODEL_PATH.stat().st_size} bytes)")

    # ---- 3. test image ------------------------------------------------------
    if not IMAGE_PATH.is_file():
        return fail(f"test image not found: {IMAGE_PATH} (run scripts/make_test_face.py)")
    bgr = cv2.imread(str(IMAGE_PATH))
    if bgr is None:
        return fail(f"could not decode test image: {IMAGE_PATH}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[image] {IMAGE_PATH}  ({bgr.shape[1]}x{bgr.shape[0]})")

    # ---- 4. build landmarker -------------------------------------------------
    try:
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:  # noqa: BLE001
        return fail(f"could not create FaceLandmarker: {exc}")

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ---- 5. single detection (also serves as warm-up) ------------------------
    try:
        result = landmarker.detect(mp_image)
    except Exception as exc:  # noqa: BLE001
        return fail(f"detection raised: {exc}")

    n_faces = len(result.face_landmarks)
    print(f"[detect] faces detected: {n_faces}")
    if n_faces == 0:
        return fail("no face detected in test image - environment NOT verified")

    if not result.face_blendshapes:
        return fail("face found but no blendshapes returned (output_face_blendshapes should be True)")

    n_landmarks = len(result.face_landmarks[0])
    n_blend = len(result.face_blendshapes[0])
    has_matrix = bool(result.facial_transformation_matrixes)
    print(f"[detect] landmarks per face: {n_landmarks}   blendshapes: {n_blend}   "
          f"transformation matrix: {'yes' if has_matrix else 'no'}")

    print("[blendshapes] top 10 by score:")
    top10 = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)[:10]
    for rank, cat in enumerate(top10, 1):
        print(f"  {rank:2d}. {cat.category_name:<24s} {cat.score:.4f}")

    # ---- 6. FPS estimate over TIMED_RUNS back-to-back inferences ------------
    t0 = time.perf_counter()
    for _ in range(TIMED_RUNS):
        landmarker.detect(mp_image)
    elapsed = time.perf_counter() - t0
    fps = TIMED_RUNS / elapsed if elapsed > 0 else float("inf")
    print(f"[fps] {TIMED_RUNS} inferences in {elapsed * 1000:.1f} ms  "
          f"-> avg {elapsed / TIMED_RUNS * 1000:.2f} ms/frame  ~ {fps:.1f} FPS "
          f"(single image, CPU, IMAGE mode)")

    landmarker.close()
    print("[OK] environment verified offline: model loaded, face detected, blendshapes produced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
