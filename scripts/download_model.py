"""One-time, install-time download of the MediaPipe Face Landmarker model.

This is the ONLY place in the repo that touches the network. Run once after
cloning (or not at all — the model file is committed to the repo). Nothing at
runtime (verify_env.py, src/facetrace/*) makes any network call.

Usage:
    python scripts/download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"


def main() -> int:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"[skip] model already exists: {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
        return 0

    print(f"[download] {MODEL_URL}")
    print(f"[to]       {MODEL_PATH}")
    tmp_path = MODEL_PATH.with_suffix(".task.part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, open(tmp_path, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        tmp_path.replace(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - report anything and fail loudly
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"[error] download failed: {exc}", file=sys.stderr)
        return 1

    size = MODEL_PATH.stat().st_size
    print(f"[ok] saved {size} bytes ({size / 1024 / 1024:.2f} MiB) -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
