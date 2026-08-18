"""capture — 取像與 Face Landmarker 推論。

職責:
- 用 OpenCV 開本機鏡頭,逐幀取像(BGR ndarray)。
- 用 MediaPipe Face Landmarker(models/face_landmarker.task,執行期零網路)
  以 VIDEO 模式對每一幀推論(單調遞增 timestamp_ms),取得:
    * face landmarks(478 點,含虹膜)
    * blendshapes(52 個 MediaPipe 原生係數,名稱照原樣傳遞)
    * facial transformation matrix(頭部姿態,4x4)
- 每一幀附上時間戳與「是否偵測到臉」旗標;沒臉時回傳空結果,
  不內插、不硬猜、不沿用上一幀的值。

介面:
- open_camera(index=0, ...) -> cv2.VideoCapture
- load_landmarker(model_path=None) -> vision.FaceLandmarker(VIDEO 模式)
- FrameSource(...) -> 綁定鏡頭 + landmarker + 單調時間戳的取像來源
- FrameSource.next_frame() -> FrameResult
- FrameSource.release()

紅線:不輸出情緒標籤;不呼叫網路;不引入 MediaPipe Face Landmarker 以外的模型。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from mediapipe import Image as MpImage
from mediapipe import ImageFormat as MpImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# repo root = .../src/facetrace/capture.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class CaptureError(RuntimeError):
    """鏡頭或模型無法使用時丟出,由呼叫端印訊息並非零退出。"""


@dataclass(frozen=True)
class FrameResult:
    """單一幀的推論結果。

    沒偵測到臉時:has_face=False,且 blendshapes / landmarks / head_pose 全為
    None——不補值、不沿用上一幀。下游看到 has_face=False 就該標成 invalid。
    """

    timestamp_ms: int
    frame_bgr: np.ndarray
    has_face: bool
    blendshapes: Optional[dict[str, float]] = None
    landmarks: Optional[list] = None          # list[NormalizedLandmark]
    head_pose: Optional[np.ndarray] = None    # 4x4 facial transformation matrix


def open_camera(
    index: int = 0,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    backend: Optional[int] = None,
) -> cv2.VideoCapture:
    """開鏡頭。失敗丟 CaptureError。

    Windows 上預設用 CAP_DSHOW:實測開啟速度比 MSMF 快,吞吐相同(約 30 FPS,
    受硬體上限決定)。
    """
    if backend is None:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise CaptureError(
            f"could not open camera index {index}. "
            "Check that a webcam is connected and not in use by another app."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, _ = camera.read()
    if not ok:
        camera.release()
        raise CaptureError(
            f"camera index {index} opened but returned no frame. "
            "Another app may be holding it, or the driver refused this resolution."
        )
    return camera


def load_landmarker(
    model_path: Optional[Path | str] = None,
    num_faces: int = 1,
) -> mp_vision.FaceLandmarker:
    """載入 Face Landmarker(VIDEO 模式,開 blendshapes 與 transformation matrix)。

    完全從本機檔案載入,不觸網路。
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise CaptureError(
            f"model not found: {path}\n"
            "Expected models/face_landmarker.task in the repo. "
            "If missing, run scripts/download_model.py once (needs network)."
        )
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=num_faces,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        return mp_vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:  # noqa: BLE001 - surface any init failure to the caller
        raise CaptureError(f"could not create FaceLandmarker: {exc}") from exc


class FrameSource:
    """鏡頭 + landmarker + 單調遞增時間戳,逐幀產出 FrameResult。

    MediaPipe VIDEO 模式要求 timestamp 嚴格遞增,所以時間戳由本類別統一發放
    (以 perf_counter 為基準,並強制嚴格遞增),呼叫端不必自己管。
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_path: Optional[Path | str] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        mirror: bool = True,
    ) -> None:
        self.mirror = mirror
        self._camera = open_camera(camera_index, width, height)
        try:
            self._landmarker = load_landmarker(model_path)
        except CaptureError:
            self._camera.release()
            raise
        self._t0 = time.perf_counter()
        self._last_ts_ms = -1
        self._closed = False

    # -- internals ---------------------------------------------------------
    def _next_timestamp_ms(self) -> int:
        ts = int((time.perf_counter() - self._t0) * 1000.0)
        if ts <= self._last_ts_ms:      # VIDEO mode needs strictly increasing stamps
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    # -- public API --------------------------------------------------------
    def next_frame(self) -> FrameResult:
        """抓一幀並推論。抓不到幀丟 CaptureError;偵測不到臉回 has_face=False。"""
        if self._closed:
            raise CaptureError("FrameSource is already released")

        ok, frame_bgr = self._camera.read()
        if not ok or frame_bgr is None:
            raise CaptureError("camera returned no frame (disconnected?)")
        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)  # 鏡像,讓使用者看自己像照鏡子

        timestamp_ms = self._next_timestamp_ms()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            MpImage(image_format=MpImageFormat.SRGB, data=rgb), timestamp_ms
        )

        if not result.face_landmarks:
            # 沒臉:回空結果。不補值、不沿用上一幀。
            return FrameResult(
                timestamp_ms=timestamp_ms, frame_bgr=frame_bgr, has_face=False
            )

        blendshapes = None
        if result.face_blendshapes:
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}

        head_pose = None
        if result.facial_transformation_matrixes:
            head_pose = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)

        return FrameResult(
            timestamp_ms=timestamp_ms,
            frame_bgr=frame_bgr,
            has_face=True,
            blendshapes=blendshapes,
            landmarks=result.face_landmarks[0],
            head_pose=head_pose,
        )

    def release(self) -> None:
        """釋放鏡頭與 landmarker。重複呼叫安全。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._camera.release()
        finally:
            self._landmarker.close()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
