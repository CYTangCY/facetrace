"""run_capture.py — M0-T02 demo:鏡頭 + face mesh 疊圖 + 四條訊號終端輸出。

顯示鏡頭畫面與 face mesh,終端每 0.2 秒印一次四條訊號值、valid/信心、即時 FPS。
按 q 離開。執行期零網路。

用法:
    python scripts/run_capture.py
    python scripts/run_capture.py --no-mic          # 不開麥克風
    python scripts/run_capture.py --mesh-every 2    # 隔幀畫 mesh(省疊圖成本)
    python scripts/run_capture.py --duration 10 --no-window   # 無視窗計時量測

離開碼:0 正常結束;1 鏡頭或模型不可用。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

# 讓腳本可直接執行(不必先 pip install -e .)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from facetrace.capture import CaptureError, FrameSource  # noqa: E402
from facetrace.signals import SIGNAL_NAMES, MicLevel, SignalExtractor  # noqa: E402

from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

PRINT_INTERVAL_S = 0.2
FPS_WINDOW = 30
WINDOW_NAME = "FaceTrace M0-T02 - capture + signals (press q to quit)"

_draw = mp_vision.drawing_utils
_styles = mp_vision.drawing_styles
_conn = mp_vision.FaceLandmarksConnections


def draw_mesh(frame_bgr, landmarks, full: bool = True) -> None:
    """把 face mesh 畫在畫面上(MediaPipe 1.0.1 內建繪圖工具)。

    full=False 時只畫輪廓與虹膜,省掉最貴的 tesselation。
    """
    if full:
        _draw.draw_landmarks(
            image=frame_bgr,
            landmark_list=landmarks,
            connections=_conn.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=_styles.get_default_face_mesh_tesselation_style(),
        )
    _draw.draw_landmarks(
        image=frame_bgr,
        landmark_list=landmarks,
        connections=_conn.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=_styles.get_default_face_mesh_contours_style(),
    )
    _draw.draw_landmarks(
        image=frame_bgr,
        landmark_list=landmarks,
        connections=_conn.FACE_LANDMARKS_LEFT_IRIS + _conn.FACE_LANDMARKS_RIGHT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=_styles.get_default_face_mesh_iris_connections_style(),
    )


def format_line(sample, fps: float) -> str:
    """一行文字:四條訊號 + 信心 + FPS。invalid 的欄位印 ----(不印假值)。"""
    parts = []
    for name in SIGNAL_NAMES:
        value = sample.values.get(name)
        if value is None or not sample.valid.get(name, False):
            parts.append(f"{name}=  ----")
        else:
            parts.append(f"{name}={value:6.3f}")
    face = "face" if sample.has_face else "NOFACE"
    return (
        f"[{sample.timestamp_ms / 1000.0:6.2f}s] "
        + "  ".join(parts)
        + f"  | {face} conf={sample.confidence:4.2f}  fps={fps:5.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="FaceTrace M0-T02 capture demo")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--model", type=Path, default=None, help="path to face_landmarker.task")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-mic", action="store_true", help="skip microphone entirely")
    parser.add_argument("--no-window", action="store_true", help="no imshow (timing runs)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="auto-stop after N seconds (0 = until q)")
    parser.add_argument("--mesh-every", type=int, default=1,
                        help="draw mesh every N frames (2 = every other frame)")
    parser.add_argument("--light-mesh", action="store_true",
                        help="skip tesselation, draw contours+irises only")
    args = parser.parse_args()

    mic = None
    source = None
    try:
        try:
            source = FrameSource(
                camera_index=args.camera,
                model_path=args.model,
                width=args.width,
                height=args.height,
            )
        except CaptureError as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1

        if not args.no_mic:
            mic = MicLevel()
            if mic.available:
                print(f"[mic]    {mic.device_name}")
        else:
            print("[mic]    disabled (--no-mic); mic_volume absent for this run")

        extractor = SignalExtractor(mic=mic)
        print(f"[camera] index {args.camera} @ {args.width}x{args.height}")
        print("[keys]   q = quit")
        print("-" * 100)

        frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
        last_print = 0.0
        started = time.perf_counter()
        prev_t = started
        frame_count = 0

        while True:
            try:
                frame = source.next_frame()
            except CaptureError as exc:
                print(f"[FAIL] {exc}", file=sys.stderr)
                return 1

            sample = extractor.extract(frame)
            frame_count += 1

            now = time.perf_counter()
            frame_times.append(now - prev_t)
            prev_t = now
            fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0.0

            if not args.no_window:
                if frame.has_face and frame.landmarks and frame_count % args.mesh_every == 0:
                    draw_mesh(frame.frame_bgr, frame.landmarks, full=not args.light_mesh)
                if not frame.has_face:
                    cv2.putText(frame.frame_bgr, "NO FACE - signals invalid", (12, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame.frame_bgr, f"{fps:.1f} FPS  conf {sample.confidence:.2f}",
                            (12, frame.frame_bgr.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, frame.frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if now - last_print >= PRINT_INTERVAL_S:
                print(format_line(sample, fps), flush=True)
                last_print = now

            if args.duration and (now - started) >= args.duration:
                break

        elapsed = time.perf_counter() - started
        print("-" * 100)
        print(f"[done] {frame_count} frames in {elapsed:.2f} s -> average {frame_count / elapsed:.1f} FPS")
        return 0

    except KeyboardInterrupt:
        print("\n[done] interrupted")
        return 0
    finally:
        if source is not None:
            source.release()
        if mic is not None:
            mic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
