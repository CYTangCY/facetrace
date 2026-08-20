"""run_ui.py — M0-T03:即時曲線視窗。

上半鏡頭畫面 + face mesh,下半四條捲動曲線(三態渲染:正常色 / 灰 / 斷開)。
取像與推論跑在背景執行緒,UI 執行緒只負責畫。執行期零網路。

用法:
    python scripts/run_ui.py
    python scripts/run_ui.py --no-mic
    python scripts/run_ui.py --seconds 15        # 曲線視窗顯示最近 15 秒

關閉視窗或按 q / Esc 離開。離開碼:0 正常;1 鏡頭或模型不可用。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from facetrace.capture import CaptureError, FrameSource  # noqa: E402
from facetrace.signals import MicLevel, SignalExtractor  # noqa: E402
from facetrace.ui import (  # noqa: E402
    WINDOW_SECONDS,
    CaptureWorker,
    MainWindow,
)

from PySide6.QtCore import QThread, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _report(label: str, times: list[float]) -> None:
    if not times:
        print(f"  {label:<22s} (no samples)")
        return
    arr = np.asarray(times[5:]) if len(times) > 10 else np.asarray(times)
    print(f"  {label:<22s} mean {arr.mean() * 1000:6.2f} ms   median {np.median(arr) * 1000:6.2f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="FaceTrace M0-T03 live plot window")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-mic", action="store_true", help="skip microphone entirely")
    parser.add_argument("--seconds", type=float, default=WINDOW_SECONDS,
                        help="scrolling window width in seconds")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="auto-close after N seconds (0 = run until quit); for timing runs")
    args = parser.parse_args()

    # 鏡頭先開:失敗就直接退出,連 Qt 都不用起來。
    try:
        source = FrameSource(
            camera_index=args.camera, model_path=args.model,
            width=args.width, height=args.height,
        )
    except CaptureError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    mic = None
    if not args.no_mic:
        mic = MicLevel()
        if mic.available:
            print(f"[mic]    {mic.device_name}")
    else:
        print("[mic]    disabled (--no-mic); mic_volume absent for this run")
    print(f"[camera] index {args.camera} @ {args.width}x{args.height}")
    print("[keys]   q / Esc = quit (or just close the window)")

    extractor = SignalExtractor(mic=mic)
    app = QApplication(sys.argv)
    window = MainWindow(window_seconds=args.seconds)

    thread = QThread()
    worker = CaptureWorker(source, extractor)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # 跨執行緒 signal 預設 queued connection -> on_frame 在主執行緒執行
    worker.frameReady.connect(window.on_frame)

    def on_failed(message: str) -> None:
        print(f"[FAIL] {message}", file=sys.stderr)
        window.close()

    worker.failed.connect(on_failed)
    window.closing.connect(worker.stop)

    started = time.perf_counter()
    thread.start()
    window.show()
    if args.duration > 0:
        QTimer.singleShot(int(args.duration * 1000), window.close)
    exit_code = app.exec()

    # --- 收工:先停 worker,等執行緒真的結束,再釋放硬體 ---
    worker.stop()
    thread.quit()
    if not thread.wait(3000):
        print("[warn] capture thread did not stop within 3s; terminating", file=sys.stderr)
        thread.terminate()
        thread.wait()
    source.release()
    if mic is not None:
        mic.close()

    n = len(worker.t_capture)
    # 從第一幀到最後一幀,排除啟動與收工時間,才是真正的取像速率
    if worker.t_first is not None and worker.t_last and n > 1:
        elapsed, n_span = worker.t_last - worker.t_first, n - 1
    else:
        elapsed, n_span = time.perf_counter() - started, n
    print("\n[timing] per-frame cost (first 5 frames dropped as warm-up)")
    _report("capture+inference *", worker.t_capture)
    _report("signal extract", worker.t_extract)
    _report("UI video + mesh", window.t_video)
    _report("UI redraw (curves)", window.t_render)
    print("  * capture+inference is dominated by cap.read() blocking until the")
    print("    camera delivers the next frame -- it is wait, not work.")
    if n:
        def _m(xs: list[float]) -> float:
            return float(np.mean(xs[5:])) if len(xs) > 10 else (float(np.mean(xs)) if xs else 0.0)

        work = _m(worker.t_extract) + _m(window.t_video) + _m(window.t_render)
        print(f"\n  frames processed:      {n} frames, {n_span} intervals in {elapsed:.2f} s -> {n_span / elapsed:.1f} FPS")
        print(f"  frames with a face:    {worker.n_face}   without a face: {worker.n_noface}")
        print(f"  our work per frame:    {work * 1000:.2f} ms "
              f"({work / (1 / 30) * 100:.0f}% of the 33.3 ms budget at 30 FPS)")
    print("[done] camera, microphone and landmarker released")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
