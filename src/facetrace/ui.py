"""ui — pyqtgraph 曲線視窗(PySide6 原生視窗,不做網頁)。

畫面組成:
- 上半:鏡頭畫面 + face mesh 疊圖(BGR ndarray → QImage)。
- 下半:四個獨立子圖(brow_down、mouth_smile、head_stability、mic_volume),
  共用時間軸,捲動視窗預設顯示最近 WINDOW_SECONDS 秒。
- 狀態列:即時 FPS、追蹤信心值、NO FACE 指示。

三態渲染(本模組的重點,對應紅線三):
- SignalState.OK        → 各訊號自己的顏色,實線
- SignalState.UNCERTAIN → 灰線(有量測但信心低;線是連的,不是斷的)
- SignalState.ABSENT    → NaN + connect='finite' → 線斷開,絕不跨洞連線

取像不在 UI 執行緒:CaptureWorker 是 QObject,moveToThread 後在背景跑
FrameSource.next_frame() + SignalExtractor.extract(),用 Qt signal 把結果
送回主執行緒繪圖(跨執行緒 signal 預設是 queued connection)。
UI 執行緒不呼叫 OpenCV 取像,背景執行緒不碰任何 widget。

介面:
- MainWindow()、push_frame()、push_sample()、run_app()
- show_diary() 是 M0-T04 的事,本任務只留簽名。

紅線:UI 文字與圖例不得出現情緒詞;不呼叫網路;沒有量測就斷線,不內插補洞。
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional

# 必須在 import pyqtgraph 之前決定綁定。環境裡只有 PySide6,pyqtgraph 本來就會
# 自己選對;這行是防呆——萬一日後有人把 PyQt5 裝進同一個 venv,選擇仍然確定。
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot  # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from .signals import SIGNAL_NAMES, SignalSample, SignalState  # noqa: E402

# --- 外觀常數(可調)---------------------------------------------------------
WINDOW_SECONDS = 30.0     # 捲動視窗顯示最近幾秒
MAX_SAMPLES = 2400        # 約 80 秒 @30 FPS 的緩衝上限
FPS_WINDOW = 30           # 即時 FPS 的滾動平均幀數
VIDEO_WIDTH = 640         # 影像面板寬度

SIGNAL_COLORS: dict[str, str] = {
    "brow_down": "#4FC3F7",
    "mouth_smile": "#FFB74D",
    "head_stability": "#81C784",
    "mic_volume": "#BA68C8",
}
UNCERTAIN_COLOR = "#9E9E9E"   # 灰:有量測但信心低
BACKGROUND = "#101216"

pg.setConfigOptions(antialias=False, background=BACKGROUND, foreground="#C7CBD1")


# =============================================================================
# face mesh 疊圖
# =============================================================================
def draw_face_mesh(frame_bgr: np.ndarray, landmarks) -> None:
    """把 face mesh 畫上去(就地修改 frame_bgr)。

    用 mediapipe 1.0.1 的內建繪圖工具(位置在 tasks.python.vision,不是舊的
    mp.solutions——舊路徑在 1.0.1 已經不存在)。
    """
    from mediapipe.tasks.python import vision as mp_vision

    draw, styles = mp_vision.drawing_utils, mp_vision.drawing_styles
    conn = mp_vision.FaceLandmarksConnections
    draw.draw_landmarks(
        image=frame_bgr, landmark_list=landmarks,
        connections=conn.FACE_LANDMARKS_TESSELATION, landmark_drawing_spec=None,
        connection_drawing_spec=styles.get_default_face_mesh_tesselation_style(),
    )
    draw.draw_landmarks(
        image=frame_bgr, landmark_list=landmarks,
        connections=conn.FACE_LANDMARKS_CONTOURS, landmark_drawing_spec=None,
        connection_drawing_spec=styles.get_default_face_mesh_contours_style(),
    )
    draw.draw_landmarks(
        image=frame_bgr, landmark_list=landmarks,
        connections=conn.FACE_LANDMARKS_LEFT_IRIS + conn.FACE_LANDMARKS_RIGHT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=styles.get_default_face_mesh_iris_connections_style(),
    )


# =============================================================================
# 背景取像執行緒
# =============================================================================
class CaptureWorker(QObject):
    """在背景執行緒跑 next_frame() + extract(),結果用 signal 送回 UI 執行緒。

    cap.read() 實測阻塞約 27 ms(等鏡頭吐下一幀)。放進 Qt event loop 會讓
    整個介面卡在那裡,所以整條取像+推論鏈都在這裡跑。
    """

    frameReady = Signal(object)   # (frame_bgr, landmarks, SignalSample, fps)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, source, extractor) -> None:
        super().__init__()
        self._source = source
        self._extractor = extractor
        # run() 是個不回到 event loop 的迴圈,所以停止旗標用 threading.Event,
        # 不能靠 queued slot(那需要 event loop 才送得到)。
        self._stop = threading.Event()
        self.t_capture: list[float] = []
        self.t_extract: list[float] = []
        self.n_face = 0
        self.n_noface = 0
        self.t_first: float | None = None
        self.t_last: float | None = None

    def stop(self) -> None:
        self._stop.set()

    @Slot()
    def run(self) -> None:
        from .capture import CaptureError

        frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
        prev = time.perf_counter()
        while not self._stop.is_set():
            try:
                t0 = time.perf_counter()
                frame = self._source.next_frame()
                t1 = time.perf_counter()
                sample = self._extractor.extract(frame)
                t2 = time.perf_counter()
            except CaptureError as exc:
                self.failed.emit(str(exc))
                break
            except Exception as exc:  # noqa: BLE001 - never kill the thread silently
                self.failed.emit(f"{type(exc).__name__}: {exc}")
                break

            self.t_capture.append(t1 - t0)
            self.t_extract.append(t2 - t1)
            if frame.has_face:
                self.n_face += 1
            else:
                self.n_noface += 1

            now = time.perf_counter()
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
            frame_times.append(now - prev)
            prev = now
            total = sum(frame_times)
            fps = len(frame_times) / total if total > 0 else 0.0

            self.frameReady.emit((frame.frame_bgr, frame.landmarks, sample, fps))
        self.finished.emit()


# =============================================================================
# 主視窗
# =============================================================================
class MainWindow(QMainWindow):
    """鏡頭畫面 + 四條訊號的即時捲動曲線。"""

    closing = Signal()

    def __init__(self, window_seconds: float = WINDOW_SECONDS) -> None:
        super().__init__()
        self.window_seconds = window_seconds
        self.setWindowTitle("FaceTrace - live signal trace")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- 上半:影像 ---
        self.video_label = QLabel("waiting for camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(VIDEO_WIDTH, 360)
        self.video_label.setStyleSheet(f"background:{BACKGROUND}; color:#C7CBD1;")
        layout.addWidget(self.video_label, stretch=3)

        # --- 圖例(說明灰色代表什麼;不得出現情緒詞)---
        self.legend_label = QLabel(
            "solid colour = tracked   |   "
            f"<span style='color:{UNCERTAIN_COLOR}'>grey = low tracking confidence</span>   |   "
            "gap = no measurement (never interpolated)"
        )
        self.legend_label.setStyleSheet("color:#8A9099; font-size:11px;")
        layout.addWidget(self.legend_label)

        # --- 下半:四個子圖 ---
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, stretch=4)

        self._plots: dict[str, pg.PlotItem] = {}
        self._curve_ok: dict[str, pg.PlotDataItem] = {}
        self._curve_uncertain: dict[str, pg.PlotDataItem] = {}

        first_plot = None
        for row, name in enumerate(SIGNAL_NAMES):
            plot = self.plot_widget.addPlot(row=row, col=0)
            plot.setYRange(0.0, 1.0, padding=0.05)
            plot.setLabel("left", name)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)
            if row < len(SIGNAL_NAMES) - 1:
                plot.getAxis("bottom").setStyle(showValues=False)
            else:
                plot.setLabel("bottom", "seconds")

            # 灰線先畫,正常顏色畫在上面
            self._curve_uncertain[name] = plot.plot(
                pen=pg.mkPen(UNCERTAIN_COLOR, width=2), connect="finite"
            )
            self._curve_ok[name] = plot.plot(
                pen=pg.mkPen(SIGNAL_COLORS[name], width=2), connect="finite"
            )
            self._plots[name] = plot

        self.setCentralWidget(central)
        self.status = self.statusBar()
        self.status.showMessage("starting...")

        # --- 資料緩衝 ---
        self._t: deque[float] = deque(maxlen=MAX_SAMPLES)
        self._values: dict[str, deque[float]] = {n: deque(maxlen=MAX_SAMPLES) for n in SIGNAL_NAMES}
        self._states: dict[str, deque[int]] = {n: deque(maxlen=MAX_SAMPLES) for n in SIGNAL_NAMES}
        self.t_render: list[float] = []
        self.t_video: list[float] = []
        self.resize(980, 900)

    # -- 影像 --------------------------------------------------------------
    def push_frame(self, frame_bgr: np.ndarray, landmarks=None) -> None:
        """顯示一幀。有 landmarks 就順便畫 face mesh。"""
        if frame_bgr is None:
            return
        t0 = time.perf_counter()
        if landmarks:
            draw_face_mesh(frame_bgr, landmarks)
        frame_bgr = np.ascontiguousarray(frame_bgr)
        h, w = frame_bgr.shape[:2]
        image = QImage(frame_bgr.data, w, h, frame_bgr.strides[0],
                       QImage.Format.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ))
        self.t_video.append(time.perf_counter() - t0)

    # -- 訊號 --------------------------------------------------------------
    def push_sample(self, sample: SignalSample) -> None:
        """收一個取樣點並重畫曲線。ABSENT 存成 NaN,讓線自然斷開。"""
        self._t.append(sample.timestamp_ms / 1000.0)
        for name in SIGNAL_NAMES:
            state = sample.state.get(name, SignalState.ABSENT)
            value = sample.values.get(name)
            # ABSENT 一律存 NaN——這就是「不內插、不補洞」在資料層的落實
            self._values[name].append(np.nan if value is None else float(value))
            self._states[name].append(int(state))
        self._redraw()

    @staticmethod
    def _mask_for(values: np.ndarray, states: np.ndarray, want: int) -> np.ndarray:
        """取出屬於 want 狀態的線段,其餘填 NaN。

        每個線段(i, i+1)歸屬於右端點 i+1 的狀態,所以某個點只要「下一點是
        want」就要保留,好讓 OK↔UNCERTAIN 交界處的線接得上、不出現假的缺口。
        但 ABSENT 的點永遠不保留,所以線絕不會跨過沒有量測的洞。
        """
        own = states == want
        measured = states != int(SignalState.ABSENT)
        nxt = np.empty_like(own)
        nxt[:-1] = own[1:]
        nxt[-1] = False
        keep = own | (measured & nxt)
        return np.where(keep, values, np.nan)

    def _redraw(self) -> None:
        if not self._t:
            return
        t0 = time.perf_counter()
        t = np.fromiter(self._t, dtype=float, count=len(self._t))
        now = t[-1]
        for name in SIGNAL_NAMES:
            values = np.fromiter(self._values[name], dtype=float, count=len(self._values[name]))
            states = np.fromiter(self._states[name], dtype=int, count=len(self._states[name]))
            self._curve_ok[name].setData(
                t, self._mask_for(values, states, int(SignalState.OK)))
            self._curve_uncertain[name].setData(
                t, self._mask_for(values, states, int(SignalState.UNCERTAIN)))
        self._plots[SIGNAL_NAMES[0]].setXRange(
            max(0.0, now - self.window_seconds), max(self.window_seconds, now), padding=0
        )
        self.t_render.append(time.perf_counter() - t0)

    # -- 背景執行緒送來的整包 ------------------------------------------------
    @Slot(object)
    def on_frame(self, payload) -> None:
        """CaptureWorker.frameReady 的接收端(已經在主執行緒)。"""
        frame_bgr, landmarks, sample, fps = payload
        self.push_frame(frame_bgr, landmarks)
        self.push_sample(sample)
        face = "face tracked" if sample.has_face else "NO FACE"
        self.status.showMessage(
            f"{fps:5.1f} FPS   |   confidence {sample.confidence:4.2f}   |   {face}"
        )

    # -- 生命週期 ----------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.closing.emit()
        super().closeEvent(event)

    # -- M0-T04 的事,本任務不實作 ------------------------------------------
    def show_diary(self, all_samples, change_points) -> None:
        """完整軌跡 + 變化點的日記頁。M0-T04 實作,這裡只留簽名。"""
        raise NotImplementedError("show_diary is M0-T04")


def run_app(window: MainWindow, app: Optional[QApplication] = None) -> int:
    """顯示視窗並進入 Qt event loop,回傳離開碼。"""
    owns_app = app is None
    if owns_app:
        app = QApplication.instance() or QApplication([])
    window.show()
    return app.exec()
