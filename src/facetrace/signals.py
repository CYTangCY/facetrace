"""signals — 四條訊號抽取。

M0 只有這四條,不加第五條:
- brow_down       (browDownLeft + browDownRight) / 2,0..1
- mouth_smile     (mouthSmileLeft + mouthSmileRight) / 2,0..1
- head_stability  由 facial transformation matrix 的逐幀變化率映射,0..1(越穩定越高)
- mic_volume      麥克風 RMS 映射到 0..1(可選;沒有麥克風時整條缺席)

每個取樣點都帶三態旗標 SignalState(M0-T03 修正,理由見 run log M0-T03_001):
- ABSENT     沒有量測(臉不在畫面、麥克風缺席)→ 值為 None,UI 把線斷開。
- UNCERTAIN  有量測但追蹤信心低 → 值照常寫入,UI 畫成灰線(線是連的)。
- OK         有量測且信心足夠 → 值照常,UI 畫正常顏色。

這三態必須分開,否則紅線三「信心低就變灰」在畫面上永遠看不到灰線,只看得到
斷點——因為「沒量到」和「量到但不可信」被壓成了同一種輸出。
禁止內插填補 ABSENT 區段;ABSENT 時值為 None,不沿用上一幀。

追蹤信心(代理指標):
mediapipe 1.0.1 的 FaceLandmarkerResult 只有 face_landmarks /
face_blendshapes / facial_transformation_matrixes 三個欄位,沒有逐幀信心分數;
NormalizedLandmark 的 visibility 與 presence 欄位實測恆為 None。因此改用代理
指標:最近 CONFIDENCE_WINDOW 幀的偵測成功率。詳見 run log M0-T02_001。

變化點(change point)與 smooth 是 M0-T04 的事,本模組此階段不實作,
草案簽名保留在下方註解。

紅線:訊號名只准是上述四個 key;程式碼、註解、輸出裡不出現情緒詞當輸出名。
"""

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from .capture import FrameResult

# 四條訊號的正式名稱,順序固定(UI 與 run log 都照這個順序)
SIGNAL_NAMES: tuple[str, ...] = ("brow_down", "mouth_smile", "head_stability", "mic_volume")

# --- 追蹤信心代理指標 ---------------------------------------------------------
CONFIDENCE_WINDOW = 15      # 近 15 幀 ≈ 30 FPS 下的 0.5 秒
CONFIDENCE_THRESHOLD = 0.6  # 偵測率低於此值 -> UNCERTAIN(灰線),不是 ABSENT

# --- head_stability 映射常數(由 scripts 外的實測校正而來,見 run log)---------
# 實測(合成臉、已知角速度):完全靜止的雜訊底約 2.4 deg/s;1 deg/frame 的旋轉
# 被忠實地量成 29.8 deg/s。K_ROT=60 deg/s 讓「坐著不動」落在 0.95 附近、
# 「明顯轉頭」約 0.4、「搖頭」掉到 0.05 以下,對比清楚。
K_ROT_DEG_PER_S = 60.0
# 平移實測:靜止雜訊底約 0.19 unit/s,整張臉快速位移約 36 unit/s。
K_TRANS_PER_S = 30.0
# dt 的合理範圍;超出就不算變化率(避免掉幀造成假的劇烈變化)
MIN_DT_S = 1e-3
MAX_DT_S = 0.5

# --- mic_volume 映射常數 ------------------------------------------------------
# RMS(-1..1 的浮點取樣)轉 dBFS 後線性映射到 0..1。
# -60 dBFS ≈ 安靜房間底噪,-10 dBFS ≈ 大聲說話,正常說話落在中段。
MIC_DB_FLOOR = -60.0
MIC_DB_CEILING = -10.0
MIC_SAMPLE_RATE = 16000
MIC_BLOCK_SIZE = 1024


class SignalState(IntEnum):
    """單一訊號在單一時間點的三態。UI 的渲染方式直接由它決定。"""

    ABSENT = 0     # 沒有量測 -> 值為 None -> 線斷開(NaN,絕不跨洞連線)
    UNCERTAIN = 1  # 有量測但信心低 -> 值可用 -> 畫灰線(線是連的)
    OK = 2         # 有量測且信心足夠 -> 畫正常顏色


@dataclass
class SignalSample:
    """單一時間點的四條訊號。

    state[name] 是唯一的真相來源;valid 與 measured 都由它推導,
    所以不可能出現「valid=True 但沒量測」這種自相矛盾的組合。

    - state == ABSENT     -> values[name] is None,UI 斷線
    - state == UNCERTAIN  -> values[name] 有值,UI 畫灰線
    - state == OK         -> values[name] 有值,UI 畫正常顏色
    """

    timestamp_ms: int
    values: dict[str, Optional[float]] = field(default_factory=dict)
    state: dict[str, SignalState] = field(default_factory=dict)
    confidence: float = 0.0     # 追蹤信心代理值 0..1(近 N 幀偵測率)
    has_face: bool = False

    @property
    def valid(self) -> dict[str, bool]:
        """只有 OK 才算 valid(向後相容 T02 的 sample.valid[name] 用法)。"""
        return {k: (v == SignalState.OK) for k, v in self.state.items()}

    @property
    def measured(self) -> dict[str, bool]:
        """有沒有量測值。UNCERTAIN 也算有量測——這正是灰線存在的前提。"""
        return {k: (v != SignalState.ABSENT) for k, v in self.state.items()}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class MicLevel:
    """麥克風 RMS 取樣(sounddevice callback,獨立執行緒)。

    開不起來(沒有麥克風、被佔用、沒有驅動)時 available=False,印一行警告,
    絕不 crash——mic_volume 是可選訊號,整條缺席即可。
    """

    def __init__(self, samplerate: int = MIC_SAMPLE_RATE, blocksize: int = MIC_BLOCK_SIZE) -> None:
        self.available = False
        self.device_name = "(none)"
        self._rms = 0.0
        self._lock = threading.Lock()
        self._stream = None

        try:
            import sounddevice as sd

            info = sd.query_devices(kind="input")
            # 裝置名稱可能含非 ASCII 字元,Windows 主控台編碼有限,先消毒
            raw_name = str(info.get("name", "input"))
            self.device_name = raw_name.encode("ascii", "replace").decode("ascii")

            self._stream = sd.InputStream(
                samplerate=samplerate,
                blocksize=blocksize,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self.available = True
        except Exception as exc:  # noqa: BLE001 - any audio failure degrades gracefully
            reason = str(exc).encode("ascii", "replace").decode("ascii")
            print(
                f"[warn] microphone unavailable ({type(exc).__name__}: {reason}); "
                "mic_volume will be absent for this run",
                file=sys.stderr,
            )
            self.available = False
            self._stream = None

    def _callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            pass  # 溢位/欠載只影響單一區塊,忽略即可
        block = np.asarray(indata, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
        with self._lock:
            self._rms = rms

    def level(self) -> Optional[float]:
        """回傳 0..1 的音量,沒有麥克風時回 None。"""
        if not self.available:
            return None
        with self._lock:
            rms = self._rms
        if rms <= 1e-9:
            return 0.0
        db = 20.0 * math.log10(rms)
        return _clamp01((db - MIC_DB_FLOOR) / (MIC_DB_CEILING - MIC_DB_FLOOR))

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001 - closing must never raise
                pass
            self._stream = None
        self.available = False


def _rotation_angle_deg(prev_pose: np.ndarray, cur_pose: np.ndarray) -> float:
    """兩個 4x4 姿態矩陣之間的旋轉夾角(度)。"""
    r_delta = prev_pose[:3, :3].T @ cur_pose[:3, :3]
    cos_theta = (float(np.trace(r_delta)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


class SignalExtractor:
    """把 FrameResult 轉成 SignalSample。

    持有兩份跨幀狀態:
    - 上一幀的頭部姿態(算 head_stability 的變化率用)
    - 近 N 幀的偵測成功率(追蹤信心代理指標)
    """

    def __init__(self, mic: Optional[MicLevel] = None) -> None:
        self._mic = mic
        self._detect_history: list[bool] = []
        self._prev_pose: Optional[np.ndarray] = None
        self._prev_pose_ts_ms: Optional[int] = None

    @property
    def confidence(self) -> float:
        """近 N 幀的偵測成功率。歷史還沒滿 N 幀時,以已有的幀數計算。"""
        if not self._detect_history:
            return 0.0
        return sum(self._detect_history) / len(self._detect_history)

    def extract(self, frame: FrameResult) -> SignalSample:
        self._detect_history.append(frame.has_face)
        if len(self._detect_history) > CONFIDENCE_WINDOW:
            self._detect_history.pop(0)

        confidence = self.confidence
        # 有沒有量測,只看臉在不在;信心只決定 OK 還是 UNCERTAIN。
        # 這是 M0-T03 的核心修正:低信心不再抹掉量測值,否則灰線永遠畫不出來。
        confident = confidence >= CONFIDENCE_THRESHOLD
        face_state = SignalState.OK if confident else SignalState.UNCERTAIN

        values: dict[str, Optional[float]] = {}
        state: dict[str, SignalState] = {}

        # --- blendshape 訊號:brow_down / mouth_smile -------------------------
        blend = frame.blendshapes
        for name, (left_key, right_key) in (
            ("brow_down", ("browDownLeft", "browDownRight")),
            ("mouth_smile", ("mouthSmileLeft", "mouthSmileRight")),
        ):
            if frame.has_face and blend and left_key in blend and right_key in blend:
                values[name] = _clamp01((blend[left_key] + blend[right_key]) / 2.0)
                state[name] = face_state
            else:
                values[name] = None
                state[name] = SignalState.ABSENT

        # --- head_stability --------------------------------------------------
        stability: Optional[float] = None
        if frame.has_face and frame.head_pose is not None:
            if self._prev_pose is not None and self._prev_pose_ts_ms is not None:
                dt = (frame.timestamp_ms - self._prev_pose_ts_ms) / 1000.0
                if MIN_DT_S <= dt <= MAX_DT_S:
                    rot_rate = _rotation_angle_deg(self._prev_pose, frame.head_pose) / dt
                    trans_rate = (
                        float(np.linalg.norm(frame.head_pose[:3, 3] - self._prev_pose[:3, 3])) / dt
                    )
                    stability = _clamp01(
                        math.exp(-(rot_rate / K_ROT_DEG_PER_S + trans_rate / K_TRANS_PER_S))
                    )
                # dt 超出合理範圍(掉幀、臉剛回來):不算,維持 None
            self._prev_pose = frame.head_pose
            self._prev_pose_ts_ms = frame.timestamp_ms
        else:
            # 臉不在畫面:丟掉上一幀姿態,臉回來後從頭累積,
            # 不跨越空窗期算變化率(那會是假的劇烈變化)。
            # 注意門檻是 has_face,不是信心——低信心期間姿態仍然量得到,
            # 線要連著畫成灰色,只有真的沒臉才准斷。
            self._prev_pose = None
            self._prev_pose_ts_ms = None

        values["head_stability"] = stability
        state["head_stability"] = SignalState.ABSENT if stability is None else face_state

        # --- mic_volume(可選,與臉無關)---------------------------------------
        mic_value = self._mic.level() if self._mic is not None else None
        values["mic_volume"] = mic_value
        # 麥克風有開就是可信的量測,不受臉的信心影響。
        state["mic_volume"] = SignalState.ABSENT if mic_value is None else SignalState.OK

        return SignalSample(
            timestamp_ms=frame.timestamp_ms,
            values=values,
            state=state,
            confidence=confidence,
            has_face=frame.has_face,
        )


# --- M0-T04 的事,本任務不實作 ------------------------------------------------
# def smooth(samples, window) -> samples
#     EMA 或移動平均,只作用於 valid 區段,不跨 invalid 區段內插。
# def change_points(samples, window, z_threshold) -> list[ChangePoint]
#     rolling z-score,只標「第 N 秒有明顯變化」與幅度,不解釋成任何東西。
