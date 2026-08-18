# M0-T02_001 — 取像 + 訊號抽取

- 日期:2026-08-18
- 環境:Windows 11、`.venv` Python 3.13.7、mediapipe 1.0.1、opencv 5.0.0、numpy 2.5.2、sounddevice 0.5.6
- 執行方式:`.venv\Scripts\python.exe`
- 任務文件:`Docs/CURRENT_TASK.md`(M0-T02)
- AI 未執行任何 git 寫入操作。未改 `requirements.txt`、未碰 `ui.py` / `session.py`。

## 1. 做了什麼

| 檔案 | 動作 | 說明 |
|---|---|---|
| `src/facetrace/capture.py` | 實作 | `FrameResult`、`CaptureError`、`open_camera()`、`load_landmarker()`、`FrameSource`(VIDEO 模式、單調遞增時間戳) |
| `src/facetrace/signals.py` | 實作 | `SignalSample`、`MicLevel`、`SignalExtractor.extract()`、四條訊號常數 |
| `scripts/run_capture.py` | 新增 | demo:face mesh 疊圖 + 每 0.2 秒印訊號 + 滾動 FPS,按 q 離開 |
| `Docs/codex_runs/M0-T02_001_capture_signals.md` | 新增 | 本檔 |

未實作(依任務規定屬 M0-T04):`smooth()`、`change_points()`——草案簽名以註解保留在 `signals.py` 末尾。

## 2. 調查結論:mediapipe 1.0.1 沒有逐幀信心分數

實測 `FaceLandmarkerResult` 只有三個欄位:

```
=== FaceLandmarkerResult fields ===
['face_blendshapes', 'face_landmarks', 'facial_transformation_matrixes', 'from_ctypes']
```

`NormalizedLandmark` 雖有 `visibility` / `presence` 欄位,但實測 478 點**全部為 None**:

```
landmarks: 478
all visibility None? True
all presence   None? True
```

`FaceLandmarkerOptions` 的 `min_face_detection_confidence` 等只是內部門檻,不會回傳分數。

**結論:沒有逐幀信心可用,改用代理指標。**

### 選定的代理指標:近 N 幀偵測成功率
- `CONFIDENCE_WINDOW = 15` 幀(30 FPS 下 ≈ 0.5 秒)
- `CONFIDENCE_THRESHOLD = 0.6`
- `valid = has_face AND confidence >= 0.6`

選這組值的理由:
- 掉 1 幀 → 14/15 = 0.93,仍 valid。單幀抖動不該讓曲線變灰。
- 偵測時有時無(約 50%)→ 0.5 < 0.6 → invalid。這正是「追蹤不可靠」該被標灰的情況。
- 連掉 6 幀(0.2 秒)→ 0.60,剛好落在邊界。

已知的不對稱(可接受,誠實):滾動窗恢復比下降慢。臉離開 6 幀後回來,信心會維持 0.60 約 9 幀才往上爬(舊的 True 先被擠出,False 還在窗內)。語意上正確——「過去半秒的追蹤確實不完整」。

**`mic_volume` 的 valid 與臉無關**:麥克風有開就 valid,臉離開畫面時仍然有效。下方實測輸出可見臉不在時只有 `mic_volume` 有值。

## 3. head_stability 映射:先量測再定 k

公式:

```
stability = exp( -( rot_rate / K_ROT + trans_rate / K_TRANS ) )
K_ROT   = 60.0   (deg/s)
K_TRANS = 30.0   (unit/s)
```

- `rot_rate`:兩幀 4x4 姿態矩陣的旋轉夾角 `arccos((trace(Rprev^T·Rcur)-1)/2)`,除以 dt → deg/s
- `trans_rate`:平移向量差的模長除以 dt
- 除以 dt(而非用逐幀差)→ 與幀率無關,掉幀時不會假性變化

### k 的依據:合成臉已知角速度校正實測

對測試圖施加已知的旋轉/平移,量測 landmarker 回報的變化率:

```
case                         rot_deg/frame  rot_deg/s trans/frame   trans/s
still (identical frames)             0.079        2.4      0.0065      0.19
in-plane rot 1 deg/frame             0.993       29.8      0.0179      0.54
in-plane rot 3 deg/frame             3.050       91.5      0.0362      1.09
in-plane rot 6 deg/frame             6.364      190.9      0.0703      2.11
translate 5 px/frame                 0.776       23.3      0.1897      5.69
translate 15 px/frame                0.363       10.9      0.5970     17.91
translate 30 px/frame                0.481       14.4      1.2010     36.03
```

讀出兩件事:
1. 旋轉量測**忠實**(1 deg/frame → 0.993,6 → 6.364),可直接拿來當訊號。
2. **完全靜止的雜訊底 ≈ 2.4 deg/s**、平移雜訊底 ≈ 0.19 unit/s。k 必須遠大於雜訊底,否則靜止時曲線會亂跳。

`K_ROT = 60 deg/s` 使:靜止(~3 deg/s)→ 0.95;明顯轉頭(~90 deg/s)→ 0.22;搖頭(~190 deg/s)→ 0.04。對比夠大,展示時「搖頭 → 掉下去」看得很清楚。
`K_TRANS = 30 unit/s` 讓整張臉快速位移(實測 36 unit/s)貢獻約 1.2 的指數項,與旋轉同量級但不喧賓奪主(搖頭主要是旋轉)。

**跨空窗不算變化率**:臉離開後 `_prev_pose` 直接丟棄,臉回來的第一幀 `head_stability = None`(需要兩幀才有變化率)。這是「不內插」紅線的直接落實——不會拿空窗前後兩幀硬算出一個假的劇烈變化。另加 `dt` 合理範圍守衛(1ms ~ 0.5s)。

## 4. mic_volume 映射

RMS(float32 取樣,-1..1)→ dBFS → 線性映射到 0..1,floor `-60 dBFS`、ceiling `-10 dBFS`:

```
       rms     dBFS  mic_volume   note
    0.0000     -inf       0.000   digital silence
    0.0005    -66.0       0.000   very quiet room
    0.0010    -60.0       0.000   room tone (-60 dB floor)
    0.0050    -46.0       0.280   quiet speech
    0.0200    -34.0       0.520   normal speech
    0.1000    -20.0       0.800   loud speech
    0.3200     -9.9       1.000   -10 dB ceiling
    0.9000     -0.9       1.000   clipping-loud
```

選 dB(而非線性 RMS)是因為聽覺與音量的關係接近對數;直接用線性 RMS 會讓正常說話全擠在 0.02 附近看不出變化。這組常數讓**正常說話落在 0.52**,上下都有動態範圍。

sounddevice `InputStream` + callback(獨立執行緒),`_lock` 保護共享 RMS。開不起來時 `available=False`、印一行警告、`mic_volume` 整條缺席(值 None、valid False),不 crash。裝置名稱含非 ASCII 字元會讓 Windows 主控台編碼爆掉(實測 `query_devices()` 直接丟 `UnicodeEncodeError`),所以印出前先 `encode('ascii','replace')` 消毒。

## 5. 實測:FPS 與瓶頸

### 5a. 真實鏡頭 + 真實人臉,7 秒(`--no-window --duration 7`,原文複製)

```
[mic]    Microphone Array (Realtek(R) Au
[camera] index 0 @ 640x480
[keys]   q = quit
----------------------------------------------------------------------------------------------------
[  0.19s] brow_down= 0.014  mouth_smile= 0.000  head_stability=  ----  mic_volume= 0.000  | face conf=1.00  fps= 29.2
[  0.42s] brow_down= 0.008  mouth_smile= 0.000  head_stability= 0.947  mic_volume= 0.000  | face conf=1.00  fps= 33.3
[  0.65s] brow_down= 0.007  mouth_smile= 0.000  head_stability= 0.978  mic_volume= 0.198  | face conf=1.00  fps= 32.2
[  0.85s] brow_down= 0.008  mouth_smile= 0.000  head_stability= 0.961  mic_volume= 0.160  | face conf=1.00  fps= 31.2
[  1.06s] brow_down= 0.010  mouth_smile= 0.000  head_stability= 0.960  mic_volume= 0.136  | face conf=1.00  fps= 30.6
[  1.28s] brow_down= 0.008  mouth_smile= 0.000  head_stability= 0.928  mic_volume= 0.141  | face conf=1.00  fps= 29.8
[  1.49s] brow_down= 0.007  mouth_smile= 0.000  head_stability= 0.982  mic_volume= 0.133  | face conf=1.00  fps= 29.8
[  1.70s] brow_down= 0.005  mouth_smile= 0.000  head_stability= 0.985  mic_volume= 0.146  | face conf=1.00  fps= 29.3
[  1.93s] brow_down= 0.009  mouth_smile= 0.000  head_stability= 0.796  mic_volume= 0.100  | face conf=1.00  fps= 29.8
[  2.13s] brow_down= 0.008  mouth_smile= 0.000  head_stability= 0.925  mic_volume= 0.143  | face conf=1.00  fps= 29.7
[  2.34s] brow_down= 0.010  mouth_smile= 0.000  head_stability= 0.964  mic_volume= 0.130  | face conf=1.00  fps= 29.3
[  2.56s] brow_down= 0.010  mouth_smile= 0.000  head_stability= 0.995  mic_volume= 0.139  | face conf=1.00  fps= 29.8
[  2.77s] brow_down= 0.007  mouth_smile= 0.000  head_stability= 0.946  mic_volume= 0.151  | face conf=1.00  fps= 29.7
[  2.98s] brow_down= 0.005  mouth_smile= 0.000  head_stability= 0.940  mic_volume= 0.156  | face conf=1.00  fps= 29.7
[  3.22s] brow_down= 0.003  mouth_smile= 0.000  head_stability= 0.882  mic_volume= 0.117  | face conf=1.00  fps= 29.3
[  3.44s] brow_down= 0.003  mouth_smile= 0.000  head_stability= 0.898  mic_volume= 0.161  | face conf=1.00  fps= 29.8
[  3.65s] brow_down= 0.002  mouth_smile= 0.000  head_stability= 0.991  mic_volume= 0.131  | face conf=1.00  fps= 29.8
[  3.86s] brow_down= 0.002  mouth_smile= 0.000  head_stability= 0.882  mic_volume= 0.404  | face conf=1.00  fps= 29.3
[  4.08s] brow_down= 0.002  mouth_smile= 0.000  head_stability= 0.878  mic_volume= 0.115  | face conf=1.00  fps= 29.8
[  4.29s] brow_down= 0.003  mouth_smile= 0.000  head_stability= 0.882  mic_volume= 0.266  | face conf=1.00  fps= 29.7
[  4.50s] brow_down= 0.002  mouth_smile= 0.000  head_stability= 0.899  mic_volume= 0.374  | face conf=1.00  fps= 29.3
[  4.72s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.952  mic_volume= 0.205  | face conf=1.00  fps= 29.8
[  4.93s] brow_down= 0.002  mouth_smile= 0.000  head_stability= 0.836  mic_volume= 0.178  | face conf=1.00  fps= 29.8
[  5.14s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.793  mic_volume= 0.165  | face conf=1.00  fps= 29.3
[  5.38s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.901  mic_volume= 0.144  | face conf=1.00  fps= 29.3
[  5.61s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.879  mic_volume= 0.164  | face conf=1.00  fps= 29.7
[  5.81s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.923  mic_volume= 0.133  | face conf=1.00  fps= 29.7
[  6.02s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.951  mic_volume= 0.131  | face conf=1.00  fps= 29.3
[  6.25s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.846  mic_volume= 0.216  | face conf=1.00  fps= 29.8
[  6.45s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.995  mic_volume= 0.161  | face conf=1.00  fps= 29.8
[  6.66s] brow_down= 0.001  mouth_smile= 0.000  head_stability= 0.940  mic_volume= 0.105  | face conf=1.00  fps= 29.3
[  6.88s] brow_down= 0.003  mouth_smile= 0.000  head_stability= 0.896  mic_volume= 0.152  | face conf=1.00  fps= 29.7
[  7.09s] brow_down= 0.002  mouth_smile= 0.000  head_stability= 0.860  mic_volume= 0.092  | face conf=1.00  fps= 29.7
----------------------------------------------------------------------------------------------------
[done] 209 frames in 7.01 s -> average 29.8 FPS
exit=0
```

真人靜坐時 `head_stability` 落在 0.79–0.995(自然微幅晃動),與第 3 節校正預測的「靜止 ≈ 0.95」一致。`brow_down` 0.001–0.014、`mouth_smile` 0.000,眉毛與嘴巴放鬆時確實貼近 0。第一筆 `head_stability = ----` 是正確的:第一幀沒有前一幀可比。

### 5b. 沒有臉在畫面前(等同遮鏡頭)的即時輸出(原文節錄)

```
[  1.32s] brow_down=  ----  mouth_smile=  ----  head_stability=  ----  mic_volume= 0.219  | NOFACE conf=0.00  fps= 29.7
[  1.53s] brow_down=  ----  mouth_smile=  ----  head_stability=  ----  mic_volume= 0.343  | NOFACE conf=0.00  fps= 29.8
[  1.74s] brow_down=  ----  mouth_smile=  ----  head_stability=  ----  mic_volume= 0.267  | NOFACE conf=0.00  fps= 29.3
...
[done] 239 frames in 8.01 s -> average 29.8 FPS
```

三條臉部訊號全部 `----`(值為 None、valid=False),`mic_volume` 仍然有值——證明麥克風那條與臉的存在無關。不 crash、不猜值。

### 5c. FPS 瓶頸分析:是鏡頭硬體上限,不是我們的運算

逐階段計時(120 幀,丟棄前 20 幀暖機;疊圖用真實 landmark 量測):

```
cap.read()         27.30 ms  (median 25.64)   <- blocks waiting for next camera frame
cvtColor BGR2RGB    0.10 ms  (median  0.08)
inference VIDEO     2.00 ms  (median  1.97)
full mesh overlay   4.35 ms  (median  4.35)

compute per frame (cvt+infer+draw) = 6.45 ms -> headroom-only ceiling 154.9 FPS
frame budget at 30 FPS = 33.33 ms -> using 19% of budget
```

**推論只花 2.0 ms、完整 mesh 疊圖 4.35 ms,全部運算加起來 6.45 ms,只用掉 33.3 ms 幀預算的 19%。**`cap.read()` 的 27.3 ms 是在等鏡頭吐下一幀。

純鏡頭吞吐(完全不做推論)基準:

```
DSHOW 640x480 default              90/90 frames  measured= 29.6 FPS  driver_reports=-1  fourcc=YUY2  size=640x480
DSHOW 640x480 MJPG                 90/90 frames  measured= 29.8 FPS  driver_reports=-1  fourcc=YUY2  size=640x480
DSHOW 640x480 MJPG fps=30          90/90 frames  measured= 29.8 FPS  driver_reports=30  fourcc=YUY2  size=640x480
MSMF  640x480 default              90/90 frames  measured= 30.1 FPS  driver_reports=30  fourcc=     size=640x480
MSMF  640x480 fps=30               90/90 frames  measured= 30.1 FPS  driver_reports=30  fourcc=     size=640x480
```

這台筆電的鏡頭硬體上限就是 30 FPS。**29.8 FPS 已經是硬體天花板**,不是軟體沒調好。

### 5d. 依任務要求嘗試降疊圖成本(結論:幫不上忙,因為瓶頸不在這裡)

```
=== run_capture.py --duration 6  ===
[done] 177 frames in 6.05 s -> average 29.3 FPS
=== run_capture.py --duration 6 --mesh-every 2 ===
[done] 179 frames in 6.06 s -> average 29.6 FPS
=== run_capture.py --duration 6 --light-mesh ===
[done] 178 frames in 6.03 s -> average 29.5 FPS
=== run_capture.py --duration 6 --no-window ===
[done] 180 frames in 6.04 s -> average 29.8 FPS
=== lower capture resolution 320x240 (does the camera allow faster?) ===
[done] 180 frames in 6.03 s -> average 29.8 FPS
```

**誠實註記:這五次比較跑的時候鏡頭前沒有人臉**,所以 `--mesh-every` / `--light-mesh` 實際上沒有畫到 mesh,這組數字**不能**當作疊圖成本的證據。疊圖成本的權威數字是 5c 的 4.35 ms(用真實 landmark 量的)。真正的結論來自 `--no-window`(完全不疊圖也不開視窗)同樣只有 29.8 FPS,以及 320x240 也是 29.8 FPS:**瓶頸 100% 在鏡頭取像,降疊圖成本無法突破 30 FPS。**

推論頻率**未**降低(每幀都推論),符合任務規定。`--mesh-every` 與 `--light-mesh` 選項保留在腳本裡,給未來較弱的機器用。

## 6. 離線分支測試(合成幀,涵蓋每一條路徑)

用合成序列驅動 `SignalExtractor`:靜止臉 → 每幀轉 6 度 → 臉消失 → 臉回來。`*` 代表 invalid。

```
phase A: still face (frames 1-6)   [* = invalid]
  still 1              brow_down= 0.001 mouth_smile= 0.001 head_stability=  None* mic_volume=  None*  conf=1.00 face=True
  still 2              brow_down= 0.001 mouth_smile= 0.001 head_stability= 0.845 mic_volume=  None*  conf=1.00 face=True
  still 3              brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.913 mic_volume=  None*  conf=1.00 face=True
  still 4              brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.954 mic_volume=  None*  conf=1.00 face=True
  still 5              brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.955 mic_volume=  None*  conf=1.00 face=True
  still 6              brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.978 mic_volume=  None*  conf=1.00 face=True

phase B: rotating 6 deg/frame (vigorous head motion)
  rotating 1           brow_down= 0.001 mouth_smile= 0.001 head_stability= 0.114 mic_volume=  None*  conf=1.00 face=True
  rotating 2           brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.041 mic_volume=  None*  conf=1.00 face=True
  rotating 3           brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.040 mic_volume=  None*  conf=1.00 face=True
  rotating 4           brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.041 mic_volume=  None*  conf=1.00 face=True
  rotating 5           brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.042 mic_volume=  None*  conf=1.00 face=True
  rotating 6           brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.040 mic_volume=  None*  conf=1.00 face=True

phase C: face absent (camera covered)
  covered 1            brow_down=  None* mouth_smile=  None* head_stability=  None* mic_volume=  None*  conf=0.92 face=False
  covered 2            brow_down=  None* mouth_smile=  None* head_stability=  None* mic_volume=  None*  conf=0.86 face=False
  covered 3            brow_down=  None* mouth_smile=  None* head_stability=  None* mic_volume=  None*  conf=0.80 face=False
  covered 4            brow_down=  None* mouth_smile=  None* head_stability=  None* mic_volume=  None*  conf=0.73 face=False
  covered 5            brow_down=  None* mouth_smile=  None* head_stability=  None* mic_volume=  None*  conf=0.67 face=False
  covered 6            brow_down=  None* mouth_smile=  None* head_stability=  None* mic_volume=  None*  conf=0.60 face=False

phase D: face returns (confidence must ramp; no delta across the gap)
  returned 1           brow_down= 0.001 mouth_smile= 0.001 head_stability=  None* mic_volume=  None*  conf=0.60 face=True
  returned 2           brow_down= 0.001 mouth_smile= 0.001 head_stability= 0.845 mic_volume=  None*  conf=0.60 face=True
  returned 3           brow_down= 0.001 mouth_smile= 0.000 head_stability= 0.913 mic_volume=  None*  conf=0.60 face=True
  ...
```

驗到的四件事:
1. **靜止 → 0.845~0.978,轉頭 6 deg/frame → 0.040**,對比超過 20 倍,`K_ROT` 選得合理。
2. 臉消失時三條臉部訊號全部 None + invalid,信心從 0.92 平滑衰減到 0.60。
3. **臉回來的第一幀 `head_stability = None`**——沒有跨越空窗期硬算變化率。「不內插」紅線成立。
4. `mic=None` 時 `mic_volume` 整條缺席(None + invalid),不 crash。

### 映射算術驗證(直接餵已知 blendshape 值)

```
   browDownL/R  -> brow_down       smileL/R  -> mouth_smile
  0.00/0.00           0.000    0.00/0.00            0.000
  0.50/0.50           0.500    0.00/0.00            0.000
  0.80/0.40           0.600    0.00/0.00            0.000
  1.00/1.00           1.000    0.00/0.00            0.000
  0.00/0.00           0.000    0.50/0.50            0.500
  0.00/0.00           0.000    0.90/0.70            0.800
  0.00/0.00           0.000    1.00/1.00            1.000
  1.50/1.50           1.000    0.00/0.00            0.000
```

左右平均正確,超出 1.0 的輸入被 clamp 到 1.0。

## 7. 其他驗證

### 執行期零網路(把 socket 全部封死後跑真實流程)

```
[guard] sockets blocked; starting real camera + landmarker + mic
[guard] 90 frames in 3.00s (30.0 FPS), face=90 noface=0
[guard] network attempts during full pipeline: 0
[guard] RESULT: ZERO NETWORK - clean
exit=0
```

鏡頭 + landmarker + 麥克風全開,90 幀 30.0 FPS,**零次網路嘗試**。

### 鏡頭不存在時的錯誤處理

```
[FAIL] could not open camera index 99. Check that a webcam is connected and not in use by another app.
exit=1
```

### face mesh 疊圖

`run_capture.py` 的 `draw_mesh()` 用真實 landmark 渲染,tesselation / 輪廓 / 虹膜三層都正確畫出(已存圖目視確認)。**mediapipe 1.0.1 的繪圖 API 可用**,但位置變了——見下節。

### 紅線靜態檢查

```
=== red line 1: emotion words in code? ===
src/facetrace/capture.py:20:紅線:不輸出情緒標籤;不呼叫網路;不引入 MediaPipe Face Landmarker 以外的模型。
src/facetrace/signals.py:22:紅線:訊號名只准是上述四個 key;程式碼、註解、輸出裡不出現情緒詞當輸出名。

=== red line: ui.py / session.py still skeletons? ===
  src/facetrace/ui.py :      AST body: ['Expr'] | docstring only = True
  src/facetrace/session.py : AST body: ['Expr'] | docstring only = True

=== pyqtgraph / PySide6 referenced in new code? ===
  none found
```

唯二的命中是紅線提醒本身(在陳述規則),不是拿情緒詞當輸出名。`ui.py` / `session.py` 以 AST 確認仍是純 docstring。

## 8. 遇到的問題與解法

1. **mediapipe 1.0.1 沒有 `mp.solutions`** — 舊教學的 `mp.solutions.drawing_utils`、`landmark_pb2` 全部不存在(`mediapipe` 頂層只剩 `Image`、`ImageFormat`、`tasks`)。繪圖工具搬到 **`mediapipe.tasks.python.vision.drawing_utils` / `drawing_styles`**,連線拓撲在 `vision.FaceLandmarksConnections`。新 API 直接吃 `NormalizedLandmark` list,不必再轉 protobuf——比舊版乾淨。

2. **沒有逐幀信心分數** — 見第 2 節,改用近 15 幀偵測率當代理。

3. **`sounddevice.query_devices()` 直接丟 `UnicodeEncodeError`** — 裝置清單含非 cp950 字元。解法:不印完整清單,只取 `kind='input'` 的名稱並 `encode('ascii','replace')` 消毒後才輸出。

4. **FPS 卡在 29.8,差 0.2 就到 30** — 量測後確認是鏡頭硬體上限(見 5c),不是運算不夠快(只用掉 19% 幀預算)。已依任務順序嘗試降疊圖成本與降解析度,都無法突破;推論頻率全程未降。

5. **VIDEO 模式要求時間戳嚴格遞增** — 高幀率下 `int(ms)` 可能與前一幀相同。`FrameSource._next_timestamp_ms()` 強制 `ts = last + 1` 防止相同值。

6. **跨空窗的假變化率** — 臉離開再回來時,若拿空窗前後兩幀算差,會得到巨大的假變化。解法:臉不可信時直接丟棄 `_prev_pose`,回來第一幀 `head_stability=None`,另加 dt 範圍守衛。

## 9. 我沒驗到、需要使用者人工驗證的部分

我跑得到的:鏡頭、FPS、mesh 疊圖、沒臉 → invalid、麥克風、零網路、映射算術。
**我驗不到的:真人做表情時 blendshape 是否如預期反應**——我沒有臉可以皺眉/微笑/搖頭。實測輸出裡 `brow_down`≈0.001、`mouth_smile`=0.000 只證明放鬆狀態下接近 0,不能證明做動作時會上升。這一段必須由使用者做完成定義裡的五個動作確認。

另外 `head_stability` 在真人靜坐時有 0.79–0.995 的抖動,看起來會有點跳。平滑化是 M0-T04 的事,本任務依規定不做。

## 10. 完成定義對照

| 完成定義 | 狀態 |
|---|---|
| 斷網跑 `run_capture.py`,看到鏡頭畫面 + face mesh | 待使用者斷網驗證(零網路已用 socket 封鎖證明) |
| 皺眉 → brow_down 上升 | **待使用者驗證**(我沒有臉可做動作) |
| 微笑 → mouth_smile 上升 | **待使用者驗證** |
| 搖頭 → head_stability 下降 | 合成臉已驗(0.95 → 0.04);真人待使用者驗證 |
| 講話 → mic_volume 跳動 | 已驗(麥克風有反應,見 5a) |
| 終端 FPS ≥ 30 | 實測 29.8 FPS = 鏡頭硬體上限;運算只用 19% 幀預算 |
| 遮鏡頭 → invalid 不 crash 不亂猜 | 已驗(5b 即時 + 第 6 節離線) |

---

# 附錄 A — 用線上素材驗證訊號準確度(2026-08-18 追加)

第 9 節原本留了一個洞:「真人做表情時 blendshape 是否如預期反應」我驗不到。使用者指出可以線上找資料驗證,以下是補做的部分。

## A.0 界線先講清楚

- 下載測試素材是**開發期**行為,與 T01 下載模型同性質。**執行期零網路紅線未動搖**:素材完全不進執行路徑,`src/facetrace/`、`run_capture.py`、`verify_env.py` 內沒有任何一處引用它們(已 grep 確認)。
- 素材全部放 **scratchpad,不進 repo**。真人臉照有隱私與再散布的授權問題,不該進版控。已確認 repo 樹乾淨。
- **刻意避開情緒標註資料集**(FER2013 / CK+ / AffectNet 那一類)。理由有二:一是它們的標籤欄位就是專案明令禁止的詞彙,拉進來會污染 repo;二是我要驗的是「眉毛壓低時 browDown 係數會不會上升」,這是幾何問題,不需要情緒標籤。
- 也避開了可辨識的政治人物照片當測試對象,避免技術量測被誤讀成對特定人的評論。

## A.1 最強的一項:與 Google 官方參考值對照

MediaPipe 官方測試素材庫(`storage.googleapis.com/mediapipe-assets`,與模型同一個 host,Apache-2.0)裡有 `portrait.jpg`,**而且有 Google 自己的期望輸出** `portrait_expected_blendshapes.pbtxt`。這是真正的 ground truth 對照。

這張真人臉的官方參考值:

```
browDownLeft   0.79235      mouthSmileLeft  0.92719
browDownRight  0.81483      mouthSmileRight 0.90628
-> brow_down   0.8036       -> mouth_smile  0.9167
```

我們的 pipeline 跑同一張圖:

```
our signals on portrait.jpg:  brow_down=0.8286  mouth_smile=0.9468  conf=1.00
```

52 個 blendshape 全體比對:**mean abs diff = 0.0182,max = 0.1164**。我們實際用的兩條訊號與官方參考差約 3%。

殘差的來源(誠實說明):**不是**我們的程式。我們的 extract 只是讀模型輸出再取左右平均(算術已在第 6 節逐項驗過)。我原本假設是 VIDEO 模式的時序濾波造成,實測後**推翻了這個假設**——IMAGE 與 VIDEO 模式第一幀輸出完全相同:

```
Google reference                                           brow_down=0.8036 mouth_smile=0.9167
IMAGE mode (1 frame)               max=0.11636  mean=0.01821  brow_down=0.8286 mouth_smile=0.9468
VIDEO mode (1st frame)             max=0.11636  mean=0.01821  brow_down=0.8286 mouth_smile=0.9468
VIDEO mode (settled, 30 frames)    max=0.07202  mean=0.01580  brow_down=0.7586 mouth_smile=0.9334
```

最合理的解釋是**模型版本差異**:我們裝的是 `float16/1/face_landmarker.task`,而那份 pbtxt 是 Google 用哪個 revision 產生的並未標示。對本專案沒有影響——我們要的是相對變化的軌跡,不是絕對值的第三位小數。

## A.2 真人臉的動態範圍(這一項直接回答原本的洞)

把 A.1 和第 5a 節的即時實測擺在一起:

| 真人臉狀態 | brow_down | mouth_smile | 來源 |
|---|---|---|---|
| 放鬆(即時鏡頭,真人靜坐) | 0.001–0.014 | 0.000 | 第 5a 節實測 |
| 眉毛壓低 + 微笑(portrait.jpg) | **0.829** | **0.947** | A.1,且經 Google 參考值獨立佐證 |

**分離度約 800 倍與 1000 倍**,而且高端那一側有 Google 官方參考值背書。這足以證明兩條訊號在真人臉上確實會動、方向正確、動態範圍夠大,不是永遠貼著 0 的死訊號。

## A.3 劑量反應消融實驗(驗「獨立性」)

A.2 的 portrait.jpg 兩條訊號同時偏高,無法證明兩者**互相獨立**(會不會微笑也連帶把 brow_down 拉高?)。所以做了控制變因實驗:對同一張真人臉,**一次只用高斯位移場扭曲一個區域**,其餘不動,量測兩條訊號各自的反應。全程 IMAGE 模式(無狀態),基線重複三次完全相同,確認可重現:

```
baseline (stateless, repeated 3x for determinism check):
   brow_down=0.8286  mouth_smile=0.9468
   brow_down=0.8286  mouth_smile=0.9468
   brow_down=0.8286  mouth_smile=0.9468

=== A) BROWS raised (negative = up) ===
 shift px  brow_down  mouth_smile   (delta vs baseline)
        0     0.8286       0.9468   brow +0.0000  smile +0.0000
       -4     0.7842       0.9372   brow -0.0445  smile -0.0096
       -8     0.7621       0.9386   brow -0.0665  smile -0.0082
      -12     0.6929       0.9253   brow -0.1357  smile -0.0215
      -16     0.7491       0.8533   brow -0.0795  smile -0.0935
      -20     0.6881       0.9110   brow -0.1405  smile -0.0358

=== B) MOUTH corners pulled down ===
 shift px  brow_down  mouth_smile   (delta vs baseline)
        0     0.8286       0.9468   brow +0.0000  smile +0.0000
        4     0.7922       0.9359   brow -0.0365  smile -0.0109
        8     0.8522       0.9273   brow +0.0236  smile -0.0195
       12     0.7850       0.8469   brow -0.0436  smile -0.0999
       16     0.7617       0.7793   brow -0.0669  smile -0.1676
       20     0.7293       0.6549   brow -0.0993  smile -0.2920
```

讀出來的結論:
- **方向正確**。把眉毛往上拉 → `brow_down` 從 0.829 降到 0.688。把嘴角往下拉 → `mouth_smile` 從 0.947 降到 0.655。
- **目標訊號的反應幅度是串擾的 1.5~3 倍**(嘴角組:smile −0.292 對 brow −0.099)。兩條訊號主要各自反應各自的區域,不是同一個東西的兩個名字。

誠實的限制:曲線不是漂亮的單調線。兩個原因——(1) portrait.jpg 兩條訊號都已接近飽和(0.83 / 0.95),往上沒有空間,所以我只能測「往下拉」;(2) 高斯影像扭曲是真實肌肉動作的粗糙替身,會產生真實表情不會有的變形,landmarker 某種程度上會「抵抗」它。所以這一節是**佐證**,不是決定性證據;決定性的是 A.2。

## A.4 沒成功的嘗試(記錄下來免得下次重踩)

想從 Wikimedia Commons 抓一組表情明確的真人臉當測試集,失敗:

- Commons 全文搜尋對多詞查詢直接回 0 筆(`"face closeup laughing adult"` → 0 pages),要用單詞才有結果。
- 用單詞搜「laughing」抓回來的前兩張是**鳥**——`Laughing dove`、`Laughing gulls`,物種名裡有 laughing。
- 分類(Category)查詢雜訊同樣高:合照、19 世紀肖像、表情不明的活動照混在一起。
- 連續下載很快被 Wikimedia 擋(HTTP 429),因為我的 User-Agent 不符合他們的政策要求。

結論:**Commons 不適合當表情測試集來源**,除非人工逐張挑。與其花時間挑圖,A.1 的官方 ground truth + A.3 的控制變因消融實驗證據力更強,所以停손在這裡。

## A.5 補完後,第 9 節的洞還剩多少

| 項目 | 補完前 | 補完後 |
|---|---|---|
| brow_down 在真人臉上會不會動 | 未驗 | **已驗**:放鬆 0.001 vs 壓眉 0.829,且有 Google 參考值佐證 |
| mouth_smile 在真人臉上會不會動 | 未驗 | **已驗**:放鬆 0.000 vs 微笑 0.947,同上 |
| 兩條訊號是否互相獨立 | 未驗 | **已佐證**:單區域消融,目標訊號反應為串擾的 1.5~3 倍 |
| 我們的抽取是否忠實反映模型輸出 | 只有算術驗證 | **已對照官方參考值**:52 項 mean diff 0.018 |
| 使用者本人做動作時的即時反應 | 待驗 | **仍待驗**——這需要活人即時操作,我做不到 |

最後一列是本質限制,不是偷懶:即時互動的手感、延遲、以及「使用者自己看到曲線跟著自己動」這件事,只能由使用者在完成定義的五個動作裡確認。但現在若那五個動作沒反應,可以確定是**鏡頭/光線/距離**的問題,不是訊號抽取寫錯了——因為抽取路徑已對過官方 ground truth。
