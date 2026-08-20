# M0-T03_001 — 即時曲線視窗

- 日期:2026-08-18
- 環境:Windows 11、`.venv` Python 3.13.7、mediapipe 1.0.1、PySide6 6.11.1、pyqtgraph 0.14.0、opencv 5.0.0、numpy 2.5.2
- 執行方式:`.venv\Scripts\python.exe`
- 任務文件:`Docs/CURRENT_TASK.md`(M0-T03)
- AI 未執行任何 git 寫入操作。未改 `requirements.txt`、`session.py` 維持空殼。

## 1. 做了什麼

| 檔案 | 動作 | 說明 |
|---|---|---|
| `src/facetrace/signals.py` | 修正 | 三態語意:新增 `SignalState`(ABSENT / UNCERTAIN / OK),`SignalSample.state` 為唯一真相來源 |
| `src/facetrace/ui.py` | 實作 | `MainWindow`、`push_frame`、`push_sample`、`run_app`、`CaptureWorker`(背景執行緒)、`draw_face_mesh` |
| `scripts/run_ui.py` | 新增 | 接起 FrameSource + SignalExtractor + MainWindow,乾淨退出,附逐階段計時 |
| `scripts/run_capture.py` | 小改 | T02 的 CLI demo 跟上三態語意(UNCERTAIN 印 `~值`,不再誤印成 `----`) |
| `Docs/codex_runs/M0-T03_shots/*.png` | 新增 | 三態渲染截圖 |

未實作(依規定屬 M0-T04):`show_diary()` 只留簽名並 `raise NotImplementedError`;`smooth()`、`change_points()` 仍是註解草案。沒有倒數計時、沒有日記頁、沒有變化點、沒有錄製。

## 2. 三態語意修正:設計決定與理由

### 問題

T02 把兩種不同的情況壓成同一種輸出:

```python
face_trustworthy = frame.has_face and confidence >= CONFIDENCE_THRESHOLD
# 低信心 -> 值直接被抹成 None,和「臉不在」走完全同一條路徑
```

結果是紅線三「信心低就變灰」在畫面上**永遠看不到灰線**——因為低信心的取樣點沒有值可以畫,只會變成一個洞。灰色這個設計語彙形同虛設。

### 選定做法:三態列舉 `SignalState`,`valid` / `measured` 由它推導

任務允許自選(加 `measured` 字典或改三態列舉)。我選**列舉**,理由是**不可能自相矛盾**:

```python
class SignalState(IntEnum):
    ABSENT = 0     # 沒有量測 -> 值 None -> 線斷開
    UNCERTAIN = 1  # 有量測但信心低 -> 值可用 -> 灰線(線是連的)
    OK = 2         # 有量測且信心足夠 -> 正常顏色
```

`SignalSample.state` 是唯一真相來源,`valid` 與 `measured` 都是推導出來的 property:

```python
@property
def valid(self):    return {k: v == SignalState.OK     for k, v in self.state.items()}
@property
def measured(self): return {k: v != SignalState.ABSENT for k, v in self.state.items()}
```

如果改用「`valid` 布林 + 另加 `measured` 字典」兩個平行欄位,就存在寫入時忘記同步、產生 `valid=True` 但 `measured=False` 這種不可能狀態的空間。列舉從型別上就排除它。附帶好處:`sample.valid[name]` 的舊用法**完全向後相容**,T02 的 `run_capture.py` 不改也能跑(已實測)。

### 判斷門檻的關鍵改動

```python
# 有沒有量測,只看臉在不在;信心只決定 OK 還是 UNCERTAIN
confident  = confidence >= CONFIDENCE_THRESHOLD
face_state = SignalState.OK if confident else SignalState.UNCERTAIN
```

`head_stability` 的跨空窗保護門檻也跟著從 `face_trustworthy` 改成 `has_face`——**這是刻意的**。空窗保護要防的是「跨越臉不見的期間硬算變化率」;低信心期間姿態矩陣仍然量得到,線應該連著畫成灰色,只有真的沒臉才准斷。dt 範圍守衛(1 ms ~ 0.5 s)與 `mic_volume` 與臉無關的語意都維持不變。

## 3. 離線分支測試:三種狀態都會出現

合成序列:靜止臉 15 幀(信心飽和)→ 臉消失 10 幀 → 臉回來 12 幀。原文複製:

```
CONFIDENCE_THRESHOLD = 0.6

--- phase 1: face present, confidence saturated  -> expect OK ---
still 13       brow_down=0.001/OK        mouth_smile=0.000/OK        head_stability=1.000/OK         conf=1.00
still 14       brow_down=0.001/OK        mouth_smile=0.000/OK        head_stability=0.999/OK         conf=1.00
still 15       brow_down=0.001/OK        mouth_smile=0.000/OK        head_stability=1.000/OK         conf=1.00

--- phase 2: face gone (camera covered)  -> expect ABSENT + line break ---
covered 1      brow_down=None /ABSENT    mouth_smile=None /ABSENT    head_stability=None /ABSENT     conf=0.93
covered 2      brow_down=None /ABSENT    mouth_smile=None /ABSENT    head_stability=None /ABSENT     conf=0.87
covered 9      brow_down=None /ABSENT    mouth_smile=None /ABSENT    head_stability=None /ABSENT     conf=0.40
covered 10     brow_down=None /ABSENT    mouth_smile=None /ABSENT    head_stability=None /ABSENT     conf=0.33

--- phase 3: face returns, confidence still low -> expect UNCERTAIN (grey line) ---
returned 1     brow_down=0.001/UNCERTAIN mouth_smile=0.001/UNCERTAIN head_stability=None /ABSENT     conf=0.33
returned 2     brow_down=0.001/UNCERTAIN mouth_smile=0.001/UNCERTAIN head_stability=0.845/UNCERTAIN  conf=0.33
returned 3     brow_down=0.001/UNCERTAIN mouth_smile=0.000/UNCERTAIN head_stability=0.913/UNCERTAIN  conf=0.33
returned 4     brow_down=0.001/UNCERTAIN mouth_smile=0.000/UNCERTAIN head_stability=0.954/UNCERTAIN  conf=0.33
returned 5     brow_down=0.001/UNCERTAIN mouth_smile=0.000/UNCERTAIN head_stability=0.955/UNCERTAIN  conf=0.33
returned 6     brow_down=0.001/UNCERTAIN mouth_smile=0.000/UNCERTAIN head_stability=0.978/UNCERTAIN  conf=0.40
returned 7     brow_down=0.001/UNCERTAIN mouth_smile=0.000/UNCERTAIN head_stability=0.975/UNCERTAIN  conf=0.47
returned 8     brow_down=0.001/UNCERTAIN mouth_smile=0.000/UNCERTAIN head_stability=0.958/UNCERTAIN  conf=0.53
returned 9     brow_down=0.001/OK        mouth_smile=0.000/OK        head_stability=0.962/OK         conf=0.60
returned 10    brow_down=0.001/OK        mouth_smile=0.000/OK        head_stability=0.981/OK         conf=0.67
returned 11    brow_down=0.001/OK        mouth_smile=0.000/OK        head_stability=0.975/OK         conf=0.73
returned 12    brow_down=0.001/OK        mouth_smile=0.000/OK        head_stability=0.981/OK         conf=0.80

state tally over printed rows: ABSENT=13  UNCERTAIN=23  OK=21
PASS: all three states occurred
```

**臉回來後有 8 幀 UNCERTAIN(約 0.27 秒的灰線)**,第 9 幀信心回到 0.60 才轉 OK。這正是使用者遮鏡頭再放開時會看到的那段灰線。

不變量檢查(獨立的 extractor,真的走過一次空窗):

```
--- invariant checks (fresh extractor, real gap) ---
ABSENT -> all values None : True
ABSENT -> measured=False  : True
ABSENT -> valid=False     : True
first frame back: conf=0.33  (threshold 0.6)
  head_stability -> ABSENT | value: None  (no cross-gap rate: correct)
  brow_down      -> UNCERTAIN | value=0.0008 | measured=True valid=False  (grey, line stays connected)
  mic_volume (mic=None) -> ABSENT
```

臉回來的第一幀:`head_stability` 仍是 ABSENT(跨空窗保護維持不變,需要兩幀才有變化率),但 `brow_down` 已經是 UNCERTAIN 且有值——**這正是修正前後的差別**,修正前它會是 None,灰線畫不出來。

## 4. 三態渲染:每個線段只畫一次,絕不跨洞

兩條曲線疊在同一個子圖:灰線在下、正常顏色在上,各自用 NaN + `connect="finite"`。難點是 OK 與 UNCERTAIN 的交界不能出現假的缺口。規則是**每個線段歸屬於右端點的狀態**:

```python
keep[i] = own[i] or (measured[i] and own[i+1])
```

`measured[i]` 這個條件讓線永遠不會跨過 ABSENT 的洞。窮舉測試(原文):

```
all OK                             ['OK', 'OK', 'OK', 'OK']
                                   colour segs [0, 1, 2]   grey segs []   OK
all UNCERTAIN                      ['UNCERT', 'UNCERT', 'UNCERT', 'UNCERT']
                                   colour segs []   grey segs [0, 1, 2]   OK
OK -> UNCERTAIN                    ['OK', 'OK', 'UNCERT', 'UNCERT']
                                   colour segs [0]   grey segs [1, 2]   OK
UNCERTAIN -> OK (grey recovery)    ['UNCERT', 'UNCERT', 'OK', 'OK']
                                   colour segs [1, 2]   grey segs [0]   OK
gap in the middle                  ['OK', 'OK', 'ABSENT', 'ABSENT', 'OK', 'OK']
                                   colour segs [0, 4]   grey segs []   OK
gap then grey then OK              ['OK', 'ABSENT', 'ABSENT', 'UNCERT', 'UNCERT', 'OK', 'OK']
                                   colour segs [4, 5]   grey segs [3]   OK
single absent frame                ['OK', 'OK', 'ABSENT', 'OK', 'OK']
                                   colour segs [0, 3]   grey segs []   OK

RESULT: all cases correct
```

檢查的三件事都通過:覆蓋率(每個合法線段都被畫到)、無重複繪製、**沒有任何線段跨過 ABSENT**。

## 5. 截圖(三態渲染)

路徑:`Docs/codex_runs/M0-T03_shots/`

| 檔案 | 內容 |
|---|---|
| `01_normal.png` | 四條訊號正常追蹤,各自顏色實線 |
| `02_absent_gap.png` | 臉離開:三條臉部訊號**斷開**,`mic_volume` 照常繼續 |
| `03_uncertain_grey.png` | 臉回來但信心仍低:**灰線**,線是連的 |
| `04_recovered_all_three.png` | 一張圖看完:正常色 → 斷開 → 灰 → 正常色 |

**這些截圖刻意用合成臉(`scripts/test_face.jpg`)而不是即時鏡頭。** 截圖要進 repo,而 repo 會推上 GitHub;放真人臉等於把使用者的肖像放進版控。曲線的三態渲染與真人與否無關,用合成臉一樣完整證明。

`02_absent_gap.png` 可見:三條臉部訊號在 11 秒處停住,11~14.5 秒之間**完全沒有線**(不是拉一條直線過去),而 `mic_volume` 一路畫到 14.5 秒——證明麥克風那條與臉的存在無關。狀態列顯示 `NO FACE`。

`04_recovered_all_three.png` 可見完整故事:0~11 秒正常色 → 11~14.5 秒斷開 → 14.5~16.2 秒灰線 → 之後恢復正常色。

## 6. 執行緒架構

`CaptureWorker(QObject)` 用 `moveToThread(QThread)`,在背景跑 `next_frame()` + `extract()`,用 `frameReady` signal 把 `(frame_bgr, landmarks, sample, fps)` 送回主執行緒。跨執行緒 signal 預設是 queued connection,所以 `MainWindow.on_frame` 一定在主執行緒執行——UI 執行緒不碰 OpenCV 取像,背景執行緒不碰任何 widget。

一個容易踩的點:`run()` 是個不回到 event loop 的迴圈,所以停止旗標**不能**用 queued slot(那需要 event loop 才送得到),必須用 `threading.Event`。

## 7. 實測:FPS 與幀預算

### 7a. 真實鏡頭 + UI 開著,12 秒(原文複製)

```
[mic]    Microphone Array (Realtek(R) Au
[camera] index 0 @ 640x480
[keys]   q / Esc = quit (or just close the window)

[timing] per-frame cost (first 5 frames dropped as warm-up)
  capture+inference *    mean  33.66 ms   median  32.07 ms
  signal extract         mean   0.03 ms   median   0.02 ms
  UI video + mesh        mean   0.60 ms   median   0.57 ms
  UI redraw (curves)     mean   0.98 ms   median   0.90 ms
  * capture+inference is dominated by cap.read() blocking until the
    camera delivers the next frame -- it is wait, not work.

  frames processed:      359 frames, 358 intervals in 12.05 s -> 29.7 FPS
  frames with a face:    0   without a face: 359
  our work per frame:    1.60 ms (5% of the 33.3 ms budget at 30 FPS)
[done] camera, microphone and landmarker released
exit=0
```

**29.7 FPS = T02 量到的鏡頭硬體上限**;與絕對數字無關的工程指標是幀預算佔比。

**誠實註記:這次跑的時候鏡頭前沒有人臉**(`frames with a face: 0`),所以 mesh 根本沒畫,`UI video + mesh = 0.60 ms` 只是 QImage 轉換與縮放的成本,**不能**當作 UI 成本的上限。真正的最壞情況見下。

### 7b. 最壞情況的 UI 成本(用合成臉的真實 landmark 量,確定性)

```
--- video panel ---
  push_frame WITHOUT mesh (no face)      mean   0.59 ms   median   0.56 ms
  push_frame WITH face mesh              mean   4.98 ms   median   4.91 ms
  -> face mesh overlay costs               4.38 ms

--- curve redraw, buffer filled to the full 30 s window ---
  buffer holds 900 samples (30 s)
  push_sample + full redraw              mean   1.47 ms   median   1.38 ms

worst case UI work per frame = mesh 4.98 + curves 1.47 = 6.45 ms  (19% of 33.3 ms budget)
```

臉在畫面裡、mesh 全畫、曲線緩衝滿 30 秒(900 取樣點 × 8 條曲線)的最壞情況,UI 共用掉 **6.45 ms = 幀預算的 19%**,還有 81% 餘裕。mesh 成本 4.38 ms 與 T02 獨立量到的 4.35 ms 吻合,互相佐證。

## 8. 其他驗證

### 執行期零網路 + 執行緒乾淨收工(封死 socket 跑真實 UI 流程)

```
[guard] sockets blocked. threads before: 1
[guard] frames rendered: 182   face=0 noface=182
[guard] QThread stopped cleanly: True   isFinished=True
[guard] leftover non-daemon threads: []
[guard] network attempts during full UI pipeline: 0
[guard] RESULT: ZERO NETWORK - clean
exit=0
```

鏡頭 + landmarker + 麥克風 + Qt 全開,**零次網路嘗試**,QThread 正常結束,**沒有殭屍執行緒**。

### 鏡頭不存在

```
[FAIL] could not open camera index 99. Check that a webcam is connected and not in use by another app.
exit=1
```

鏡頭在 Qt 起來之前就先開,失敗直接退出,不會留一個空視窗。

### 紅線靜態檢查

```
=== red line: emotion words in new/changed code? ===
  none found

=== red line: session.py still a pure docstring skeleton? ===
  session.py AST body: ['Expr'] | docstring only = True

=== requirements.txt unchanged? ===
  (empty above = unchanged)

=== show_diary still unimplemented? ===
344:        raise NotImplementedError("show_diary is M0-T04")
```

UI 上出現的文字只有:四個訊號名、`seconds`、圖例 `solid colour = tracked | grey = low tracking confidence | gap = no measurement (never interpolated)`、`NO FACE`、`face tracked`,以及 FPS 與 confidence 數值。沒有任何情緒詞。

## 9. 遇到的問題與解法

1. **pyqtgraph 的 Qt binding** — 環境裡只有 PySide6,pyqtgraph 0.14.0 **自己就選對了**(`QT_LIB chosen: PySide6`),不需要設 `PYQTGRAPH_QT_LIB`。仍然在 `ui.py` import pyqtgraph 之前加了 `os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")` 當防呆——萬一日後有人把 PyQt5 裝進同一個 venv,選擇仍然確定。

2. **`QT_QPA_PLATFORM=offscreen` 沒有字型** — 第一版截圖所有文字都變成空方框(tofu),軸標籤、圖例、狀態列全部不可讀。改用原生 Windows 平台重跑就正常。offscreen 適合跑邏輯測試,**不適合產生要給人看的截圖**。

3. **停止背景迴圈不能用 queued slot** — `CaptureWorker.run()` 不回 event loop,queued slot 永遠送不到。改用 `threading.Event`。

4. **OK 與 UNCERTAIN 交界的假缺口** — 兩條曲線各自做 NaN 遮罩時,交界那一段會兩條都沒畫到,視覺上出現不該有的斷點。用「線段歸屬右端點」的規則解決,並以窮舉測試確認覆蓋率與不跨洞(第 4 節)。

5. **FPS 分母算錯** — 一開始用「程式啟動到收工」當分母,把鏡頭暖機與執行緒關閉的時間也算了進去,得到偏低的 28.3 FPS。改成從第一幀到最後一幀的區間,得到 29.7 FPS,與 T02 的鏡頭上限一致。

6. **UI 成本量測會被「鏡頭前沒有人」誤導** — 沒臉就不畫 mesh,量到的 0.60 ms 不是最壞情況。補了一支確定性的 benchmark(第 7b 節)才拿到真正的上限。

## 10. 完成定義對照

| 完成定義 | 狀態 |
|---|---|
| 斷網跑 `run_ui.py`,視窗開起來,上半看到自己與 face mesh | 零網路已用 socket 封鎖證明;mesh 渲染已驗(截圖);**斷網與「看到自己」待使用者驗證** |
| 皺眉 / 微笑 / 搖頭 / 講話,四條曲線各自跟著動 | **待使用者驗證**(我沒有臉可做表情;訊號抽取本身在 T02 已對過 Google 官方參考值) |
| 捲動順暢不卡頓 | 已驗:29.7 FPS,UI 最壞情況只用 19% 幀預算 |
| 遮鏡頭:曲線斷開(不是拉直線),放開後有一小段灰線再恢復 | **已驗**:離線分支測試(第 3 節)+ 截圖(第 5 節);真人操作待使用者確認 |
| 關掉視窗乾淨結束,無殘留錯誤,鏡頭燈熄滅 | 已驗:exit=0,QThread isFinished=True,無殭屍執行緒,鏡頭與麥克風已釋放 |
