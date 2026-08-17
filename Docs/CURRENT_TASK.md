# CURRENT_TASK

## M0-T02:取像 + 訊號抽取

## 目標
把 M0 的感測地基做出來:鏡頭畫面疊 face mesh,終端即時印出四條訊號值與追蹤信心,VIDEO 模式下 30 FPS 以上。這是 8/20 主建造日的第一塊,T03 的曲線視窗直接吃這裡的輸出。

## 工程師要做的
1. 實作 `src/facetrace/capture.py`:
   - OpenCV 開本機鏡頭逐幀取像。
   - Face Landmarker 以 **VIDEO 模式**推論(單調遞增時間戳),輸出 blendshapes + landmarks + transformation matrix。
   - 每幀產出 FrameResult(timestamp_ms、frame_bgr、has_face、blendshapes、landmarks、head_pose)。沒臉時回空結果,不內插。
2. 實作 `src/facetrace/signals.py` 的 extract 部分:
   - brow_down = browDownLeft/Right 平均(0..1)
   - mouth_smile = mouthSmileLeft/Right 平均(0..1)
   - head_stability = transformation matrix 逐幀變化量映射(穩定→高,0..1),公式自選並在 run log 說明
   - mic_volume = 麥克風 RMS(sounddevice,已在環境內);沒有麥克風時整條缺席,不准 crash
   - 每個取樣點帶 valid 旗標;追蹤信心的代理指標由工程師調查後決定(FaceLandmarker 若無逐幀分數,可用「近 N 幀偵測率」等代理),寫進 run log。
   - smooth 與 change_points 本任務**不做**(T04 的事)。
3. 新增 `scripts/run_capture.py` demo 腳本:
   - cv2.imshow 顯示鏡頭畫面 + face mesh 疊圖(MediaPipe 內建繪圖工具即可)。
   - 終端以約每 0.2 秒一次的頻率印四條訊號值 + valid/信心 + 即時 FPS(印太快會拖累 FPS)。
   - 按 q 離開,資源正確釋放。找不到鏡頭時印清楚錯誤並非零退出。
4. 寫 run log 到 `Docs/codex_runs/M0-T02_001_capture_signals.md`,附真實終端輸出與實測 FPS。

## 不准碰的
- `ui.py`、`session.py` 維持空殼;不引入 pyqtgraph/PySide6 程式碼(T03 的事)。
- 不做錄製、不做 60 秒計時、不做變化點。
- 不改 requirements.txt(除非裝不起來,要在 run log 說明)。

## 完成定義(使用者人工驗證)
- 斷網跑 `python scripts/run_capture.py`:看到鏡頭畫面 + face mesh。
- 皺眉 → brow_down 上升;微笑 → mouth_smile 上升;搖頭 → head_stability 下降;講話 → mic_volume 跳動(有麥克風時)。
- 終端印的 FPS ≥ 30。
- 用手遮住鏡頭:訊號標 invalid,不 crash、不亂猜值。
