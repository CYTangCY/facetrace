"""ui — pyqtgraph 曲線視窗(M0-T01 空殼,尚無實作)。

原生視窗(PySide6 + pyqtgraph),不做網頁。

畫面組成(M0):
- 上半:鏡頭畫面 + face mesh 疊圖(OpenCV 幀轉 QImage 顯示)。
- 下半:四條訊號的即時捲動曲線(brow_down、mouth_smile、head_stability、mic_volume),
  各自獨立子圖,共用時間軸(0..60 秒)。
- 倒數計時與狀態列(錄製中 / replay 中 / 未偵測到臉)。
- Session 結束後切到「日記頁」:完整 60 秒軌跡 + 自動標出的變化點
  (只標「第 N 秒有明顯變化」,不貼任何情緒標籤)。

不確定性呈現:
- invalid 取樣點(信心低、臉離開)畫成灰色線段;不內插、不補洞。

未來介面(草案,實作時可調):
- MainWindow(signals_spec) -> 視窗物件
- MainWindow.push_frame(frame_bgr, landmarks)
- MainWindow.push_sample(SignalSample)
- MainWindow.show_diary(all_samples, change_points)
- run_app(controller) -> 進入 Qt event loop

紅線:UI 文字與圖例不得出現情緒詞;不呼叫網路。
"""
