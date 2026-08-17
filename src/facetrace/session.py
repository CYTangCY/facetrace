"""session — 60 秒流程、錄製與 replay(M0-T01 空殼,尚無實作)。

流程狀態機(M0):
  IDLE → PREVIEW(開鏡頭,看到 face mesh)→ RECORDING(60 秒倒數,曲線捲動)
       → DIARY(日記頁:完整軌跡 + 變化點)→ IDLE
  以及 REPLAY(讀取先前錄製檔,以同樣的時間軸重播曲線與畫面)。

錄製(必要功能):
- 每個 session 存到本機一個資料夾:時間戳、每幀 SignalSample(含 valid 旗標)、
  變化點清單、以及可選的原始影像/音量序列,供 replay 使用。
- 資料只寫本機磁碟,不上傳、不呼叫網路。

Replay(必要功能,不是加分項):
- 不開鏡頭,從錄製檔重播,UI 行為與現場一致;展示與功能凍結後的備份都靠它。

未來介面(草案,實作時可調):
- Session(config) -> 控制器
- Session.start(duration_s=60) / Session.stop()
- Session.record_to(path) / Session.load_replay(path)
- Session.tick() -> 驅動 capture → signals → ui 一輪
- 事件:on_change_point(ChangePoint)、on_finished(summary)

紅線:不輸出情緒標籤;執行期零網路;replay 必須可用。
"""
