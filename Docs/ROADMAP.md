# ROADMAP

## M0:Jessie Demo(截止 2026-08-21 中午凍結)

| 任務 | 內容 | 完成定義 |
|---|---|---|
| M0-T01 | repo 骨架 + 環境離線驗證 | 斷網狀態下 MediaPipe Face Landmarker 對一張測試圖跑出 blendshapes,requirements.txt 鎖版本,模型檔進 repo 的 models/ |
| M0-T02 | 取像 + 訊號抽取 | 鏡頭畫面疊 face mesh,終端即時印出四條訊號值與追蹤信心,30 FPS 以上 |
| M0-T03 | 即時曲線視窗 | pyqtgraph 視窗,四條曲線捲動,信心低於閾值該段變灰 |
| M0-T04 | 60 秒 session + 日記頁 | 倒數計時、session 結束跳日記頁:完整軌跡 + PELT 變化點標記(動態計算前降頻到 2 Hz、需通過隨機重排檢定、最多三個;沒有變化點就誠實顯示「沒有明顯變化」) |
| M0-T05 | 錄製與 replay 模式 | session 可存成本機檔案,啟動參數 --replay 可完整重播,畫面與 live 一致 |
| M0-T06 | 凍結打磨 | 連續跑 10 次不 crash,鏡頭拔掉有優雅錯誤訊息,替 8/23 錄一段乾淨的備份 session |

## M1:GroupFlow(MUM 2026,投稿截止 10-09 AoE)
8/23 之後回倫敦再規劃。方向:多手機感測、2-4 人、發言輪替與互動指標、facilitator dashboard。
進入 M1 時升級成完整版協作框架(補 ARCHITECTURE.md、DESIGN_RATIONALE.md、REQUIREMENTS.md)。

## 排隊區(想到但不做的東西寫這裡)
- 聲音 prosody 訊號(M1 再說)
- 向量記憶 / 感恩檢索(另一個專案,不進這個 repo)
- 中文 UI(M0 用英文,展示時口頭中文)
- M1 方法升級:變化點 PELT → KCP-RS(多訊號/多人)、群體同步 MdRQA 或 mv-SUSY、發言輪替均等度。路徑與引用見 Docs/references/TIMESERIES_METHOD.md 第 4 節。
