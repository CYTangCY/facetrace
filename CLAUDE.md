# CLAUDE.md

## 專案身份
FaceTrace 是一個本機執行的表情訊號日記原型。對鏡頭講 60 秒,表情訊號變成一條時間軌跡。
它是描述性的訊號視覺化工具,不是情緒辨識器,不是診斷工具,不是產品。
兩個用途:2026-08-23 向聊心茶室團隊展示;之後長成 MUM 2026 GroupFlow demo 的地基。

## 規劃前必讀
- Docs/CONFIRMED_PROJECT_CONTEXT.md
- Docs/ROADMAP.md
- Docs/CURRENT_TASK.md
- Docs/references/DESIGN_EVIDENCE.md(設計決策的理論與法規依據;本機限定,不在 GitHub 上)

## 專案紅線(技術)
- 系統永遠不輸出情緒標籤。只輸出訊號曲線、變化點、不確定性。程式碼、UI、註解裡都不准出現 anger、sadness、happiness 這類詞當輸出名。
- 執行期零網路呼叫。模型檔在安裝時下載,執行時斷網必須完整運作。
- 追蹤信心低或臉離開畫面時,曲線變灰,不准內插硬猜。
- Replay 模式是必要功能,不是加分項。
- M0 只做四條訊號:brow_down、mouth_smile、head_stability、mic_volume(可選)。不加第五條。
- 不訓練任何模型。MediaPipe Face Landmarker 之外不引入任何模型。

## 技術棧(已定,不重開討論)
Python 3.11+、OpenCV(取像與顯示)、MediaPipe Face Landmarker(blendshapes + 頭部姿態)、pyqtgraph(曲線)、numpy。原生視窗,不做網頁。

## Git 規則
指揮官給精確的 git add 清單,使用者自己 commit 和 push。AI 不碰 git 寫入操作。

## Token 效率
定向 grep 和分段讀取。新檔案和高風險點完整讀。

## 每次任務的規劃輸出格式
1. 任務解讀
2. 工程師要改的檔案
3. 工程師不准碰的檔案
4. 實作計畫
5. 人工測試清單
6. 風險
7. 給工程師 AI 的完整 prompt

## 審查清單
- 超出 CURRENT_TASK 範圍了嗎
- 違反紅線了嗎(尤其:有沒有偷偷出現情緒標籤、有沒有網路呼叫)
- run log 的測試結果是真跑的還是聲稱的
