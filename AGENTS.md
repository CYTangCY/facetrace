# AGENTS.md

FaceTrace 工程師 AI 的短版規則。完整版見 `CLAUDE.md`;規劃前必讀 `Docs/CONFIRMED_PROJECT_CONTEXT.md`、`Docs/ROADMAP.md`、`Docs/CURRENT_TASK.md`。

## 專案紅線(違反即打掉重做)
1. **永遠不輸出情緒標籤。** 只輸出訊號曲線、變化點、不確定性。程式碼、UI、註解、輸出裡都不准出現 anger、sadness、happiness 這類詞當輸出名。MediaPipe 原生 blendshape 名稱(如 `mouthSmileLeft`)可以印。
2. **執行期零網路呼叫。** 模型檔在安裝時下載,執行時斷網必須完整運作。
3. **低信心變灰,不內插。** 追蹤信心低或臉離開畫面時,曲線變灰,不准內插硬猜。
4. **Replay 模式是必要功能**,不是加分項。
5. **M0 只做四條訊號**:brow_down、mouth_smile、head_stability、mic_volume(可選)。不加第五條。不訓練任何模型,MediaPipe Face Landmarker 之外不引入任何模型。

## Run log 規則
- 每次任務寫一份 run log:`Docs/codex_runs/<任務ID>_<序號>_<名稱>.md`(例:`Docs/codex_runs/M0-T01_001_env_setup.md`)。
- 內容:做了什麼、改了哪些檔案、實際安裝/使用的版本、遇到的問題與解法。
- **必須貼真實執行輸出**(複製貼上,不准改寫、不准摘要成「已通過」)。沒有真跑過的測試不准寫成通過。
- 審查時會核對:有沒有超出 CURRENT_TASK 範圍、有沒有違反紅線、run log 的結果是真跑的還是聲稱的。

## Git
AI 不碰任何 git 寫入操作(add / commit / push 一律不准)。任務完成時只輸出建議的 `git add` 清單,由使用者自己 commit 和 push。
