# READING_LIST 文獻清單(帶註解)
**優先序照讀。粗體是必讀,其他掃過知道在講什麼即可。**

## 科學基礎

1. **Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements. Psychological Science in the Public Interest, 20(1), 1-68.**
   一句話:表情到離散情緒的推論在信度、特異度、概化三個層面不成立。本專案「不貼標籤」的科學根據,也是作者的武器,精讀。

2. Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System. Consulting Psychologists Press.
   一句話:FACS 是描述性的肌肉動作編碼系統;它的描述層是可守的,基本情緒理論那一層才是被 Barrett 批判的。知道這個區分即可。

3. Ramseyer, F., & Tschacher, W. (2011). Nonverbal synchrony in psychotherapy: Coordinated body movement reflects relationship quality and outcome. Journal of Consulting and Clinical Psychology, 79(3), 284-295.
   一句話:心理治療歷程研究用非語言訊號做雙人動態分析的代表作;未來學界合作的接口。

## SOTA 地圖(知道三波演進即可)

4. Rethinking Facial Expression Recognition in the Era of Multimodal Large Language Models. arXiv:2511.00389 (2025).
   一句話:MLLM 化這一波的基準論文,讀 intro 與結論。

5. Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning. NeurIPS 2024.
   一句話:第三波代表作,掃架構圖,理解為什麼全是伺服器級。

6. MPA-FER (ICCV 2025)。CLIP 化那一波的例子,掃摘要。

7. LibreFace (WACV 2024)。開源 AU 與表情工具箱,ONNX 輕量;工程參考。

8. MediaPipe Face Landmarker 官方文件(blendshapes 章節)。
   一句話:M0 的訊號來源,52 個係數的定義,動手前必讀。

## 法規

9. **EU AI Act Article 5(1)(f) 原文 + Commission Guidelines C(2025) 884 相關段落。**
   一句話:職場教育情緒推斷禁令與醫療例外;細節見 REGULATORY_MAP.md。

## 臨床應用線(M1 之後才需要深讀)

10. DAIC-WOZ 資料集與 AVEC 挑戰賽任一總覽論文。
    一句話:臨床訪談多模態分析的評測傳統,了解臨床線怎麼做 evaluation。

## 內部文件互相參照

- 設計決策與依據的對應:Docs/references/DESIGN_EVIDENCE.md
- 法規細節:Docs/references/REGULATORY_MAP.md
- 更早的完整領域整理(含任務分類表、demo 決策矩陣):見專案外的 fer_review_v1.md,已整併進上述兩份的部分不再重複維護。
