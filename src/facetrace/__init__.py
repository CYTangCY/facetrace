"""FaceTrace — 本機執行的表情訊號日記原型。

對鏡頭講 60 秒,表情訊號變成一條時間軌跡。
描述性的訊號視覺化工具:只輸出訊號曲線、變化點、不確定性,
永遠不輸出情緒標籤。執行期零網路。

子模組(M0):
- capture  取像與 Face Landmarker 推論
- signals  四條訊號抽取(brow_down、mouth_smile、head_stability、mic_volume)
- ui       pyqtgraph 曲線視窗
- session  60 秒流程、錄製與 replay
"""
