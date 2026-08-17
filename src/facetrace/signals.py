"""signals — 四條訊號抽取(M0-T01 空殼,尚無實作)。

M0 只有這四條,不加第五條:
- brow_down       由 browDownLeft / browDownRight blendshape 合成(0..1)
- mouth_smile     由 mouthSmileLeft / mouthSmileRight blendshape 合成(0..1)
- head_stability  由 facial transformation matrix 的逐幀變化量推得(越穩定越高)
- mic_volume      麥克風音量 RMS(可選;沒有麥克風時整條缺席)

每條訊號的每個取樣點都要帶「confidence / valid」旗標:
- 追蹤信心低或臉離開畫面 → 該點標為 invalid,下游 UI 畫成灰色。
- 禁止內插填補 invalid 區段。

變化點(change point):對每條訊號做 rolling z-score,超過門檻的時間點標成
「第 N 秒有明顯變化」。只標時間與幅度,不解釋成任何情緒。

未來介面(草案,實作時可調):
- extract(frame_result) -> SignalSample(timestamp_ms, values: dict[str, float],
                                         valid: dict[str, bool])
- smooth(samples, window) -> samples(EMA 或移動平均,只作用於 valid 區段)
- change_points(samples, window, z_threshold) -> list[ChangePoint(timestamp_ms, signal, z)]

紅線:訊號名只准是上述四個 key;程式碼、註解、輸出裡不出現情緒詞當輸出名。
"""
