"""capture — 取像與 Face Landmarker 推論(M0-T01 空殼,尚無實作)。

職責:
- 用 OpenCV 開本機鏡頭,逐幀取像(BGR ndarray)。
- 用 MediaPipe Face Landmarker(models/face_landmarker.task,執行期零網路)
  對每一幀推論,取得:
    * face landmarks(468/478 點,含虹膜)
    * blendshapes(52 個 MediaPipe 原生係數,名稱照原樣傳遞)
    * facial transformation matrix(頭部姿態)
- 每一幀附上時間戳與「是否偵測到臉」旗標;沒臉時回傳空結果,不內插、不硬猜。

未來介面(草案,實作時可調):
- open_camera(index: int = 0) -> 鏡頭句柄
- load_landmarker(model_path) -> Face Landmarker 實例(VIDEO 或 LIVE_STREAM 模式)
- next_frame(camera, landmarker) -> FrameResult(timestamp_ms, frame_bgr, has_face,
                                                 blendshapes, landmarks, head_pose)
- release(camera, landmarker)

紅線:不輸出情緒標籤;不呼叫網路;不引入 MediaPipe 以外的模型。
"""
