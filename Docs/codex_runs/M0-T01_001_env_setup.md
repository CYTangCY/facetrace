# M0-T01_001 — repo 骨架 + 環境離線驗證

- 日期:2026-08-17 深夜 → 2026-08-18 00:03(本機時間)
- 機器:Windows 11 Home 10.0.26200,系統 Python 3.13.7(`C:\Python313\python.exe`)
- 任務文件:`Docs/CURRENT_TASK.md`(M0-T01)
- 工程師 AI:Claude(Fable 5)。AI 未執行任何 git 寫入操作。

## 1. 做了什麼

| # | 項目 | 結果 |
|---|------|------|
| 1 | 骨架:`src/facetrace/{__init__,capture,signals,ui,session}.py`(只有 module docstring,無實作、無第三方 import)| 完成;用**系統 Python(未裝任何第三方套件)**import 全部成功,證明零依賴 |
| 2 | `Docs/codex_runs/.gitkeep`、`Docs/completed_tasks/.gitkeep` | 完成 |
| 3 | `AGENTS.md`(紅線五條 + run log 規則 + git 規則短版)| 完成 |
| 4 | 環境:專案內 `.venv`(Python 3.13.7),安裝五個套件並鎖版 → `requirements.txt` | 完成;`pip check` 無衝突 |
| 5 | `scripts/download_model.py` 一次性下載模型 → `models/face_landmarker.task` | 完成;3,758,596 bytes(3.58 MiB),md5 `b0e7274907a1644404fef66b28dd6d85` |
| 6 | `.gitignore`(`__pycache__/`、`.venv/` 等);`models/*.task` **不在**排除清單 | 完成 |
| 7 | `scripts/make_test_face.py` → `scripts/test_face.jpg`(OpenCV 純程式生成,640x800,30,111 bytes)| 完成;**第一次嘗試就被 Face Landmarker 偵測到 1 張臉**,不需改用真人照片,無授權問題 |
| 8 | `scripts/verify_env.py`(載模型 → 跑測試圖 → 印臉數 / top-10 blendshapes / 20 次推論 FPS;內建 socket 守衛;失敗即非零退出)| 完成;主路徑 exit=0,兩條失敗路徑 exit=1(輸出見下)|
| 9 | 本 run log | 完成 |

### 沒做的(依任務規定)
- 沒寫任何 UI 程式、沒實作任何訊號邏輯。
- 沒做 git add / commit / push。

## 2. 實際安裝的版本(`pip freeze` 摘出五個目標套件,已寫入 `requirements.txt`)

```
mediapipe==1.0.1
numpy==2.5.2
opencv-python==5.0.0.93
pyqtgraph==0.14.0
PySide6==6.11.1
```

`pip index versions mediapipe`(Python 3.13):`1.0.1, 1.0.0, 0.10.35, 0.10.33, 0.10.32, 0.10.31, 0.10.30`。最新版 1.0.1 在此環境**一次裝成**,不需降版。

完整安裝結果(`pip install` 最後一行,原文):
```
Successfully installed PySide6-6.11.1 PySide6_Addons-6.11.1 PySide6_Essentials-6.11.1 absl-py-2.5.0 certifi-2026.7.22 cffi-2.1.1 colorama-0.4.6 contourpy-1.3.3 cycler-0.12.1 flatbuffers-25.12.19 fonttools-4.63.0 kiwisolver-1.5.0 matplotlib-3.11.1 mediapipe-1.0.1 numpy-2.5.2 opencv-contrib-python-5.0.0.93 opencv-python-5.0.0.93 packaging-26.3 pillow-12.3.0 pycparser-3.0 pyparsing-3.3.2 pyqtgraph-0.14.0 python-dateutil-2.9.0.post0 shiboken6-6.11.1 six-1.17.0 sounddevice-0.5.6
```

`pip check`(原文):
```
No broken requirements found.
```

`pip show mediapipe`(原文節錄):
```
Version: 1.0.1
Requires: absl-py, certifi, flatbuffers, matplotlib, numpy, opencv-contrib-python, sounddevice
```

## 3. 模型檔

`python scripts/download_model.py`(原文):
```
[download] https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
[to]       D:\Code\HY_DEMO\models\face_landmarker.task
[ok] saved 3758596 bytes (3.58 MiB) -> D:\Code\HY_DEMO\models\face_landmarker.task
exit=0
```
`ls -l models/`:`-rw-r--r-- 1 fcxsw 197609 3758596 Aug 18 00:00 face_landmarker.task`

這是整個 repo **唯一**會碰網路的腳本,而且只在安裝期跑一次。模型檔要進 repo,clone 下來就有。

## 4. 測試圖來源

`scripts/test_face.jpg` 由 `scripts/make_test_face.py` 用 OpenCV + numpy 基本圖元(橢圓、多邊形、漸層遮罩、高斯模糊)畫出,無外部素材、無授權問題、可重現。第一次偵測測試(原文):
```
faces: 1
  eyeWideRight             0.4745
  browOuterUpLeft          0.4440
  eyeWideLeft              0.4418
  browInnerUp              0.3493
  browOuterUpRight         0.3291
```
因為偵測成功,**沒有**改用真人照片。

## 5. `python scripts/verify_env.py` 真實完整輸出(複製貼上,未改寫)

執行方式:`.venv\Scripts\python.exe scripts\verify_env.py`(repo 根目錄)

```
[guard] outbound network sockets disabled for this process
[env]   python     3.13.7  (D:\Code\HY_DEMO\.venv\Scripts\python.exe)
[env]   mediapipe  1.0.1
[env]   opencv     5.0.0
[env]   numpy      2.5.2
[model] D:\Code\HY_DEMO\models\face_landmarker.task  (3758596 bytes)
[image] D:\Code\HY_DEMO\scripts\test_face.jpg  (640x800)
WARNING: Logging before InitGoogle() is written to STDERR
W0000 00:00:1787007848.681587   53104 face_landmarker_graph.cc:180] Sets FaceBlendshapesGraph acceleration to xnnpack by default.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
W0000 00:00:1787007848.686825   33032 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1787007848.697044   42976 inference_feedback_manager.cc:121] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
[detect] faces detected: 1
[detect] landmarks per face: 478   blendshapes: 52   transformation matrix: yes
[blendshapes] top 10 by score:
   1. eyeWideRight             0.4745
   2. browOuterUpLeft          0.4440
   3. eyeWideLeft              0.4418
   4. browInnerUp              0.3493
   5. browOuterUpRight         0.3291
   6. eyeLookDownRight         0.1763
   7. eyeLookDownLeft          0.1667
   8. eyeLookInRight           0.0985
   9. eyeLookOutLeft           0.0914
  10. mouthPressLeft           0.0829
[fps] 20 inferences in 184.6 ms  -> avg 9.23 ms/frame  ~ 108.4 FPS (single image, CPU, IMAGE mode)
[OK] environment verified offline: model loaded, face detected, blendshapes produced
exit=0
```

(`WARNING/W0000/INFO` 那幾行是 MediaPipe/TFLite 原生 C++ 的 stderr 訊息,無害;`GLOG_minloglevel` 環境變數壓不掉它們,保留原樣。)

### 5a. 失敗路徑測試(在 scratchpad 複製一份跑,未動 repo 檔案;已濾掉 W0000 雜訊行)

case A — 模型不存在:
```
[FAIL] model not found: C:\Users\fcxsw\AppData\Local\Temp\claude\D--Code-HY-DEMO\e5ed44d6-f98c-46d4-827f-2987123b57b2\scratchpad\failtest\models\face_landmarker.task
       expected models/face_landmarker.task in the repo. If it is missing, run scripts/download_model.py once (needs network).
[guard] outbound network sockets disabled for this process
[env]   python     3.13.7  (D:\Code\HY_DEMO\.venv\Scripts\python.exe)
[env]   mediapipe  1.0.1
[env]   opencv     5.0.0
[env]   numpy      2.5.2
exit=1
```

case B — 偵測不到臉(400x400 純灰圖):
```
[FAIL] no face detected in test image � environment NOT verified
[guard] outbound network sockets disabled for this process
[env]   python     3.13.7  (D:\Code\HY_DEMO\.venv\Scripts\python.exe)
[env]   mediapipe  1.0.1
[env]   opencv     5.0.0
[env]   numpy      2.5.2
[model] C:\Users\fcxsw\...\scratchpad\failtest\models\face_landmarker.task  (3758596 bytes)
[image] C:\Users\fcxsw\...\scratchpad\failtest\scripts\test_face.jpg  (400x400)
[detect] faces detected: 0
exit=1
```
(當時 FAIL 訊息裡有一個 em-dash,在 Windows 管線下顯示成 `�`;之後已改成純 ASCII `-`。stderr 出現在 stdout 之前是管線緩衝造成,之後已加 `sys.stdout.reconfigure(line_buffering=True)`,見第 5 節的最終輸出順序已正常。)

### 5b. socket 守衛自測(證明「零網路」是被強制的,不是聲稱的)

把 `verify_env.py` 當模組載入(守衛在 import 時安裝),再用 urllib 嘗試連 storage.googleapis.com(原文):
```
guard OK -> blocked with: _NetworkDisabled - verify_env.py: outbound network is disabled by design
```
守衛以 monkeypatch `socket.socket.connect / connect_ex / socket.create_connection / socket.getaddrinfo` 實作,在 import mediapipe 之前生效。在守衛開啟下 verify_env.py 主路徑仍 exit=0 → 模型載入與推論全程沒有任何網路呼叫。

## 6. 遇到的問題與解法

1. **系統 Python 是 3.13,不是文件寫的 3.11+ 下限值** — 3.13 ≥ 3.11 符合要求;mediapipe 1.0.1 有 cp313 wheel,直接裝成。用 `.venv` 隔離,不污染系統 Python。
2. **mediapipe 1.0.1 依賴 `opencv-contrib-python`,和任務指定的 `opencv-python` 同時被裝進 `.venv`**(兩者都提供 `cv2` 套件目錄,後裝的 contrib 覆蓋)。實測 `import cv2` 正常(5.0.0)、`pip check` 無衝突、Face Landmarker 正常。`requirements.txt` 照任務要求鎖 `opencv-python==5.0.0.93`;pip 會自動帶入同版 contrib。若之後遇到 cv2 怪問題,第一個嫌疑就是這個雙包共存,屆時可考慮只留 contrib。
3. **MediaPipe 1.0.1 API 與 0.10.x 系列相同**(`mediapipe.tasks.python.vision.FaceLandmarker`、`FaceLandmarkerOptions`、`RunningMode.IMAGE`、`output_face_blendshapes`),沒有踩到 breaking change。
4. **`GLOG_minloglevel` / `TF_CPP_MIN_LOG_LEVEL` 壓不掉原生 W0000 警告** — 無害,不處理。
5. **Windows 管線編碼(cp1252/cp950)** — 非 ASCII 字元在管線輸出會變 `�`,`verify_env.py` 已全改為 ASCII 輸出。骨架檔的中文 docstring 不受影響(只是我用一行測試 print 它時撞到,與模組無關)。
6. **使用者跑 `python scripts/verify_env.py` 時要用 `.venv`** — 直接打 `python` 會用系統 Python(沒裝套件)。先 `.venv\Scripts\activate`,或直接 `.venv\Scripts\python.exe scripts\verify_env.py`。

## 7. 待使用者人工驗證(完成定義)

- 關 Wi-Fi(斷網),在 repo 根目錄跑:
  ```
  .venv\Scripts\activate
  python scripts\verify_env.py
  ```
- 預期看到 `[detect] faces detected: 1`、十行 blendshape、`[fps] ...`、最後 `[OK] ...`,exit code 0。
