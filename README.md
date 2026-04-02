# 👁️ RoboEye

> CNN 應用於機器人視覺之研究與實作 — 期末報告專案

## 📌 專案簡介

本專案為機器人視覺課程期末報告，聚焦於 **CNN（卷積神經網路）在機器人視覺領域的應用**，涵蓋論文研讀、程式實作與簡報製作。

## 📁 目錄結構

```
RoboEye/
├── papers/              # 論文 PDF 與閱讀筆記
├── src/                 # Python Demo（三種方案）
│   ├── caffe_demo.py        # 方法 1: Caffe 預訓練
│   ├── dex_demo.py          # 方法 2: MobileNetV2 + DEX（需訓練）
│   ├── insightface_demo.py  # 方法 3: InsightFace buffalo_l
│   ├── model.py             # 方法 2 模型定義
│   ├── train.py             # 方法 2 訓練腳本
│   ├── dataset.py           # UTKFace 資料集載入器
│   ├── detect_face.py       # 人臉偵測模組
│   ├── age_predictor.py     # 方法 1 年齡/性別預測模組
│   └── download_models.py   # Caffe 模型下載腳本
├── scripts/             # 匯出工具
│   ├── export_onnx.py       # 方法 2 PyTorch → ONNX
│   └── export_caffe_onnx.py # 方法 1 Caffe → ONNX
├── docs/                # Web Demo（瀏覽器端推論）
│   ├── index.html
│   ├── js/app.js
│   └── models/              # ONNX 模型檔
├── slides/              # 簡報檔案
├── models/              # Python Demo 用的模型檔
└── README.md
```

## 🎯 研究方向

**CNN 即時年齡預測 — 三種方法比較研究**

| 方案 | 方法 | 模型 | 年齡輸出 | 需訓練 | 狀態 |
|------|------|------|----------|--------|------|
| 1 | Caffe 預訓練 (2015) | Levi & Hassner CaffeNet | 8 個年齡區間 | 否 | ✅ 可跑 |
| 2 | MobileNetV2 + DEX | 自訓練 on UTKFace | 精確數字 (0-100) | 是 | ⏳ 待訓練 |
| 3 | InsightFace buffalo_l | 大型預訓練模型 | 精確數字 | 否 | ✅ 可跑 |

### 核心論文
1. **DEX** (ICCV 2015) — VGG-16 + softmax expected value，年齡預測經典方法 → 對應方案 2
2. **REGA** (Review) — 即時情緒/性別/年齡三合一 CNN 架構 → 對應方案 1
3. **Age & Gender with Transfer Learning** (arXiv 2021) — MobileNet/EfficientNet 輕量方案 → 對應方案 3

## 🗓️ 時程規劃

| 階段 | 時間 | 內容 |
| ---- | ---- | ---- |
| 論文研讀 | 4月上旬 ~ 4月中旬 | 精讀 3 篇核心論文 |
| 架構整理 | 4月中旬 ~ 4月下旬 | 確定簡報架構、撰寫筆記 |
| 程式實作 | 5月上旬 ~ 5月中旬 | 即時年齡預測 Demo 開發 |
| 簡報製作 | 5月中旬 ~ 5月下旬 | 製作 4 分鐘簡報 + 4 分鐘 Demo |
| 演練修正 | 6月初 | 試講、調整 |

## 🌐 Web Demo (GitHub Pages)

瀏覽器端即時推論，不需後端。

### 部署步驟

```bash
# 1. 匯出 MobileNetV2 模型為 ONNX
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output docs/models/roboeye.onnx

# 2. (選用) 匯出 Caffe 模型為 ONNX
python scripts/export_caffe_onnx.py --output-dir docs/models

# 3. Push 到 GitHub，然後到 Settings → Pages → Source 選 "main branch /docs folder"
```

### 本地測試

```bash
# 需要 HTTPS 才能使用 webcam (getUserMedia)
cd docs
python -m http.server 8080
# 打開 http://localhost:8080 (Chrome 允許 localhost 使用 webcam)
```

## 🛠️ 技術棧

- Python / PyTorch (or Keras)
- OpenCV（人臉偵測 + webcam 串流）
- VGG-16 / WideResNet / MobileNet（年齡、性別、情緒模型）
- ONNX Runtime Web（瀏覽器端 CNN 推論）
- face-api.js（瀏覽器端人臉偵測）

## 📄 授權

本專案僅供學術研究與課程使用。
