# 👁️ RoboEye

> CNN 應用於機器人視覺之研究與實作 — 期末報告專案

## 📌 專案簡介

本專案為機器人視覺課程期末報告，聚焦於 **CNN（卷積神經網路）在機器人視覺領域的應用**，涵蓋論文研讀、程式實作與簡報製作。

## 📁 目錄結構

```
RoboEye/
├── papers/          # 論文 PDF 與閱讀筆記
├── src/             # 程式實作
├── slides/          # 簡報檔案
├── assets/          # 圖片、圖表素材
└── README.md
```

## 🎯 研究方向

**CNN 即時年齡預測** — 透過 webcam 即時辨識人臉並預測年齡（附加：性別 / 情緒辨識）

### 核心論文
1. **DEX** (ICCV 2015) — VGG-16 + softmax expected value，年齡預測經典方法
2. **REGA** (Review) — 即時情緒/性別/年齡三合一 CNN 架構
3. **Age & Gender with Transfer Learning** (arXiv 2021) — MobileNet/EfficientNet 輕量方案

## 🗓️ 時程規劃

| 階段 | 時間 | 內容 |
| ---- | ---- | ---- |
| 論文研讀 | 4月上旬 ~ 4月中旬 | 精讀 3 篇核心論文 |
| 架構整理 | 4月中旬 ~ 4月下旬 | 確定簡報架構、撰寫筆記 |
| 程式實作 | 5月上旬 ~ 5月中旬 | 即時年齡預測 Demo 開發 |
| 簡報製作 | 5月中旬 ~ 5月下旬 | 製作 4 分鐘簡報 + 4 分鐘 Demo |
| 演練修正 | 6月初 | 試講、調整 |

## 🛠️ 技術棧

- Python / PyTorch (or Keras)
- OpenCV（人臉偵測 + webcam 串流）
- VGG-16 / WideResNet / MobileNet（年齡、性別、情緒模型）

## 📄 授權

本專案僅供學術研究與課程使用。
