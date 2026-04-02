# 論文筆記

## 研究主題：CNN 即時年齡預測（+ 情緒 / 性別辨識）

---

## 論文 1（經典核心）

### DEX: Deep EXpectation of Apparent Age from a Single Image

- **作者**：Rasmus Rothe, Radu Timofte, Luc Van Gool（ETH Zurich）
- **發表**：ICCV 2015 Workshop → IJCV 2018
- **論文連結**：https://openaccess.thecvf.com/content_iccv_2015_workshops/w11/papers/Rothe_DEX_Deep_EXpectation_ICCV_2015_paper.pdf
- **GitHub 實作**：https://github.com/basavaraj-hampiholi/Age-Estimation--DEX-in-Pytorch

#### 核心概念

- 使用 **VGG-16** 架構，預訓練於 ImageNet
- 將年齡預測轉為 **分類問題**（101 個類別，0-100 歲），再用 softmax expected value 做迴歸
- 比直接 regression 更準確

#### 資料集

- **IMDB-WIKI**：50 萬張名人臉部照片（最大公開年齡資料集）

#### 成果

- ChaLearn LAP 2015 年齡預測競賽 **冠軍**（115 隊參賽）
- **顯著超越人類基準**

#### 與本專案的關聯

- 年齡預測的理論基礎，簡報必講
- 「分類 → 期望值」的技巧是核心亮點

---

## 論文 2（三合一架構，最接近 Demo）

### REGA: Real-Time Emotion, Gender, Age Detection Using CNN — A Review

- **論文連結**：https://www.researchgate.net/publication/348637154_REGA_Real-Time_Emotion_Gender_Age_Detection_Using_CNN-A_Review
- **GitHub 實作**：https://github.com/oarriaga/face_classification

#### 核心概念

- 同時進行 **情緒 + 性別 + 年齡** 即時辨識
- 架構：WideResNet（年齡/性別）+ 傳統 CNN（情緒）
- 搭配 OpenCV 進行即時 webcam 推論

#### 資料集

- **IMDB-WIKI**：年齡 + 性別（95% accuracy）
- **FER2013**：7 類表情情緒（~66% accuracy）

#### 與本專案的關聯

- 最接近我們 Demo 的架構設計
- 可直接參考其 real-time pipeline

---

## 論文 3（實作導向，Transfer Learning）

### Age and Gender Prediction using Deep CNNs and Transfer Learning

- **發表**：arXiv 2021
- **論文連結**：https://arxiv.org/pdf/2110.12633

#### 核心概念

- 比較三種架構：**UNet、MobileNet、EfficientNet**
- 使用 Transfer Learning 預訓練模型做微調
- 教學性強，適合簡報解說

#### 資料集

- **UTKFace**：2 萬張照片，年齡 0-116 歲，含年齡/性別/種族標註

#### 與本專案的關聯

- Transfer Learning 概念簡報容易講解
- MobileNet 輕量適合即時推論

---

## 資料集總整理

| 資料集 | 大小 | 標註 | 用途 |
|--------|------|------|------|
| IMDB-WIKI | 50 萬張 | 年齡、性別 | 年齡/性別訓練（最大公開資料集）|
| UTKFace | 2 萬張 | 年齡、性別、種族 | 年齡/性別訓練（輕量替代）|
| FER2013 | 3.5 萬張 | 7 類表情 | 情緒辨識 |
