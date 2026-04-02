# 簡報大綱 — RoboEye: CNN 即時年齡預測

> 簡報 4 分鐘 + Demo 4 分鐘

---

## 簡報部分（4 分鐘）

### Slide 1 — 封面（10s）
- 題目：RoboEye — CNN 即時年齡預測
- 姓名、課程

### Slide 2 — 動機與問題（30s）
- 電腦能不能「看懂」一個人的年齡？
- 應用場景：智慧零售、安防監控、人機互動

### Slide 3 — CNN 基本原理（1min）
- Convolution → Pooling → Fully Connected 圖解
- 為什麼 CNN 適合影像任務？局部特徵擷取

### Slide 4 — 年齡預測方法（1min）
- DEX 方法：分類 → softmax expected value（而非直接回歸）
- 為什麼這樣更準？避免離群值影響
- 資料集：IMDB-WIKI（50 萬張）

### Slide 5 — 系統架構（50s）
- Pipeline 圖：Webcam → 人臉偵測 → CNN 推論 → 顯示結果
- 使用的模型：SSD（人臉偵測）+ Age/Gender CNN
- 技術棧：Python + OpenCV

### Slide 6 — 結論與未來展望（30s）
- 成果摘要
- 限制：光線、角度、遮擋
- 未來：加入情緒辨識、提升精度

---

## Demo 部分（4 分鐘）

### Phase 1 — 基本展示（1min）
- 開啟 webcam，展示自己的年齡/性別預測
- 說明畫面上的資訊

### Phase 2 — 觀眾互動（2min）
- 邀請 2-3 位觀眾到鏡頭前
- 讓 AI 猜年齡，製造趣味性
- 多人同框測試

### Phase 3 — 挑戰測試（1min）
- 戴口罩 / 戴眼鏡 / 遮半臉
- 不同角度 / 不同距離
- 展示模型的限制與邊界情況
