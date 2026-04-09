"""
RoboEye CNN 模型 — MobileNetV2 + DEX Expected Value

架構:
    MobileNetV2 (pretrained ImageNet) → 共用 backbone
    ├── Age Head  : FC → 101 classes → softmax expected value (0~100 歲)
    └── Gender Head: FC → 2 classes
"""

import torch
import torch.nn as nn
import torchvision.models as models


class AgeGenderModel(nn.Module):
    """
    輕量年齡/性別預測模型

    年齡採用 DEX (Deep EXpectation) 方法:
    - 將 0~100 歲視為 101 類分類問題
    - softmax 輸出機率分布後，計算期望值 E[age] = Σ(i * p(i))
    - 可微分，訓練時用 cross-entropy，推論時取期望值
    """

    NUM_AGE_CLASSES = 101  # 0 ~ 100 歲

    def __init__(self, pretrained=True):
        super().__init__()

        # Backbone: MobileNetV2 (輕量、適合即時推論)
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # 取出 feature extractor (去掉最後的 classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV2 最後一層輸出 1280 channels
        in_features = 1280

        # Age head: 101 類 (DEX)
        self.age_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, self.NUM_AGE_CLASSES),
        )

        # Gender head: 2 類
        self.gender_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 2),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) 輸入影像

        Returns:
            age_logits:  (B, 101) 年齡分類 logits
            gender_logits: (B, 2) 性別分類 logits
        """
        feat = self.features(x)           # (B, 1280, 7, 7)
        feat = self.pool(feat)            # (B, 1280, 1, 1)
        feat = feat.flatten(1)            # (B, 1280)

        age_logits = self.age_head(feat)
        gender_logits = self.gender_head(feat)

        return age_logits, gender_logits

    @staticmethod
    def expected_age(age_logits):
        """
        DEX: 從 logits 計算期望年齡

        E[age] = Σ_{i=0}^{100} i * softmax(logits)_i
        """
        probs = torch.softmax(age_logits, dim=1)                   # (B, 101)
        ages = torch.arange(0, 101, dtype=torch.float32,
                            device=age_logits.device)               # (101,)
        expected = (probs * ages).sum(dim=1)                        # (B,)
        return expected

    @staticmethod
    def age_probs(age_logits):
        """回傳年齡機率分布 (用於視覺化)"""
        return torch.softmax(age_logits, dim=1)