"""
UTKFace 資料集載入器

UTKFace 檔名格式: [age]_[gender]_[race]_[date&time].jpg
    - age: 0 ~ 116
    - gender: 0 (male), 1 (female)

下載: https://susanqq.github.io/UTKFace/
解壓後放到 data/UTKFace/ 即可
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UTKFaceDataset(Dataset):
    """
    UTKFace 資料集

    Args:
        root: UTKFace 圖片資料夾路徑
        split: "train" 或 "val" (依 80/20 切分)
        transform: 圖片轉換 (若為 None 則使用預設)
        max_age: 年齡上限 (超過此值的樣本會被 clip)
    """

    def __init__(self, root, split="train", transform=None, max_age=100):
        self.root = Path(root)
        self.max_age = max_age
        self.transform = transform or self._default_transform(split)

        # 解析所有合法的圖片檔名
        self.samples = []
        for fname in sorted(os.listdir(self.root)):
            parts = fname.split("_")
            if len(parts) < 3:
                continue
            try:
                age = min(int(parts[0]), self.max_age)
                gender = int(parts[1])
                if gender not in (0, 1):
                    continue
                self.samples.append((fname, age, gender))
            except ValueError:
                continue

        # 按固定順序切分 train / val (80/20)
        split_idx = int(len(self.samples) * 0.8)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    @staticmethod
    def _default_transform(split):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, age, gender = self.samples[idx]
        img = Image.open(self.root / fname).convert("RGB")
        img = self.transform(img)
        return img, age, gender
