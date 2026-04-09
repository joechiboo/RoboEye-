"""
RoboEye 訓練腳本

使用方式:
    python src/train.py --data data/UTKFace --epochs 20 --batch-size 64

訓練完成後模型會儲存到 checkpoints/ 資料夾
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import AgeGenderModel
from dataset import UTKFaceDataset


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    age_mae_sum = 0
    gender_correct = 0
    n_samples = 0

    age_criterion = nn.CrossEntropyLoss()
    gender_criterion = nn.CrossEntropyLoss()

    for imgs, ages, genders in loader:
        imgs = imgs.to(device)
        ages = ages.to(device)
        genders = genders.to(device)

        age_logits, gender_logits = model(imgs)

        # DEX: 年齡用 cross-entropy (101 類分類)
        age_loss = age_criterion(age_logits, ages)
        gender_loss = gender_criterion(gender_logits, genders)
        loss = age_loss + gender_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 統計
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        # 年齡 MAE (用 expected value)
        with torch.no_grad():
            pred_ages = model.expected_age(age_logits)
            age_mae_sum += (pred_ages - ages.float()).abs().sum().item()
            gender_correct += (gender_logits.argmax(1) == genders).sum().item()

    return {
        "loss": total_loss / n_samples,
        "age_mae": age_mae_sum / n_samples,
        "gender_acc": gender_correct / n_samples,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    age_mae_sum = 0
    gender_correct = 0
    n_samples = 0

    age_criterion = nn.CrossEntropyLoss()
    gender_criterion = nn.CrossEntropyLoss()

    for imgs, ages, genders in loader:
        imgs = imgs.to(device)
        ages = ages.to(device)
        genders = genders.to(device)

        age_logits, gender_logits = model(imgs)

        age_loss = age_criterion(age_logits, ages)
        gender_loss = gender_criterion(gender_logits, genders)
        loss = age_loss + gender_loss

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        pred_ages = model.expected_age(age_logits)
        age_mae_sum += (pred_ages - ages.float()).abs().sum().item()
        gender_correct += (gender_logits.argmax(1) == genders).sum().item()

    return {
        "loss": total_loss / n_samples,
        "age_mae": age_mae_sum / n_samples,
        "gender_acc": gender_correct / n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="RoboEye 訓練")
    parser.add_argument("--data", type=str, required=True,
                        help="UTKFace 資料夾路徑")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / cpu / auto")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (Windows 建議 0)")
    args = parser.parse_args()

    # 裝置
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] 使用裝置: {device}")

    # 資料集
    train_ds = UTKFaceDataset(args.data, split="train")
    val_ds = UTKFaceDataset(args.data, split="val")
    print(f"[INFO] 訓練集: {len(train_ds)} 張, 驗證集: {len(val_ds)} 張")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # 模型
    model = AgeGenderModel(pretrained=True).to(device)

    # 優化器: backbone 用較小學習率, head 用較大學習率
    backbone_params = model.features.parameters()
    head_params = list(model.age_head.parameters()) + list(model.gender_head.parameters())

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # 訓練
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}  MAE: {train_metrics['age_mae']:.2f}  "
            f"Gender: {train_metrics['gender_acc']:.1%} | "
            f"Val Loss: {val_metrics['loss']:.4f}  MAE: {val_metrics['age_mae']:.2f}  "
            f"Gender: {val_metrics['gender_acc']:.1%}"
        )

        # 儲存最佳模型
        if val_metrics["age_mae"] < best_mae:
            best_mae = val_metrics["age_mae"]
            path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": best_mae,
                "val_gender_acc": val_metrics["gender_acc"],
            }, path)
            print(f"  → 儲存最佳模型 (MAE: {best_mae:.2f})")

    print(f"\n[完成] 最佳驗證 MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()
