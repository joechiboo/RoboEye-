"""
將 PyTorch AgeGenderModel 匯出為 ONNX 格式，供瀏覽器端 onnxruntime-web 使用。

使用方式：
    python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output docs/models/roboeye.onnx
"""

import argparse
import sys
sys.path.insert(0, "src")

import torch
from model import AgeGenderModel


def main():
    parser = argparse.ArgumentParser(description="Export AgeGenderModel to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="PyTorch checkpoint 路徑")
    parser.add_argument("--output", type=str, default="docs/models/roboeye.onnx",
                        help="ONNX 輸出路徑")
    args = parser.parse_args()

    # 載入模型
    model = AgeGenderModel(pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] 載入 checkpoint (Epoch {ckpt['epoch']}, MAE: {ckpt['val_mae']:.2f})")

    # 匯出 ONNX (使用舊版 exporter，權重嵌入單一檔案，方便瀏覽器載入)
    # 固定 batch=1 以避免 ORT Web 的 dynamic shape 推論問題
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy_input,),
        args.output,
        input_names=["input"],
        output_names=["age_logits", "gender_logits"],
        opset_version=13,
        dynamo=False,
    )
    print(f"[INFO] ONNX 模型已匯出至 {args.output}")


if __name__ == "__main__":
    main()
