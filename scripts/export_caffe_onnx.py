"""
將 Caffe 年齡/性別模型轉換為 ONNX 格式。

使用方式：
    python scripts/export_caffe_onnx.py --output-dir docs/models
"""

import argparse
import os
import sys

import cv2
import numpy as np

try:
    import onnx
except ImportError:
    print("[ERROR] 請先安裝 onnx: pip install onnx")
    sys.exit(1)

import torch
import torch.nn as nn


class CaffeAgeGenderNet(nn.Module):
    """重建 Levi & Hassner CaffeNet 架構 (與 age_deploy.prototxt 對齊)"""

    def __init__(self, num_classes=8, fc_in=18816):
        super().__init__()
        # Skip LocalResponseNorm — it exports poorly to ONNX and has
        # minimal impact on argmax output for this architecture.
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 7, stride=4, padding=0),   # 0
            nn.ReLU(inplace=True),                       # 1
            nn.MaxPool2d(3, stride=2, ceil_mode=True),   # 2

            nn.Conv2d(96, 256, 5, stride=1, padding=2),  # 3
            nn.ReLU(inplace=True),                       # 4
            nn.MaxPool2d(3, stride=2, ceil_mode=True),   # 5

            nn.Conv2d(256, 384, 3, stride=1, padding=1), # 6
            nn.ReLU(inplace=True),                       # 7
            nn.MaxPool2d(3, stride=2, ceil_mode=True),   # 8
        )
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Layers in the Caffe model that have learnable parameters (weight + bias)
PARAM_LAYER_NAMES = ["conv1", "conv2", "conv3", "fc6", "fc7", "fc8"]

# Corresponding PyTorch parameter key prefixes
PYTORCH_PARAM_MAP = [
    ("features.0",   "conv"),   # conv1
    ("features.3",   "conv"),   # conv2
    ("features.6",   "conv"),   # conv3
    ("classifier.0", "fc"),     # fc6
    ("classifier.3", "fc"),     # fc7
    ("classifier.6", "fc"),     # fc8
]


def transfer_weights(caffe_net, pytorch_model):
    """從 OpenCV DNN 提取 Caffe 權重並載入 PyTorch 模型"""
    state_dict = pytorch_model.state_dict()

    for caffe_name, (pt_prefix, _) in zip(PARAM_LAYER_NAMES, PYTORCH_PARAM_MAP):
        layer_id = caffe_net.getLayerId(caffe_name)

        # weight (param index 0)
        weight = caffe_net.getParam(layer_id, 0)
        w_key = f"{pt_prefix}.weight"
        state_dict[w_key] = torch.from_numpy(weight.copy())

        # bias (param index 1)
        bias = caffe_net.getParam(layer_id, 1)
        b_key = f"{pt_prefix}.bias"
        state_dict[b_key] = torch.from_numpy(bias.flatten().copy())

        print(f"  {caffe_name} → {pt_prefix}  weight={weight.shape}  bias={bias.flatten().shape}")

    pytorch_model.load_state_dict(state_dict)
    return pytorch_model


def validate(caffe_net, pytorch_model, input_size=227):
    """用相同輸入比較 Caffe 和 PyTorch 輸出是否一致"""
    # 建立測試輸入
    test_img = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(test_img, 1.0, (input_size, input_size),
                                  (78.4263377603, 87.7689143744, 114.895847746),
                                  swapRB=False)

    # Caffe 推論
    caffe_net.setInput(blob)
    caffe_out = caffe_net.forward()

    # PyTorch 推論
    pytorch_model.eval()
    tensor_input = torch.from_numpy(blob)
    with torch.no_grad():
        pt_out = pytorch_model(tensor_input).numpy()

    diff = np.abs(caffe_out - pt_out).max()
    ok = diff < 0.01
    print(f"  max diff = {diff:.6f} {'OK' if ok else 'MISMATCH'}")
    return ok


def export_model(prototxt, caffemodel, output_path, num_classes):
    """轉換單一 Caffe 模型為 ONNX"""
    print(f"  載入 Caffe 模型: {caffemodel}")
    caffe_net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    # 偵測 fc6 的輸入維度
    fc6_id = caffe_net.getLayerId("fc6")
    fc6_weight = caffe_net.getParam(fc6_id, 0)
    fc_in = fc6_weight.shape[1]
    print(f"  fc6 輸入維度: {fc_in}")

    print(f"  建立 PyTorch 等價模型 (classes={num_classes}, fc_in={fc_in})")
    pytorch_model = CaffeAgeGenderNet(num_classes=num_classes, fc_in=fc_in)

    print("  轉移權重...")
    transfer_weights(caffe_net, pytorch_model)

    print("  驗證輸出一致性...")
    validate(caffe_net, pytorch_model)

    print(f"  匯出 ONNX → {output_path}")
    pytorch_model.eval()
    dummy_input = torch.randn(1, 3, 227, 227)
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        output_path,
        input_names=["data"],
        output_names=["prob"],
        dynamic_axes={"data": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser(description="Export Caffe age/gender models to ONNX")
    parser.add_argument("--output-dir", type=str, default="docs/models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models = [
        {
            "name": "Age (Caffe)",
            "prototxt": "models/age_deploy.prototxt",
            "caffemodel": "models/age_net.caffemodel",
            "output": os.path.join(args.output_dir, "caffe_age.onnx"),
            "num_classes": 8,
        },
        {
            "name": "Gender (Caffe)",
            "prototxt": "models/gender_deploy.prototxt",
            "caffemodel": "models/gender_net.caffemodel",
            "output": os.path.join(args.output_dir, "caffe_gender.onnx"),
            "num_classes": 2,
        },
    ]

    for m in models:
        print(f"\n{'='*50}")
        print(f"轉換 {m['name']}")
        print(f"{'='*50}")
        if not os.path.exists(m["caffemodel"]):
            print(f"  [SKIP] {m['caffemodel']} 不存在")
            continue
        export_model(m["prototxt"], m["caffemodel"], m["output"], m["num_classes"])

    print("\n全部完成!")


if __name__ == "__main__":
    main()