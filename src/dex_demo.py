"""
RoboEye — 即時年齡 / 性別預測 Demo（方法 2: MobileNetV2 + DEX 自訓練）

使用方式：
    python src/dex_demo.py --checkpoint checkpoints/best_model.pth

快捷鍵：
    q — 離開
"""

import argparse

import cv2
import torch
from torchvision import transforms

from detect_face import load_face_detector, detect_faces
from model import AgeGenderModel

GENDER_LABELS = ["Male", "Female"]

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser(description="RoboEye DEX Demo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="模型 checkpoint 路徑")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    print(f"[INFO] 裝置: {device}")
    print("[INFO] 載入模型中...")
    face_net = load_face_detector()

    model = AgeGenderModel(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] 載入 checkpoint (Epoch {ckpt['epoch']}, MAE: {ckpt['val_mae']:.2f})")

    # 開啟 webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 無法開啟攝影機")
        return

    print("[INFO] 按 q 離開")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, face_net)
        for (x, y, w, h) in faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # BGR → RGB, 前處理
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            tensor = TRANSFORM(face_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                age_logits, gender_logits = model(tensor)
                age = model.expected_age(age_logits).item()
                gender = GENDER_LABELS[gender_logits.argmax(1).item()]
                confidence = torch.softmax(gender_logits, dim=1).max().item()

            label = f"{gender} ({confidence:.0%}), {age:.1f}y"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("RoboEye - Age & Gender (DEX)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
