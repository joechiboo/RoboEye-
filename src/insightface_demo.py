"""
RoboEye — 即時年齡 / 性別預測 Demo（方法 3: InsightFace buffalo_l）

使用方式：
    python src/insightface_demo.py

快捷鍵：
    q — 離開
"""

import cv2
from insightface.app import FaceAnalysis


def main():
    # 載入 InsightFace 模型（buffalo_l 準確度最高）
    print("[INFO] 載入 InsightFace 模型中（首次執行會自動下載）...")
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[INFO] 模型載入完成")

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

        # InsightFace 一步完成：人臉偵測 + 年齡 + 性別
        faces = app.get(frame)

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box

            age = face.age
            gender = "Male" if face.gender == 1 else "Female"

            label = f"{gender}, {age} yrs"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("RoboEye - Age & Gender (InsightFace)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
