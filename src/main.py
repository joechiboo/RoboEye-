"""
RoboEye — 即時年齡 / 性別預測 Demo

使用方式：
    python src/main.py

快捷鍵：
    q — 離開
"""

import cv2
from detect_face import load_face_detector, detect_faces
from age_predictor import load_age_model, load_gender_model, predict_age, predict_gender


def main():
    # 載入模型
    print("[INFO] 載入模型中...")
    face_net = load_face_detector()
    age_net = load_age_model()
    gender_net = load_gender_model()
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

        # 偵測人臉
        faces = detect_faces(frame, face_net)

        for (x, y, w, h) in faces:
            # 確保座標在影像範圍內
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # 預測年齡與性別
            age = predict_age(face_img, age_net)
            gender = predict_gender(face_img, gender_net)

            # 繪製結果
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("RoboEye - Age & Gender Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
