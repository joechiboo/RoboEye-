"""
人臉偵測模組 — 使用 OpenCV Haar Cascade 或 DNN 偵測人臉
"""

import cv2
import numpy as np


def load_face_detector():
    """載入 OpenCV 預訓練的人臉偵測器 (DNN-based, 較準確)"""
    model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
    config_path = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net


def detect_faces(frame, net, confidence_threshold=0.5):
    """
    偵測影像中的人臉

    Args:
        frame: BGR 影像 (numpy array)
        net: OpenCV DNN 人臉偵測模型
        confidence_threshold: 信心閾值

    Returns:
        faces: list of (x, y, w, h) bounding boxes
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces
