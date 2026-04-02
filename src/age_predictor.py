"""
年齡預測模組 — 使用預訓練 CNN 模型預測年齡
"""

import cv2
import numpy as np


# OpenCV DNN 預訓練年齡模型的年齡分組
AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]


def load_age_model():
    """載入預訓練的年齡預測模型"""
    model_path = "models/age_net.caffemodel"
    config_path = "models/age_deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net


def load_gender_model():
    """載入預訓練的性別預測模型"""
    model_path = "models/gender_net.caffemodel"
    config_path = "models/gender_deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net


GENDER_LIST = ["Male", "Female"]


def predict_age(face_img, age_net):
    """
    預測人臉年齡

    Args:
        face_img: 裁切後的人臉影像 (BGR)
        age_net: 年齡預測模型

    Returns:
        age_label: 預測的年齡區間字串
    """
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                  (78.4263377603, 87.7689143744, 114.895847746),
                                  swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    age_label = AGE_BUCKETS[preds[0].argmax()]
    return age_label


def predict_gender(face_img, gender_net):
    """
    預測人臉性別

    Args:
        face_img: 裁切後的人臉影像 (BGR)
        gender_net: 性別預測模型

    Returns:
        gender_label: "Male" 或 "Female"
    """
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                  (78.4263377603, 87.7689143744, 114.895847746),
                                  swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    gender_label = GENDER_LIST[preds[0].argmax()]
    return gender_label
