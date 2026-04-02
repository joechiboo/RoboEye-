"""
下載預訓練模型

使用方式：
    python src/download_models.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODELS = {
    # 人臉偵測 (SSD + ResNet10)
    "deploy.prototxt":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel":
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",

    # 年齡預測
    "age_deploy.prototxt":
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
    "age_net.caffemodel":
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel",

    # 性別預測
    "gender_deploy.prototxt":
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt",
    "gender_net.caffemodel":
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel",
}


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in MODELS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            print(f"[SKIP] {filename} 已存在")
            continue

        print(f"[DOWN] 下載 {filename} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"[DONE] {filename}")
        except Exception as e:
            print(f"[FAIL] {filename}: {e}")


if __name__ == "__main__":
    download_models()
