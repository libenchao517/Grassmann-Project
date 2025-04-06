################################################################################
## 可视化数据集
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
## 导入模块
import os
import cv2
import numpy as np
from PIL import Image
os.makedirs("Figure", exist_ok=True)
os.makedirs("Figure/GUMAP-Figure", exist_ok=True)
################################################################################
## 视频转图片函数
# def convert_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     video_path = video_path[:-4]
#     i=1
#     while cap.isOpened():
#         flag, frame = cap.read()
#         if flag:
#             image = Image.fromarray(frame)
#             image_path = video_path + str(i) + ".jpg"
#             image.save(image_path)
#             i+=1
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
# root = "GUMAP/Weizmann"
# video_list = [item for item in os.listdir(root) if item.endswith("avi")]
# for video in video_list:
#     video_path = os.path.join(root, video)
#     convert_video(video_path)
################################################################################
## 图片拼接
name_list=["ETH-80", "EYB", "UCF-Sport", "UT-Kinect", "Weizmann", "UTD-MHAD"]
for name in name_list:
    img = Image.new("RGB", (5000, 500), color=(255, 255, 255))
    data_path = os.path.join("GUMAP-Figure", name)
    file_list = os.listdir(data_path)
    for i, file in enumerate(file_list):
        file_path = os.path.join(data_path, file)
        image = Image.open(file_path)
        print(name, i, image.size)
        image = image.resize((500, 500))
        img.paste(image, (i*500, 0))
    img.save(os.path.join("Figure", "GUMAP-Figure", name + ".png"))
