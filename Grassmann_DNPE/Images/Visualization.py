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
from PIL import Image
################################################################################
os.makedirs("Figure", exist_ok=True)
name_list=["Traffic", "CASIA-B", "Ballet", "ETH-80", "EYB", "UCF-S", "UT", "RGBD"]
for name in name_list:
    img = Image.new("RGB", (3000, 500), color=(255, 255, 255))
    data_path = name
    file_list = os.listdir(data_path)
    for i, file in enumerate(file_list):
        file_path = os.path.join(data_path, file)
        image = Image.open(file_path)
        image = image.resize((500, 500))
        img.paste(image, (i*500, 0))
    img.save("./Figure/" + name + ".jpg")
