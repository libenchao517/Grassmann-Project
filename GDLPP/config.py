################################################################################
# 本文件是GDLPP项目的配置文件
################################################################################
# 导入模块
import json
from pathlib import Path
import numpy as np
################################################################################
class GrassmannDLPP_config:
    """
    GDLPP项目的配置文件
    """
    def __init__(self):
        """
        初始化函数
        """
        self.data_name = [  # 可执行的数据集列表
            "ETH-80",
            "EYB",
            "FPHA",
            "RGB-D",
            "UCF-S",
            "UT-Kinect"
        ]
        self.self_data = []
        self.basic_data = [  # 用于参数实验的数据集列表
            "ETH-80",
            "EYB",
            "UCF-S",
            "UT-Kinect"
        ]
        self.GDLPP_data = [  # 比较GDLPP算法的数据集
            "ETH-80",
            "EYB",
            "FPHA",
            "RGB-D",
            "UCF-S",
            "UT-Kinect"
        ]
        self.none_data = []
        self.n_neighbors = 8 # 邻居数
        self.random_state = np.random.randint(2024) # 随机的随机种子
        self.converged_tol = 1.0   # 终止条件
        self.max_epoch = 2         # 最大迭代次数
        self.return_time = True    # 是否返回时间
        self.train_size = {        # 各数据集的训练集规模
            "ETH-80" : 1,
            "EYB" : 1,
            "FPHA" : 1,
            "RGB-D" : 1,
            "UCF-S" : 1,
            "UT-Kinect" : 1
        }
        self.is_discriminant = {  # 是否运用判别思想
            "GLPP" : False,
            "GDLPP" : True,
            "GFLPP" : True,
            "GSLPP" : True
        }
        self.is_semi_supervised = {  # 是否是半监督方法
            "GLPP" : False,
            "GDLPP" : False,
            "GFLPP" : False,
            "GSLPP" : True
        }
        self.is_self_supervised = { # 是否是自监督方法
            "GLPP" : False,
            "GDLPP" : False,
            "GFLPP" : True,
            "GSLPP" : False
            }
        self.remain_eta = {    # 各数据集的累积贡献率
            "ETH-80" : 0.95,
            "EYB" : 0.95,
            "FPHA" : 0.95,
            "RGB-D" : 0.95,
            "UCF-S" : 0.99,
            "UT-Kinect": 0.75
        }
        self.grassmann_p = {   # 各数据集的子空间阶数
            "ETH-80" : 15,
            "EYB" : 5,
            "FPHA" : 10,
            "RGB-D" : 10,
            "UCF-S" : 11,
            "UT-Kinect" : 14
        }
        self.alpha = {   # GSLPP方法的正则化系数
            "ETH-80" : 0.45,
            "EYB" : 0.50,
            "FPHA" : 0.50,
            "RGB-D" : 0.50,
            "UCF-S" : 0.65,
            "UT-Kinect" : 0.45,
        }
        self._load_paras()    # 加载数据集的参数
        self._check_paras()   # 检查缺失的参数

    def _load_paras(self):
        """
        加载各数据集的低维维度
        :return:
        """
        root = Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]
        leaf = ["DATA", "GRASSMANN", "Grassmann_data_paras.json"]
        root = list(root) + leaf
        json_path = "/".join(root)
        with open(json_path, 'r', encoding='utf-8') as paras:
            grassmann_paras = json.load(paras)
        paras.close()
        self.low_dimensions = grassmann_paras["low_dimensions"]

    def _check_paras(self):
        """
        检查实验参数是否完整
        :return:
        """
        for dn in self.data_name:
            if dn not in self.train_size.keys():
                self.train_size[dn] = 0.50
        for dn in self.data_name:
            self.low_dimensions[dn] = self.low_dimensions.get(dn + "-" + str(self.grassmann_p.get(dn))).get(str(self.remain_eta.get(dn)))
