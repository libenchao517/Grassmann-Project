################################################################################
# 本文件是GUMAP项目的配置文件
################################################################################
class GrassmannUMAP_config:
    def __init__(self):
        self.basic_data = [ # 参数实验数据集
            "ETH-80",
            "Weizmann",
            "EYB",
            "UTD-MHAD"
        ]
        self.GUMAP_data = [ # 聚类分析数据集
            "ETH-80",
            "Weizmann",
            "UCF-S",
            "UT-Kinect",
            "EYB",
            "UTD-MHAD",
        ]
        self.GUMAP_components = 20  # 目标维度
        self.GUMAP_neighbors = 10   # 邻居数
        self.random_state = 517     # 随机种子
        self.return_time = True     # 是否返回时间
        self.grassmann_p = 10       # 子空间结束
