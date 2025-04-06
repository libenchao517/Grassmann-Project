################################################################################
# 本文件用于对Riemannian降维算法的评价进行标准化
################################################################################
# 导入模块
import os
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import normalized_mutual_info_score
from .Assessment import K_Nearest_Neighbors
from .Assessment import Support_Vector_Machine
from .Assessment import Adjusted_Rand_Score
from .Assessment import Ajusted_Mutual_Info_Score
from .Assessment import Normalized_Mutual_Info_Score
from .Assessment import Homogeneity_Score
from .Assessment import Completeness_Score
from .Assessment import V_Measure_Score
from .Assessment import Fowlkes_Mallows_Score
from .Assessment import Cluster_Accuracy
from Draw import Confusion_Matrix
from Draw import Draw_Embedding
from Grassmann import GrassmannKMeans
from Grassmann import GrassmannLBG
from Grassmann import CGMKE
from Grassmann import Shared_Nearest_Neighbor_DPC
from Grassmann import Grassmann_AffinityPropagation
from Grassmann import Grassmann_AgglomerativeClustering
from Grassmann import Grassmann_DBSCAN
from Grassmann import Grassmann_HDBSCAN
from Grassmann import Grassmann_OPTICS
from Grassmann import Grassmann_SpectralClustering
from Grassmann import GrassmannDistance
from time import perf_counter
################################################################################
class Analysis_Riemannian:
    """
    Riemannian降维算法标准化分析过程
    """
    def __init__(self, object, save_file=True, print_results=True):
        """
        初始化函数
        :param object: 降维方法的实例化对象
        :param save_file: 是否保存结果文件
        :param print_results: 是否打印结果
        """
        self.object = object
        self.save_file = save_file
        self.print_results = print_results
        # 初始化分类器和指标
        self.knn = K_Nearest_Neighbors()
        self.svm = Support_Vector_Machine()
        self.nmi = Normalized_Mutual_Info_Score()
        self.GD = GrassmannDistance()
        # 初始化存储结果的表格
        self.result = pd.DataFrame(
            columns=[
                "Method", "Datasets",
                "ACC", "PRE", "F1", "REC", "MCC", "NMI",
                "time", "train-size", "sampling",
                "WT", "FT", "BT", "JT", "ST", "NT"],
            index=["Total"])
        if save_file:
            self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
            self.cmat_path = "-".join(self.object.para[0:4]) + '-Confusion-Matrix'
            self.cmat = Confusion_Matrix(path="./Figure/" + self.object.data_name, filename=self.cmat_path)
        self.xn = 80
        self.classifer = ["GKNN", "GSVM", "GRLGQ", "GrNet", "GPML", "default"]

    def Analysis(self, classification = True):
        """
        分析的过程
        :param classification: 分类标志
        :return:
        """
        if self.save_file:
            os.makedirs("./Analysis/" + self.object.data_name, exist_ok=True)
            os.makedirs("./Figure/" + self.object.data_name, exist_ok=True)

        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)

        self.result["Method"].Total = self.object.func_name
        self.result["Datasets"].Total = self.object.data_name
        self.result["train-size"].Total = self.object.train_size
        self.result["sampling"].Total = self.object.sampling

        if self.object.func_name in self.classifer:
            classification_label = self.object.t_pred

        if self.object.space == "grassmann":
            if classification:
                self.knn.KNN_predict_odds_grassmann(self.object.embedding_train, self.object.embedding_test, self.object.target_train, self.object.target_test, self.object.para)
                classification_label = self.knn.t_pred

        elif self.object.space == "euclidean":
            if classification:
                self.knn.KNN_predict_odds_splited(self.object.embedding_train, self.object.embedding_test, self.object.target_train, self.object.target_test, self.object.para)
                classification_label = self.knn.t_pred

        if classification:
            self.result["ACC"].Total = accuracy_score(self.object.target_test, classification_label)
            self.result["PRE"].Total = precision_score(self.object.target_test, classification_label, average="macro")
            self.result["F1"].Total = f1_score(self.object.target_test, classification_label, average="macro")
            self.result["REC"].Total = recall_score(self.object.target_test, classification_label, average="macro")
            self.result["MCC"].Total = matthews_corrcoef(self.object.target_test, classification_label)
            # self.result["NMI"].Total = normalized_mutual_info_score(self.object.target_test, classification_label)

        self.result["time"].Total = self.object.time
        # 用于分析算法中每种运算的时间
        if hasattr(self.object, "WT"):
            self.result["WT"].Total = self.object.WT
        if hasattr(self.object, "FT"):
            self.result["FT"].Total = self.object.FT
        if hasattr(self.object, "BT"):
            self.result["BT"].Total = self.object.BT
        if hasattr(self.object, "JT"):
            self.result["JT"].Total = self.object.JT
        if hasattr(self.object, "ST"):
            self.result["ST"].Total = self.object.ST
        if hasattr(self.object, "NT"):
            self.result["NT"].Total = self.object.NT

        print(self.result)
        print("*" * self.xn)

        if self.save_file :
            self.result.to_excel("./Analysis/" + self.object.data_name + "/" + self.xlsx_path)
            if len(np.unique(self.object.target)) <= 15:
                self.cmat.Drawing(self.object.target_test, classification_label)
################################################################################
class Analysis_Rieman_Cluster:
    """
    格拉斯曼流形上聚类任务的性能分析
    """
    def __init__(self, object, cluster=None, save_file=True, print_results=False):
        """
        初始化函数
        :param object: 聚类方法的实例化对象
        :param cluster: 聚类方法列表
        :param save_file: 是否保存文件
        :param print_results: 是否打印结果
        """
        self.object = object
        self.save_file = save_file
        self.print_results = print_results
        # 初始化指标
        self.ari = Adjusted_Rand_Score()
        self.ami = Ajusted_Mutual_Info_Score()
        self.nmi = Normalized_Mutual_Info_Score()
        self.hms = Homogeneity_Score()
        self.cms = Completeness_Score()
        self.vms = V_Measure_Score()
        self.fms = Fowlkes_Mallows_Score()
        self.acc = Cluster_Accuracy()
        # 初始化聚类方法
        self.CGMKE = CGMKE(center_count=len(np.unique(object.target)))
        self.GKM = GrassmannKMeans(center_count=len(np.unique(object.target)), n_epoch=100)
        self.GLBG = GrassmannLBG(center_count=len(np.unique(object.target)), n_epoch=100)
        self.SNNDPCG = Shared_Nearest_Neighbor_DPC(n_cluster=len(np.unique(object.target)), n_neighbors=5)
        self.GAF = Grassmann_AffinityPropagation()
        self.GAG = Grassmann_AgglomerativeClustering(n_cluster=len(np.unique(object.target)))
        self.GDBS = Grassmann_DBSCAN()
        self.GHDBS = Grassmann_HDBSCAN()
        self.GOPT = Grassmann_OPTICS()
        self.GSC = Grassmann_SpectralClustering(n_clusters=len(np.unique(object.target)), neighbors=5)
        self.GD = GrassmannDistance()
        # 可执行的聚类方法列表
        pre_cluster = ["CGMKE", "GLBG", "GKM", "SNNDPCG", "GAF", "GAG", "GDBS", "GHDBS", "GOPT", "GSC"]
        self.cluster = pre_cluster if cluster is None else cluster
        # 保存结果的表格
        self.result = pd.DataFrame(
            columns=["Method", "Datasets", "ARI", "AMI", "NMI",
                     "HMS", "CMS", "VMS", "FMS", "ACC", "time"],
            index=self.cluster)

        if save_file:
            self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
        self.xn = 80

    def process(self, cluster_name):
        """
        执行聚类和分析结果的标准化过程
        :param cluster_name: 聚类方法名字
        :return:
        """
        # 存储算法名称
        self.object.para[-1] = cluster_name
        algo_name = self.object.para[-1] if self.object.func_name == "NOP" else self.object.func_name + "-" +  self.object.para[-1]
        self.result.loc[self.object.para[-1], "Method"] = algo_name

        # 执行聚类
        print("\r当前正在使用" + cluster_name +"方法进行聚类......", end="")
        t_start = perf_counter()
        _label = eval("self." + cluster_name + ".fit_transform(self.object.embedding_)")
        t_end = perf_counter()
        self.result.loc[self.object.para[-1], "time"] = t_end - t_start

        # 计算聚类指标
        print("\r当前正在计算" + cluster_name + "聚类算法的评价指标......", end="")
        self.ari.adjusted_rand_score_(self.object.target, _label)
        self.result.loc[self.object.para[-1], "ARI"] = self.ari.score
        self.ami.adjusted_mutual_info_score_(self.object.target, _label)
        self.result.loc[self.object.para[-1], "AMI"] = self.ami.score
        self.nmi.normalized_mutual_info_score_(self.object.target, _label)
        self.result.loc[self.object.para[-1], "NMI"] = self.nmi.score
        self.hms.homogeneity_score_(self.object.target, _label)
        self.result.loc[self.object.para[-1], "HMS"] = self.hms.score
        self.cms.completeness_score_(self.object.target, _label)
        self.result.loc[self.object.para[-1], "CMS"] = self.cms.score
        self.vms.v_measure_score_(self.object.target, _label)
        self.result.loc[self.object.para[-1], "VMS"] = self.vms.score
        self.fms.fowlkes_mallows_score_(self.object.target, _label)
        self.result.loc[self.object.para[-1], "FMS"] = self.fms.score
        self.acc.calculate_accuracy(self.object.target, _label)
        self.result.loc[self.object.para[-1], "ACC"] = self.acc.accuracy

    def Analysis(self):
        """
        分析过程的主函数
        :return:
        """
        if self.save_file:
            os.makedirs("./Analysis/" + self.object.data_name, exist_ok=True)
            os.makedirs("./Figure/" + self.object.data_name, exist_ok=True)

        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)

        self.result["Datasets"] = self.object.data_name
        
        for clust in self.cluster:
            self.process(clust)

        print("\r" + " " * self.xn)
        print(self.result)
        print("*" * self.xn)
        if self.save_file:
            self.result.to_excel("./Analysis/" + self.object.data_name + "/" + self.xlsx_path)
