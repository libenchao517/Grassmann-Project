################################################################################
# 本文件用于对流形学习的对比算法进行封装和标准化
################################################################################
# 导入模块
import io
import sys
import warnings
warnings.filterwarnings("ignore")
from time import perf_counter
import numpy as np
from Grassmann import GrassmannKernel
from Grassmann import CGMKE
from Grassmann import GrassmannKNN
from Grassmann import GrassmannSVM
from Grassmann import GrassmannKernelFDA
from Grassmann import GrassmannALL
from Grassmann import Nested_Grassmann
from Grassmann import GRLGQ_Run
from sklearn.model_selection import train_test_split
################################################################################
class Contrast_Method_GUMAP:
    """
    对GDLPP、GDNPE和GUMAP项目中的对比方法进行统一封装
    """
    def __init__(
        self,
        data,
        target,
        n_components=10,
        n_neighbors=10,
        p_grassmann=10,
        train_size = 0.5,
        random_state = 517,
        converged_tol=1.0,
        drop_tol=1e-6,
        max_epoch=20,
        verbose=False,
        func_name='GKDA',
        data_name='USPS',
        return_time=True,
        sec_part='Comparatation',
        sec_num=1
    ):
        """
        初始化函数
        :param data:   全体数据
        :param target: 全体标签
        :param n_components:  目标维度
        :param n_neighbors:   近邻数
        :param p_grassmann:   子空间阶数
        :param train_size:    训练比例
        :param random_state:  随机种子
        :param converged_tol: 收敛条件
        :param drop_tol:      过滤过小特征值的超参数
        :param max_epoch:     最大迭代次数
        :param verbose:       可视化标志
        :param func_name:     算法名称
        :param data_name:     数据名称
        :param return_time:   是否返回时间
        :param sec_part:      项目名称
        :param sec_num:       实验编号
        """
        self.data = data
        self.target = target
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.p_grassmann = p_grassmann
        self.train_size = train_size
        self.random_state = random_state
        self.converged_tol = converged_tol
        self.drop_tol = drop_tol
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.func_name = func_name
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, 'total']
        self.return_time = return_time
        self.start_text = "当前正在" + data_name + "数据集上运行" + func_name + "算法......"
        self.train_index = None   # 预定义的训练集索引
        self.test_index = None    # 预定义的测试集索引

    def print_time(self):
        """
        格式输出时间
        :return:
        """
        print(
            "\r",
            "{:8s}".format(self.para[2]),
            "{:8s}".format(self.para[3]),
            "{:8s}".format("time"),
            "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start

    def embedded(self):
        """
        统一的调用函数
        :return:
        """
        print(self.start_text, end="")
        self.split_data()
        self.time_start = perf_counter()              # 开始计时
        eval("self." + self.func_name.upper() + "()") # 运行算法
        self.time_end = perf_counter()                # 结束计时
        self.print_time()

    def split_data(self):
        """
        根据预定义的训练集索引和测试集索引划分训练集和测试集
        :return:
        """
        self.data_train = self.data[self.train_index]
        self.data_test = self.data[self.test_index]
        self.target_train = self.target[self.train_index]
        self.target_test = self.target[self.test_index]

    def NOP(self):
        """
        不对高维数据进行任何处理
        :return:
        """
        self.embedding_ = self.data

    def GKNN(self):
        """
        格拉斯曼流形上的最近邻分类器
        :return:
        """
        self.space = None
        KNN = GrassmannKNN()                        # 初始化分类器
        KNN.fit(self.data_train, self.target_train) # 训练分类器
        self.t_pred = KNN.predict(self.data_test)   # 预测

    def GSVM(self):
        """
        格拉斯曼流形上的支持向量机
        :return:
        """
        self.space = None
        SVM = GrassmannSVM()
        SVM.fit(self.data_train, self.target_train)
        self.t_pred = SVM.transform(self.data_test)

    def GRLGQ(self):
        """
        Generalized Relevance Learning Grassmann Quantization
        :return:
        """
        self.space = None
        self.t_pred, self.accuracy = GRLGQ_Run(
            dim_of_subspace=self.p_grassmann,
            nepochs=250
        ).fit(
            data_train=self.data_train,
            data_test=self.data_test,
            target_train=self.target_train,
            target_test=self.target_test)

    def NG(self):
        """
        Nested Grassmann
        :return:
        """
        self.space = "grassmann"
        self.embedding_ = Nested_Grassmann().NG_dr(self.data, m=self.n_components)
        self.embedding_train = self.embedding_[self.train_index]
        self.embedding_test = self.embedding_[self.test_index]

    def SNG(self):
        """
        Supervised Nested Grassmann
        :return:
        """
        self.space = "grassmann"
        self.embedding_ = Nested_Grassmann().NG_sdr(self.data, self.target, m=self.n_components)
        self.embedding_train = self.embedding_[self.train_index]
        self.embedding_test = self.embedding_[self.test_index]

    def GALL(self):
        """
        Grassmann Adaptive Local Learning
        :return:
        """
        self.space = "grassmann"
        GA = GrassmannALL(
            n_components=self.n_components,
            p_grassmann=self.p_grassmann,
            n_neighbors=self.n_neighbors,
            train_size=self.train_size,
            random_state=self.random_state,
            converged_tol=self.converged_tol,
            drop_tol=self.drop_tol,
            max_epoch=self.max_epoch,
            verbose=self.verbose)
        self.embedding_train, self.embedding_test = GA.fit_transform(self.data_train, self.data_test, self.target_train, self.target_test)

    def CGMKE(self):
        """
        Clustering on Grassmann Manifold via Kernel Embedding
        :return:
        """
        value, count = np.unique(self.target, return_counts=True)
        CG = CGMKE(center_count=len(count))
        self.space = "euclidean"
        self.embedding_ = CG.trans_data(self.data)
        self.embedding_train = self.embedding_[self.train_index]
        self.embedding_test = self.embedding_[self.test_index]

    def GKDA(self):
        """
        Discriminant Analysis on Grassmann Manifold
        :return:
        """
        n_cluster = len(np.unique(self.target))
        n_components = n_cluster - 1 if n_cluster>2 else n_cluster
        GK = GrassmannKernel()
        GKDA = GrassmannKernelFDA(n_components=n_components, kernel=GK.projection_kernel)
        self.space = "euclidean"
        GKDA.fit(self.data_train, self.target_train)
        self.embedding_train = GKDA.transform(self.data_train)
        self.embedding_test = GKDA.transform(self.data_test)
        self.embedding_ = np.random.random((self.data.shape[0], self.embedding_train.shape[1]))
        self.embedding_[self.train_index] = self.embedding_train
        self.embedding_[self.test_index] = self.embedding_test
