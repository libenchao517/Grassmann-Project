################################################################################
# 本文件用于实现流形学习算法模型工厂
################################################################################
# 导入模块
import numpy as np
import random
import datetime
import platform
from sklearn.decomposition import PCA
from Contrast import Contrast_Method_GUMAP
from DATA import Load_Data
from DATA import datas
from DATA import abbre_labels
from DATA import Pre_Procession as PP
from GUMAP import GrassmannUMAP
from GDLPP import GrassmannLPP
from GDLPP import GrassmannDLPP
from GDLPP import GrassmannFLPP
from GDLPP import GrassmannSLPP
from GDNPE import GrassmannNPE
from GDNPE import GrassmannDNPE
from GDNPE import GrassmannFNPE
from GDNPE import GrassmannSNPE
################################################################################
class factory:
    """
    模型工厂
    """
    def __init__(
        self,
        func_name='UMAP',
        data_name='USPS',
        return_time=True,
        train_size=0.1,
        split_type=None,
        random_state=None,
        is_noisy=False,
        sigma=0.01,
        is_clip=False,
        clip_num=0,
        is_select_target = False,
        target_num = 0,
        sec_part='Comparatation',
        sec_num=0,
    ):
        """
        初始化函数
        :param func_name:   算法名称
        :param data_name:   数据名称
        :param return_time: 是否返回时间
        :param train_size:  训练比例
        :param split_type:  训练集测试集划分方法
        :param random_state: 随机种子
        :param is_noisy:     是否添加噪声
        :param sigma:        高斯噪声强度
        :param is_clip:      是否切割数据
        :param clip_num:     子集的规模
        :param is_select_target: 是否根据类别进行采样
        :param target_num:   选择的类别的数量
        :param sec_part:     项目名称
        :param sec_num:      实验序号
        """
        self.xn = 80
        print("#" * self.xn)
        print(func_name + "算法性能测试")
        print("*" * self.xn)
        print("性能指标：")
        print("*" * self.xn)
        print("测试日期：", datetime.date.today())
        print("测试时间：", datetime.datetime.now().time().strftime("%H:%M:%S"))
        print("计算机名：", platform.node())
        print("操作系统：", platform.system())
        print("解 释 器：", platform.python_version())
        print("数 据 集：", data_name)
        print("算法名称：", func_name)
        print("*" * self.xn)
        self.data_name = data_name
        self.func_name = func_name
        self.random_state = random_state
        self.return_time = return_time
        self.train_size = train_size
        self.split_type = split_type
        self.is_noisy = is_noisy
        self.sigma =sigma
        self.is_clip = is_clip
        self.clip_num = clip_num
        self.is_select_target = is_select_target
        self.target_num = target_num
        self.sec_part = sec_part
        self.sec_num = sec_num

    def Product_Grassmann_UMAP_Object(
            self,
            n_components=30,
            n_neighbors=15,
            p_grassmann=10
    ):
        """
        生产Grassmann UMAP算法对象
        :param n_components: 目标维度
        :param n_neighbors:  近邻数
        :param p_grassmann:  子空间维度
        :return:
        """
        # 加载数据集
        self.data, self.target = Load_Data(self.data_name + "-" + str(p_grassmann)).Loading()
        # 初始化对象
        self.Grassmann_UMAP_Object = GrassmannUMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            p_grassmann=p_grassmann,
            random_state=self.random_state,
            train_size=self.train_size,
            data_name=self.data_name,
            func_name=self.func_name,
            return_time=self.return_time,
            sec_part=self.sec_part,
            sec_num=self.sec_num)
        return self.Grassmann_UMAP_Object

    def Product_Grassmann_LPP_Object(
            self,
            n_components=30,
            n_neighbors=15,
            p_grassmann=10,
            alpha=0.5,
            n_cluster=None,
            converged_tol=1,
            max_epoch=20,
            is_discriminant=False,
            is_semi_supervised=False,
            is_self_supervised=False
    ):
        """
        生产Grassmann LPP算法对象
        :param n_components:       目标维度
        :param n_neighbors:        邻居数
        :param p_grassmann:        子空间阶数
        :param alpha:              半监督算法的正则化系数
        :param n_cluster:          数据集中的类别数
        :param converged_tol:      循环结束条件
        :param max_epoch:          最大迭代次数
        :param is_discriminant:    GDLPP标志
        :param is_semi_supervised: GSLPP标志
        :param is_self_supervised: GFLPP标志
        :return:
        """
        # 加载数据集
        self.data, self.target = Load_Data(self.data_name + "-" + str(p_grassmann)).Loading()
        # 划分训练集和测试集
        train_index, test_index, sampling = self.split_index(data=self.data, target=self.target, train_size=self.train_size, random_state=self.random_state)
        # 初始化模型
        if is_discriminant:
            if is_semi_supervised:
                self.Grassmann_LPP_Object = GrassmannSLPP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    p_grassmann=p_grassmann,
                    alpha=alpha,
                    train_size=self.train_size,
                    random_state=self.random_state,
                    converged_tol=converged_tol,
                    max_epoch=max_epoch,
                    data_name=self.data_name,
                    func_name=self.func_name,
                    return_time=self.return_time,
                    sec_part=self.sec_part,
                    sec_num=self.sec_num)
            elif is_self_supervised:
                self.Grassmann_LPP_Object = GrassmannFLPP()
            else:
                self.Grassmann_LPP_Object = GrassmannDLPP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    p_grassmann=p_grassmann,
                    train_size=self.train_size,
                    random_state=self.random_state,
                    converged_tol=converged_tol,
                    max_epoch=max_epoch,
                    data_name=self.data_name,
                    func_name=self.func_name,
                    return_time=self.return_time,
                    sec_part=self.sec_part,
                    sec_num=self.sec_num)
        else:
            self.Grassmann_LPP_Object = GrassmannLPP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                p_grassmann=p_grassmann,
                train_size=self.train_size,
                random_state=self.random_state,
                converged_tol=converged_tol,
                max_epoch=max_epoch,
                data_name=self.data_name,
                func_name=self.func_name,
                return_time=self.return_time,
                sec_part=self.sec_part,
                sec_num=self.sec_num)
        # 将训练集和测试集索引添加到对象上
        setattr(self.Grassmann_LPP_Object, "train_index", train_index)
        setattr(self.Grassmann_LPP_Object, "test_index", test_index)
        setattr(self.Grassmann_LPP_Object, "sampling", sampling)
        setattr(self.Grassmann_LPP_Object, "space", "grassmann")
        return self.Grassmann_LPP_Object

    def Product_Grassmann_NPE_Object(
            self,
            n_components=30,
            n_neighbors=5,
            p_grassmann=10,
            alpha=0.5,
            n_cluster=None,
            mode = 1,
            converged_tol=1.0,
            max_epoch=2,
            is_discriminant=False,
            is_semi_supervised=False,
            is_self_supervised=False
    ):
        """
        生产Grassmann NPE算法对象
        :param n_components:       目标维度
        :param n_neighbors:        邻居数
        :param p_grassmann:        子空间阶数
        :param alpha:              半监督算法的正则化系数
        :param n_cluster:          数据集中的类别数
        :param mode:               构建k近邻图的方法
        :param converged_tol:      循环结束条件
        :param max_epoch:          最大迭代次数
        :param is_discriminant:    GDNPE标志
        :param is_semi_supervised: GSNPE标志
        :param is_self_supervised: GFNPE标志
        :return:
        """
        # 加载数据集
        self.data, self.target = Load_Data(self.data_name + "-" + str(p_grassmann)).Loading()
        # 划分训练集和测试集
        train_index, test_index, sampling = self.split_index(data=self.data, target=self.target, train_size=self.train_size, random_state=self.random_state)
        # 初始化对象
        if is_discriminant:
            if is_semi_supervised:
                self.Grassmann_NPE_Object = GrassmannSNPE(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    p_grassmann=p_grassmann,
                    train_size=self.train_size,
                    random_state=self.random_state,
                    mode=mode,
                    alpha=alpha,
                    converged_tol=converged_tol,
                    max_epoch=max_epoch,
                    data_name=self.data_name,
                    func_name=self.func_name,
                    return_time=self.return_time,
                    sec_part=self.sec_part,
                    sec_num=self.sec_num)
            elif is_self_supervised:
                self.Grassmann_NPE_Object = GrassmannFNPE()
            else:
                self.Grassmann_NPE_Object = GrassmannDNPE(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    p_grassmann=p_grassmann,
                    train_size=self.train_size,
                    random_state=self.random_state,
                    mode=mode,
                    converged_tol=converged_tol,
                    max_epoch=max_epoch,
                    data_name=self.data_name,
                    func_name=self.func_name,
                    return_time=self.return_time,
                    sec_part=self.sec_part,
                    sec_num=self.sec_num)
        else:
            self.Grassmann_NPE_Object = GrassmannNPE(
                n_components=n_components,
                n_neighbors=n_neighbors,
                p_grassmann=p_grassmann,
                train_size=self.train_size,
                random_state=self.random_state,
                mode=mode,
                converged_tol=converged_tol,
                max_epoch=max_epoch,
                data_name=self.data_name,
                func_name=self.func_name,
                return_time=self.return_time,
                sec_part=self.sec_part,
                sec_num=self.sec_num)
        # 将训练集和测试集索引添加到对象上
        setattr(self.Grassmann_NPE_Object, "train_index", train_index)
        setattr(self.Grassmann_NPE_Object, "test_index", test_index)
        setattr(self.Grassmann_NPE_Object, "sampling", sampling)
        setattr(self.Grassmann_NPE_Object, "space", "grassmann")
        return self.Grassmann_NPE_Object

    def Product_Riemannian_DR_Object(
            self,
            n_components=30,
            n_neighbors=15,
            p_grassmann = 10,
            converged_tol=1.0,
            drop_tol=1e-6,
            max_epoch=20,
            verbose=False
    ):
        """
        生产Riemannian降维算法对象
        :param n_components:       目标维度
        :param n_neighbors:        邻居数
        :param p_grassmann:        子空间阶数
        :param converged_tol:      循环结束条件
        :param drop_tol:
        :param max_epoch:          最大迭代次数
        :param verbose:            可视化标志
        :return:
        """
        # 加载数据集
        self.data, self.target = Load_Data(self.data_name + "-" + str(p_grassmann)).Loading()
        # 划分训练集和测试集
        train_index, test_index, sampling = self.split_index(data=self.data, target=self.target, train_size=self.train_size, random_state=self.random_state)
        # 初始化对象
        self.Grassmann_Contrast_Object = Contrast_Method_GUMAP(
            data=self.data,
            target=self.target,
            n_components=n_components,
            n_neighbors=n_neighbors,
            p_grassmann=p_grassmann,
            train_size=self.train_size,
            random_state=self.random_state,
            converged_tol=converged_tol,
            drop_tol=drop_tol,
            max_epoch=max_epoch,
            verbose=verbose,
            data_name=self.data_name,
            func_name=self.func_name,
            return_time=self.return_time,
            sec_part=self.sec_part,
            sec_num=self.sec_num)
        # 将训练集和测试集索引添加到对象上
        setattr(self.Grassmann_Contrast_Object, "train_index", train_index)
        setattr(self.Grassmann_Contrast_Object, "test_index", test_index)
        setattr(self.Grassmann_Contrast_Object, "sampling", sampling)
        return self.Grassmann_Contrast_Object

    def split_index(self, data, target, train_size, random_state):
        """
        划分训练集和测试集
        :param data:         全体数据
        :param target:       全体标签
        :param train_size:   训练比例
        :param random_state: 随机种子
        :return:
        """
        train_index, test_index = PP().sub_one_sampling_index(data=data, target=target, train_size=train_size, random_state=random_state)
        sampling = 'sub-one'
        return train_index, test_index, sampling
