################################################################################
# 本文件用于实现Grassmann LPP及相关算法
################################################################################
# 导入模块
import scipy
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from Grassmann import GrassmannDistance
from Grassmann import GrassmannSubSpace
from time import perf_counter
from tqdm import tqdm
from DATA import Pre_Procession as pp
################################################################################
class GrassmannLPP:
    """
    Locality Preserving Projection on Grassmann Manifold
    [1] Wang B, Hu Y, Gao J, et al.
    Locality Preserving Projections for Grassmann Manifold [C].
    In: International Joint Conference on Artificial Intelligence.
    Morgan Kaufmann, 2017, 2893-2900.
    """
    def __init__(
            self,
            n_components=2,
            p_grassmann=10,
            n_neighbors=5,
            train_size=5,
            random_state=517,
            weight_width=1.0,
            neighbors_algorithm="auto",
            converged_tol=1,
            drop_tol=1e-6,
            max_epoch=20,
            verbose=False,
            func_name="GLPP",
            data_name="ETH-80",
            sec_part="Experiment",
            sec_num=1,
            return_time=True
    ):
        """
        初始化函数
        :param n_components:        目标维度
        :param p_grassmann:         子空间阶数
        :param n_neighbors:         邻居数
        :param train_size:          训练集比例
        :param random_state:        随机种子
        :param weight_width:        热核参数
        :param neighbors_algorithm: 计算邻居的方法
        :param converged_tol:       终止迭代的条件
        :param drop_tol:
        :param max_epoch:           最大迭代次数
        :param verbose:             是否进行输出
        :param func_name:           函数名称
        :param data_name:           数据集名称
        :param sec_part:
        :param sec_num:
        :param return_time:         是否返回时间
        """
        self.n_components = n_components
        self.p_grassmann = p_grassmann
        self.n_neighbors = n_neighbors
        self.train_size = train_size
        self.random_state = random_state
        self.weight_width = weight_width
        self.neighbors_algorithm = neighbors_algorithm
        self.converged_tol = converged_tol
        self.drop_tol = drop_tol
        self.max_epoch = max_epoch
        self.GD = GrassmannDistance()
        self.metric = self.GD.gdist
        self.object_value = np.array([])
        self.verbose = verbose
        self.func_name = func_name
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, ""]
        self.return_time = return_time
        self.time = None

    def init_A(self, data):
        """
        初始化映射矩阵
        :param data: 数据集 [N, D, p]
        :return: 映射矩阵 [D, d]
        """
        top = np.eye(self.n_components)
        bottom = np.random.random((data.shape[1] - self.n_components, self.n_components))
        return np.concatenate((top, bottom), axis=0)

    def orthogonal_subspace(self, data, A):
        """
        在每次迭代中对数据集进行标准化
        :param data: 数据集 [N, D, p]
        :param A:    映射矩阵 [D, d]
        :return: [N, D, p]
        """
        D = []
        for d in data:
            q, r = np.linalg.qr(np.dot(np.transpose(A), d))
            try:
                D.append(np.dot(d, np.linalg.inv(r)))
            except np.linalg.LinAlgError:
                D.append(np.dot(d, np.linalg.pinv(r)))
        return np.array(D)

    def transform(self, data):
        """
        对数据进行降维
        :param data: 数据集 [N, D, p]
        :return: 低维数据 [N, d, p]
        """
        embedding = []
        for d in data:
            embedding.append(np.dot(np.transpose(self.components_), d))
        return np.array(embedding)

    def compute_weights(self, dist, n_neighbors=None):
        """
        计算权重矩阵
        :param dist: 距离矩阵 [N, N]
        :param n_neighbors: 邻居数
        :return:
        """
        if n_neighbors is None:
            n_neighbors = dist.shape[0] - 1
        if n_neighbors < 1:
            return np.array([[0]]), np.array([[0]])
        self.nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(dist)
        W = kneighbors_graph(self.nbrs_, n_neighbors, metric="precomputed", mode='distance')
        W.data = np.exp(-W.data ** 2)
        W = W.toarray()
        knn_index = np.where(W > 0)[1]
        knn_index = knn_index.reshape((W.shape[0], n_neighbors))
        return W, knn_index

    def optimization(self, data, W, D, knn_index, A):
        """
        优化函数
        :param data:
        :param W:
        :param D:
        :param knn_index:
        :param A:
        :return:
        """
        H = np.zeros((data.shape[1], data.shape[1]))
        J = np.zeros((data.shape[1], data.shape[1]))
        tj0 = perf_counter()
        for i in tqdm(range(data.shape[0])):
            GI = np.dot(data[i], data[i].T)
            H = H + D[i] * GI
            for j in range(self.n_neighbors):
                k = knn_index[i, j]
                G = GI - np.dot(data[k], data[k].T)
                J = J + W[i, k] * np.dot(np.dot(np.dot(G, A), A.T), G)
        tj1 = perf_counter()
        temp = tj1 - tj0
        self.JT += temp
        eig_values, eig_vectors = scipy.linalg.eig(a=J, b=H)
        sort_index_ = np.argsort(np.abs(eig_values))
        eig_values = eig_values[sort_index_]
        j = 0
        while np.abs(eig_values[j]) < self.drop_tol:
            j += 1
        index_ = sort_index_[j: j + self.n_components]
        A = eig_vectors[:, index_]
        return A, np.trace(np.dot(np.dot(A.T, J), A))

    def fit(self, data):
        """
        训练过程
        :param data:
        :return:
        """
        self.coms = []
        self.components_ = self.init_A(data)
        self.coms.append(self.components_)
        tw0 = perf_counter()
        self.dist = self.GD.pairwise_dist(data, self.metric)
        W, knn_index = self.compute_weights(self.dist, self.n_neighbors)
        tw1 =perf_counter()
        self.WT = tw1 - tw0
        D = W.sum(axis=1)
        self.JT = 0
        self.NT = 0
        epoch = 0
        # 迭代
        while len(self.object_value) <= 1 or np.abs(self.object_value[-1] - self.object_value[-2]) > self.converged_tol:
            tn0 = perf_counter()
            data = self.orthogonal_subspace(data, self.components_)
            tn1 = perf_counter()
            temp = tn1 - tn0
            self.NT += temp
            self.components_, current_object = self.optimization(data, W, D, knn_index, self.components_)
            self.coms.append(self.components_)
            epoch += 1
            if self.verbose:
                print("第{:d}次迭代的目标值：".format(epoch) + "{:.4f}".format(current_object))
            self.object_value = np.append(self.object_value, current_object)
            if epoch >= self.max_epoch:
                break
        self.NT /= epoch
        self.JT /= epoch
        return self

    def fit_transform(self, data, target):
        """
        主函数
        :param data:    数据集 [N, D, p]
        :param target:  标签
        :return:
        """
        self.data = data
        self.target = target
        self.split_data()                  # 划分训练集和测试集
        self.time_start = perf_counter()
        self.fit(self.data_train)          # 训练
        self.time_end = perf_counter()
        self.print_time()
        for a in self.coms: # 标准化数据集
            self.data_train = self.orthogonal_subspace(self.data_train, a)
            self.data_test = self.orthogonal_subspace(self.data_test, a)
        self.embedding_train = self.transform(self.data_train)
        self.embedding_test = self.transform(self.data_test)
        return self

    def split_data(self):
        """
        根据预定义的训练集索引和测试集索引进行分割
        :return:
        """
        self.data_train = self.data[self.train_index]
        self.data_test = self.data[self.test_index]
        self.target_train = self.target[self.train_index]
        self.target_test = self.target[self.test_index]

    def print_time(self):
        """
        打印运行时间
        :return:
        """
        print("\r", "{:8s}".format(self.para[2]),
              "{:8s}".format(self.para[3]),
              "{:8s}".format("time"),
              "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start
################################################################################
class GrassmannDLPP:
    """
    Discriminant Locality Preserving Projection on Grassmann Manifold
    [1] Li B, Wang T, Ran R.
    Discriminant locality preserving projection on Grassmann Manifold for image-set classification[J].
    The Journal of Supercomputing, 2025, 81(2): 1-27.
    """
    def __init__(
            self,
            n_components=2,
            p_grassmann=10,
            n_neighbors=5,
            train_size=5,
            random_state=517,
            weight_width=1.0,
            neighbors_algorithm="auto",
            converged_tol=1,
            drop_tol=1e-6,
            max_epoch=2,
            verbose=True,
            func_name="GDLPP",
            data_name="ETH-80",
            sec_part="Experiment",
            sec_num=1,
            return_time=True
    ):
        """
        初始化函数
        :param n_components:        目标维度
        :param p_grassmann:         子空间阶数
        :param n_neighbors:         邻居数
        :param train_size:          训练集比例
        :param random_state:        随机种子
        :param weight_width:        热核参数
        :param neighbors_algorithm: 计算邻居的方法
        :param converged_tol:       终止迭代的条件
        :param drop_tol:
        :param max_epoch:           最大迭代次数
        :param verbose:             是否进行输出
        :param func_name:           函数名称
        :param data_name:           数据集名称
        :param sec_part:
        :param sec_num:
        :param return_time:         是否返回时间
        """
        self.n_components = n_components
        self.p_grassmann = p_grassmann
        self.n_neighbors = n_neighbors
        self.train_size = train_size
        self.random_state = random_state
        self.weight_width = weight_width
        self.neighbors_algorithm = neighbors_algorithm
        self.converged_tol = converged_tol
        self.drop_tol = drop_tol
        self.max_epoch = max_epoch
        self.GD = GrassmannDistance()
        self.GS = GrassmannSubSpace()
        self.metric = self.GD.gdist
        self.object_value = np.array([])
        self.verbose = verbose
        self.func_name = func_name
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, ""]
        self.return_time = return_time
        self.time = None

    def init_A(self, data):
        """
        初始化映射矩阵
        :param data: 数据集 [N, D, p]
        :return: 映射矩阵 [D, d]
        """
        top = np.eye(self.n_components)
        bottom = np.random.random((data.shape[1] - self.n_components, self.n_components))
        return np.concatenate((top, bottom), axis=0)

    def orthogonal_subspace(self, data, A):
        """
        标准化数据集
        :param data: 数据集 [N, D, p]
        :param A: 映射矩阵 [D, d]
        :return:
        """
        D = []
        for d in data:
            q, r = np.linalg.qr(np.dot(np.transpose(A), d))
            try:
                D.append(np.dot(d, np.linalg.inv(r)))
            except np.linalg.LinAlgError:
                D.append(np.dot(d, np.linalg.pinv(r)))
        return np.array(D)

    def transform(self, data):
        """
        对数据集执行降维
        :param data: 数据集 [N, D, p]
        :return: [N, d, p]
        """
        embedding = []
        for d in data:
            embedding.append(np.dot(np.transpose(self.components_), d))
        return np.array(embedding)

    def compute_weights(self, dist, n_neighbors=None):
        """
        计算权重矩阵
        :param dist: 距离矩阵
        :param n_neighbors: 邻居数
        :return:
        """
        if n_neighbors is None:
            n_neighbors = dist.shape[0] - 1
        if n_neighbors < 1:
            return np.array([[0]]), np.array([[0]])
        self.nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(dist)
        W = kneighbors_graph(self.nbrs_, n_neighbors, metric="precomputed", mode='distance')
        W.data = np.exp(-W.data ** 2)
        W = W.toarray()
        knn_index = np.where(W > 0)[1]
        knn_index = knn_index.reshape((W.shape[0], n_neighbors))
        return W, knn_index

    def calculate_center(self, data, target):
        """
        计算每个簇的中心（外部均值）
        :param data:
        :param target:
        :return:
        """
        centers = []
        for t in np.unique(target):
            data_t = data[t == target]
            s = np.zeros((data_t.shape[1], data_t.shape[1]))
            for d in data_t:
                s += np.dot(d, np.transpose(d))
            eig_values, eig_vectors = np.linalg.eigh(s)
            idx = np.argsort(eig_values)[::-1][:self.p_grassmann]
            centers.append(eig_vectors[:, idx])
        return np.array(centers)

    def optimization(self, data, target, F, W, B, knn_index, knn_index_F, A):
        """
        优化函数
        :param data: 数据集
        :param target: 标签
        :param F: 中心点
        :param W: 类内权重
        :param B: 类间权重
        :param knn_index: 类内邻居索引
        :param knn_index_F: 类间邻居索引
        :param A: 映射矩阵
        :return:
        """
        tj0 = perf_counter()
        J = np.zeros((data.shape[1], data.shape[1]))
        for c, t in enumerate(np.unique(target)):
            data_t = data[target == t]
            n_neighbors = data_t.shape[0] - 1
            for i in range(data_t.shape[0]):
                GI = np.dot(data_t[i], data_t[i].T)
                for j in range(n_neighbors):
                    k = knn_index[c][i, j]
                    G = GI - np.dot(data_t[k], data_t[k].T)
                    J = J + W[c][i, k] * np.dot(np.dot(np.dot(G, A), A.T), G)
        tj1 = perf_counter()
        temp = tj1 - tj0
        self.JT += temp
        tz0 = perf_counter()
        Z = np.zeros((F.shape[1], F.shape[1]))
        for i in range(F.shape[0]):
            SI = np.dot(F[i], F[i].T)
            for j in range(F.shape[0] - 1):
                k = knn_index_F[i, j]
                S = SI - np.dot(F[k], F[k].T)
                Z = Z + B[i, k] * np.dot(np.dot(np.dot(S, A), A.T), S)
        tz1 = perf_counter()
        temp = tz1 - tz0
        self.ST += temp
        eig_values, eig_vectors = scipy.linalg.eig(a=J, b=Z)
        sort_index_ = np.argsort(np.abs(eig_values))
        eig_values = eig_values[sort_index_]
        j = 0
        while np.abs(eig_values[j]) < self.drop_tol:
            j += 1
        index_ = sort_index_[j: j + self.n_components]
        A = eig_vectors[:, index_]
        # loss = np.trace(np.dot(np.dot(A.T, J), A)) / np.trace(np.dot(np.dot(A.T, Z), A))
        loss = np.linalg.norm(np.dot(np.dot(A.T, J), A)) / np.linalg.norm(np.dot(np.dot(A.T, Z), A))
        return A, loss

    def fit(self, data, target):
        """
        训练过程
        :param data:
        :param target:
        :return:
        """
        self.coms = []
        tf0 = perf_counter()
        F = self.calculate_center(data=data, target=target)
        tf1 = perf_counter()
        self.FT = tf1 - tf0
        self.components_ = self.init_A(data)
        self.coms.append(self.components_)
        # 计算权重
        tw0 = perf_counter()
        W = []
        knn_index = []
        for t in np.unique(target):
            data_t = data[t == target]
            dist_t = self.GD.pairwise_dist(data_t, self.metric)
            W_, index_ = self.compute_weights(dist_t)
            W.append(W_)
            knn_index.append(index_)
        tw1 = perf_counter()
        self.WT = tw1 - tw0
        tb0 = perf_counter()
        dist_F = self.GD.pairwise_dist(F, self.metric)
        B, knn_index_F = self.compute_weights(dist_F)
        tb1 = perf_counter()
        self.BT = tb1 - tb0
        epoch = 0
        self.NT = 0
        self.JT = 0
        self.ST = 0
        # 迭代
        while len(self.object_value) <= 1 or np.abs(self.object_value[-1] - self.object_value[-2]) > self.converged_tol:
            tn0 = perf_counter()
            data = self.orthogonal_subspace(data, self.components_)
            tn1 = perf_counter()
            temp = tn1 - tn0
            self.NT += temp
            self.components_, current_object = self.optimization(data, target, F, W, B, knn_index, knn_index_F, self.components_)
            self.coms.append(self.components_)
            epoch += 1
            if self.verbose:
                print("第{:d}次迭代的目标值：".format(epoch) + "{:.4f}".format(current_object))
            self.object_value = np.append(self.object_value, current_object)
            if epoch >= self.max_epoch:
                break
        self.NT /= epoch
        self.JT /= epoch
        self.ST /= epoch
        return self

    def fit_transform(self, data, target):
        """
        主函数
        :param data:
        :param target:
        :return:
        """
        self.data = data
        self.target = target
        self.split_data()
        self.time_start = perf_counter()
        self.fit(self.data_train, self.target_train)
        self.time_end = perf_counter()
        self.print_time()
        for a in self.coms:
            self.data_train = self.orthogonal_subspace(self.data_train, a)
            self.data_test = self.orthogonal_subspace(self.data_test, a)
        self.embedding_train = self.transform(self.data_train)
        self.embedding_test = self.transform(self.data_test)
        return self

    def split_data(self):
        """
        分割训练数据和测试数据
        :return:
        """
        self.data_train = self.data[self.train_index]
        self.data_test = self.data[self.test_index]
        self.target_train = self.target[self.train_index]
        self.target_test = self.target[self.test_index]

    def print_time(self):
        """
        打印时间
        :return:
        """
        print("\r", "{:8s}".format(self.para[2]),
              "{:8s}".format(self.para[3]),
              "{:8s}".format("time"),
              "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start
################################################################################
class GrassmannSLPP:
    """
    Semi-Supervised Locality Preserving Projection on Grassmann Manifold
    """
    def __init__(
            self,
            n_components=2,
            p_grassmann=10,
            n_neighbors=5,
            train_size=5,
            random_state=517,
            weight_width=1.0,
            neighbors_algorithm="auto",
            converged_tol=1,
            drop_tol=1e-6,
            max_epoch=20,
            alpha=0.5,
            verbose=True,
            func_name="GSLPP",
            data_name="ETH-80",
            sec_part="Experiment",
            sec_num=1,
            return_time=True
    ):
        """
        初始化函数
        :param n_components:        目标维度
        :param p_grassmann:         子空间阶数
        :param n_neighbors:         邻居数
        :param train_size:          训练集比例
        :param random_state:        随机种子
        :param weight_width:        热核参数
        :param neighbors_algorithm: 计算邻居的方法
        :param converged_tol:       终止迭代的条件
        :param drop_tol:
        :param max_epoch:           最大迭代次数
        :param alpha:               正则化系数
        :param verbose:             是否进行输出
        :param func_name:           函数名称
        :param data_name:           数据集名称
        :param sec_part:
        :param sec_num:
        :param return_time:         是否返回时间
        """
        self.n_components = n_components
        self.p_grassmann = p_grassmann
        self.n_neighbors = n_neighbors
        self.train_size = train_size
        self.random_state = random_state
        self.weight_width = weight_width
        self.neighbors_algorithm = neighbors_algorithm
        self.converged_tol = converged_tol
        self.drop_tol = drop_tol
        self.max_epoch = max_epoch
        self.GD = GrassmannDistance()
        self.GS = GrassmannSubSpace()
        self.metric = self.GD.gdist
        self.object_value = np.array([])
        self.alpha = alpha
        self.verbose = verbose
        self.func_name = func_name
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, ""]
        self.return_time = return_time
        self.time = None

    def init_A(self, data):
        """
        初始化映射矩阵
        :param data: 数据集 [N, D, p]
        :return: 映射矩阵 [D, d]
        """
        top = np.eye(self.n_components)
        bottom = np.random.random((data.shape[1] - self.n_components, self.n_components))
        return np.concatenate((top, bottom), axis=0)

    def orthogonal_subspace(self, data, A):
        """
        标准化数据集
        :param data: 数据集 [N, D, p]
        :param A: 映射矩阵 [D, d]
        :return:
        """
        D = []
        for d in data:
            q, r = np.linalg.qr(np.dot(np.transpose(A), d))
            try:
                D.append(np.dot(d, np.linalg.inv(r)))
            except np.linalg.LinAlgError:
                D.append(np.dot(d, np.linalg.pinv(r)))
        return np.array(D)

    def transform(self, data):
        """
        对数据集执行降维
        :param data:
        :return:
        """
        embedding = []
        for d in data:
            embedding.append(np.dot(np.transpose(self.components_), d))
        return np.array(embedding)

    def compute_weights(self, dist, n_neighbors=None):
        """
        计算权重矩阵
        :param dist:
        :param n_neighbors:
        :return:
        """
        if n_neighbors is None:
            n_neighbors = dist.shape[0] - 1
        if n_neighbors < 1:
            return np.array([[0]]), np.array([[0]])
        self.nbrs_ = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", algorithm=self.neighbors_algorithm)
        self.nbrs_.fit(dist)
        W = kneighbors_graph(self.nbrs_, n_neighbors, metric="precomputed", mode='distance')
        W.data = np.exp(-W.data ** 2)
        W = W.toarray()
        knn_index = np.where(W > 0)[1]
        knn_index = knn_index.reshape((W.shape[0], n_neighbors))
        return W, knn_index

    def calculate_center(self, data, target):
        """
        计算每个簇的样本中心
        :param data:
        :param target:
        :return:
        """
        centers = []
        for t in np.unique(target):
            data_t = data[t == target]
            s = np.zeros((data_t.shape[1], data_t.shape[1]))
            for d in data_t:
                s += np.dot(d, np.transpose(d))
            eig_values, eig_vectors = np.linalg.eigh(s)
            idx = np.argsort(eig_values)[::-1][:self.p_grassmann]
            centers.append(eig_vectors[:, idx])
        return np.array(centers)

    def optimization(self, data, target, data_with_target, F, W, B, D, knn_index, knn_index_F, d_index, A):
        """
        优化函数
        :param data: 训练数据
        :param target: 标签
        :param data_with_target: 有标签数据
        :param F: 中心点集
        :param W: 全体数据的权重矩阵
        :param B: 中心点集的权重矩阵
        :param D: 标签数据的权重矩阵
        :param knn_index:   全体数据的邻居索引
        :param knn_index_F: 中心点集的邻居索引
        :param d_index:     标签数据的邻居索引
        :param A:           映射矩阵
        :return:
        """
        tj0 = perf_counter()
        J = np.zeros((data.shape[1], data.shape[1]))
        idx = 0
        # 计算标签数据部分
        for c, t in enumerate(np.unique(target)):
            data_t = data_with_target[target == t]
            n_neighbor = data_t.shape[0] - 1
            for i in range(data_t.shape[0]):
                GI = np.dot(data_t[i], data_t[i].T)
                MI = np.dot(data_t[i], data_t[i].T)
                for j in range(n_neighbor):
                    k = d_index[c][i, j]
                    G = GI - np.dot(data_t[k], data_t[k].T)
                    J = J + D[c][i, k] * np.dot(np.dot(np.dot(G, A), A.T), G) * (1 - self.alpha)
                for p in range(self.n_neighbors):
                    k = knn_index[idx, p]
                    M = MI - np.dot(data[k], data[k].T)
                    J = J + W[idx, k] * np.dot(np.dot(np.dot(M, A), A.T), M) * self.alpha
                idx += 1
        # 计算无标签数据部分
        for i in range(idx, data.shape[0]):
            GI = np.dot(data[i], data[i].T)
            for j in range(self.n_neighbors):
                k = knn_index[i, j]
                G = GI - np.dot(data[k], data[k].T)
                J = J + W[i, k] * np.dot(np.dot(np.dot(G, A), A.T), G) * self.alpha
        tj1 = perf_counter()
        temp = tj1 - tj0
        self.JT += temp
        tz0 = perf_counter()
        Z = np.zeros((F.shape[1], F.shape[1]))
        # 计算类间散度矩阵
        for i in range(F.shape[0]):
            SI = np.dot(F[i], F[i].T)
            for j in range(F.shape[0] - 1):
                k = knn_index_F[i, j]
                S = SI - np.dot(F[k], F[k].T)
                Z = Z + B[i, k] * np.dot(np.dot(np.dot(S, A), A.T), S)
        tz1 = perf_counter()
        temp = tz1 - tz0
        self.ST += temp
        # 特征值求解
        eig_values, eig_vectors = scipy.linalg.eig(a=J, b=Z)
        sort_index_ = np.argsort(np.abs(eig_values))
        eig_values = eig_values[sort_index_]
        j = 0
        while np.abs(eig_values[j]) < self.drop_tol:
            j += 1
        index_ = sort_index_[j: j + self.n_components]
        A = eig_vectors[:, index_]
        loss = np.trace(np.dot(np.dot(A.T, J), A)) / np.trace(np.dot(np.dot(A.T, Z), A))
        return A, loss

    def fit(self, data_with_target, data_no_data, target):
        """
        训练过程
        :param data_with_target: 标签数据
        :param data_no_data:     无标签数据
        :param target:           标签
        :return:
        """
        self.coms = []
        data = np.concatenate((data_with_target, data_no_data))
        tf0 = perf_counter()
        F = self.calculate_center(data=data_with_target, target=target)
        tf1 = perf_counter()
        self.FT = tf1 - tf0
        self.components_ = self.init_A(data)
        self.coms.append(self.components_)
        # 计算权重
        tw0 = perf_counter()
        D = []
        d_index = []
        for t in np.unique(target):
            data_t = data_with_target[t == target]
            if data_t.shape[0] > 1:
                dist_t = self.GD.pairwise_dist(data_t, self.metric)
                D_, d_index_ = self.compute_weights(dist_t)
            else:
                D_, d_index_ = np.array([[0]]), np.array([[0]])
            D.append(D_)
            d_index.append(d_index_)
        dist = self.GD.pairwise_dist(data, self.metric)
        W, knn_index = self.compute_weights(dist, n_neighbors=self.n_neighbors)
        tw1 = perf_counter()
        self.WT = tw1 - tw0
        tb0 = perf_counter()
        dist_F = self.GD.pairwise_dist(F, self.metric)
        B, knn_index_F = self.compute_weights(dist_F)
        tb1 = perf_counter()
        self.BT = tb1 - tb0
        self.NT = 0
        self.JT = 0
        self.ST = 0
        epoch = 0
        # 迭代
        while len(self.object_value) <= 1 or np.abs(self.object_value[-1] - self.object_value[-2]) > self.converged_tol:
            tn0 = perf_counter()
            data = self.orthogonal_subspace(data, self.components_)
            tn1 = perf_counter()
            temp = tn1 - tn0
            self.NT += temp
            data_with_target = self.orthogonal_subspace(data_with_target, self.components_)
            self.components_, current_object = self.optimization(
                data, target, data_with_target, F, W, B, D,
                knn_index, knn_index_F, d_index, self.components_)
            self.coms.append(self.components_)
            epoch += 1
            if self.verbose:
                print("第{:d}次迭代的目标值：".format(epoch) + "{:.4f}".format(current_object))
            self.object_value = np.append(self.object_value, current_object)
            if epoch >= self.max_epoch:
                break
        self.NT /= epoch
        self.JT /= epoch
        self.ST /= epoch
        return self

    def fit_transform(self, data, target):
        """
        主函数
        :param data:
        :param target:
        :return:
        """
        self.data = data
        self.target = target
        self.split_data() # 分割训练集和测试集
        # 分割标签数据和无标签数据
        data_train, data_val, target_train, target_val = pp().uniform_sampling(self.data_train, self.target_train)
        self.time_start = perf_counter()
        self.fit(data_train, data_val, target_train)
        self.time_end = perf_counter()
        self.print_time()
        for a in self.coms:
            self.data_train = self.orthogonal_subspace(self.data_train, a)
            self.data_test = self.orthogonal_subspace(self.data_test, a)
        self.embedding_train = self.transform(self.data_train)
        self.embedding_test = self.transform(self.data_test)

    def split_data(self):
        """
        分割训练集和测试集
        :return:
        """
        self.data_train = self.data[self.train_index]
        self.data_test = self.data[self.test_index]
        self.target_train = self.target[self.train_index]
        self.target_test = self.target[self.test_index]

    def print_time(self):
        """
        打印时间
        :return:
        """
        print("\r", "{:8s}".format(self.para[2]),
              "{:8s}".format(self.para[3]),
              "{:8s}".format("time"),
              "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start
################################################################################
class GrassmannFLPP:
    """
    Self-Supervised Locality Preserving Projection on Grassmann Manifold
    """
    def __init__(self):
        pass
