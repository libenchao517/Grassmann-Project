################################################################################
# 本文件用于实现Grassmann NPE及相关算法
################################################################################
# 导入模块
import roman
import scipy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Grassmann import GrassmannDistance
from Grassmann import GrassmannKernel
from time import perf_counter
from DATA import Pre_Procession as pp
from tqdm import tqdm
################################################################################
class GrassmannNPE:
    """
    Neighborhood Preserving Embedding on Grassmann Manifold
    [1] Wei D, Shen X, Sun Q, et al.
    Neighborhood preserving embedding on Grassmann manifold for image-set analysis[J].
    Pattern Recognition, 2022, 122: 108335.
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
            converged_tol=1.0,
            drop_tol=1e-6,
            max_epoch=20,
            mode=1,
            verbose=False,
            func_name="GNPE",
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
        :param mode:                权重矩阵构建方法
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
        self.mode = mode
        self.verbose = verbose
        self.func_name = "-".join([func_name, roman.toRoman(mode)])
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), self.func_name, data_name, ""]
        self.return_time = return_time
        self.time = None
        self.GD = GrassmannDistance()
        self.GK = GrassmannKernel()
        self.metric = self.GD.gdist
        self.object_value = np.array([])

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
        for x in data:
            embedding.append(np.dot(np.transpose(self.components_), x))
        return np.array(embedding)

    def Calculate_K_NN(self, data, n_neighbors=None):
        """
        训练数据集的k近邻
        :param data:
        :param n_neighbors:
        :return:
        """
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        dist = self.GD.pairwise_dist(data, self.metric)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", algorithm=self.neighbors_algorithm)
        KNN = knn.fit(dist)
        knn_dist, knn_index = KNN.kneighbors(n_neighbors=n_neighbors)
        return knn_dist, knn_index, dist

    def Calculate_W_first(self, data, knn_index, n_neighbors=None, lamda=0.01):
        """
        第一种权重矩阵构建方法
        :param data: 数据集 [N, D, p]
        :param knn_index: 邻居索引
        :param n_neighbors: 近邻数
        :param lamda: 超参数
        :return:
        """
        W = []
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        delta = self.GK.pairwise_kernel(data, self.GK.projection_kernel)
        I = np.ones((n_neighbors, 1))
        for i in range(data.shape[0]):
            dist_i = delta[knn_index[i]][:, knn_index[i]]
            U, S, V = np.linalg.svd(dist_i)
            if np.isclose(S, 0).any():
                S = S + 0.001
            Di = np.dot(np.diag(np.sqrt(S)), V)
            Ti = np.zeros((n_neighbors, 1))
            for j in range(n_neighbors):
                Ti[j] = np.square(np.linalg.norm(data[i] * data[knn_index[i, j]]))
            yi = np.dot(np.dot(np.diag(np.power(S, -0.5)), V), Ti).reshape((-1, 1))
            Hi = np.dot(Di.T-np.dot(I, yi.T), (Di.T-np.dot(I, yi.T)).T)
            ci = np.linalg.solve(Hi+lamda*np.eye(*Hi.shape), I)
            c = ci / np.dot(I.T, ci)
            W.append(c.flatten())
        return np.array(W)

    def Calculate_W_second(self, data, n_neighbors=None, miu=0.01):
        """
        第二种权重矩阵构建方法
        :param data: 数据集 [N, D, p]
        :param n_neighbors: 近邻数
        :param miu: 超参数
        :return:
        """
        W = []
        knn_index = []
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        delta = self.GK.pairwise_kernel(data, self.GK.projection_kernel)
        P = np.linalg.inv(delta + miu*np.eye(len(data)))
        et = np.eye(len(data))
        for i in range(data.shape[0]):
            mol = np.dot(np.dot(np.dot(et[i], P), delta[:, i]), et[:, i])
            den = np.dot(np.dot(et[i], P), et[:, i])
            c = np.dot(P, delta[:, i]-mol/den)
            indices_ = np.argpartition(c, -n_neighbors)[-n_neighbors:]
            W.append(c[indices_])
            knn_index.append(indices_)
        return np.array(W), np.array(knn_index)

    def optimization(self, data, W, A, knn_index):
        """
        优化函数
        :param data: 数据集 [N, D, p]
        :param W:    权重矩阵
        :param A:    映射矩阵
        :param knn_index: 邻居索引
        :return:
        """
        J = np.zeros((data.shape[1], data.shape[1]))
        F = np.zeros((data.shape[1], data.shape[1]))
        for i in tqdm(range(len(data))):
            G = np.dot(data[i], data[i].T)
            F = F + G
            for j in range(self.n_neighbors):
                k = knn_index[i, j]
                G = G - W[i, j] * np.dot(data[k], data[k].T)
            J = J + np.dot(np.dot(np.dot(G, A), A.T), G)
        eig_values, eig_vectors = scipy.linalg.eig(a=J, b=F)
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
        if self.mode == 1:
            knn_dist, knn_index, dist = self.Calculate_K_NN(data, n_neighbors=self.n_neighbors)
            W = self.Calculate_W_first(data, knn_index, n_neighbors=self.n_neighbors)
        elif self.mode == 2:
            W, knn_index = self.Calculate_W_second(data, n_neighbors=self.n_neighbors)
        epoch = 0
        while len(self.object_value) <= 1 or np.abs(self.object_value[-1] - self.object_value[-2]) > self.converged_tol:
            data = self.orthogonal_subspace(data, self.components_)
            self.components_, current_object = self.optimization(data, W, self.components_, knn_index)
            self.coms.append(self.components_)
            epoch += 1
            if self.verbose:
                print("第{:d}次迭代的目标值：".format(epoch) + "{:.4f}".format(current_object))
            self.object_value = np.append(self.object_value, current_object)
            if epoch >= self.max_epoch:
                break
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
        self.fit(self.data_train)
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
        划分训练集和测试集
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
class GrassmannDNPE:
    """
    Discriminant Neighborhood Preserving Embedding on Grassmann Manifold
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
            converged_tol=1.0,
            drop_tol=1e-6,
            max_epoch=20,
            mode=1,
            verbose=False,
            func_name="GDNPE",
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
        :param mode:                权重矩阵构建方法
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
        self.mode = mode
        self.verbose = verbose
        self.func_name = "-".join([func_name, roman.toRoman(mode)])
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), self.func_name, data_name, ""]
        self.return_time = return_time
        self.time = None
        self.GD = GrassmannDistance()
        self.GK = GrassmannKernel()
        self.metric = self.GD.gdist
        self.object_value = np.array([])

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
        for x in data:
            embedding.append(np.dot(np.transpose(self.components_), x))
        return np.array(embedding)

    def Calculate_K_NN(self, data, n_neighbors=None):
        """
        训练数据集的k近邻
        :param data:
        :param n_neighbors:
        :return:
        """
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        dist = self.GD.pairwise_dist(data, self.metric)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", algorithm=self.neighbors_algorithm)
        KNN = knn.fit(dist)
        knn_dist, knn_index = KNN.kneighbors(n_neighbors=n_neighbors)
        return knn_dist, knn_index, dist

    def Calculate_W_first(self, data, knn_index, n_neighbors=None, lamda=0.01):
        """
        第一种权重矩阵构建方法
        :param data: 数据集 [N, D, p]
        :param knn_index: 邻居索引
        :param n_neighbors: 近邻数
        :param lamda: 超参数
        :return:
        """
        W = []
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        delta = self.GK.pairwise_kernel(data, self.GK.projection_kernel)
        I = np.ones((n_neighbors, 1))
        for i in range(data.shape[0]):
            dist_i = delta[knn_index[i]][:, knn_index[i]]
            U, S, V = np.linalg.svd(dist_i)
            if np.isclose(S, 0).any():
                S = S + 0.001
            Di = np.dot(np.diag(np.sqrt(S)), V)
            Ti = np.zeros((n_neighbors, 1))
            for j in range(n_neighbors):
                Ti[j] = np.square(np.linalg.norm(data[i] * data[knn_index[i, j]]))
            yi = np.dot(np.dot(np.diag(np.power(S, -0.5)), V), Ti).reshape((-1, 1))
            Hi = np.dot(Di.T - np.dot(I, yi.T), (Di.T - np.dot(I, yi.T)).T)
            ci = np.linalg.solve(Hi + lamda * np.eye(*Hi.shape), I)
            c = ci / np.dot(I.T, ci)
            W.append(c.flatten())
        return np.array(W)

    def Calculate_W_second(self, data, n_neighbors=None, miu=0.01):
        """
        第二种权重矩阵构建方法
        :param data: 数据集 [N, D, p]
        :param n_neighbors: 近邻数
        :param miu: 超参数
        :return:
        """
        W = []
        knn_index = []
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        delta = self.GK.pairwise_kernel(data, self.GK.projection_kernel)
        P = np.linalg.inv(delta + miu*np.eye(len(data)))
        et = np.eye(len(data))
        for i in range(data.shape[0]):
            mol = np.dot(np.dot(np.dot(et[i], P), delta[:, i]), et[:, i])
            den = np.dot(np.dot(et[i], P), et[:, i])
            c = np.dot(P, delta[:, i]-mol/den)
            indices_ = np.argpartition(c, -n_neighbors)[-n_neighbors:]
            W.append(c[indices_])
            knn_index.append(indices_)
        return np.array(W), np.array(knn_index)

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
        :param F:    中心点集
        :param W:    类内权重矩阵
        :param B:    类间权重矩阵
        :param knn_index:    类内邻居索引
        :param knn_index_F:  类间邻居索引
        :param A:    映射矩阵
        :return:
        """
        J = np.zeros((data.shape[1], data.shape[1]))
        for c, t in enumerate(np.unique(target)):
            data_t = data[target == t]
            n_neighbors = data_t.shape[0] - 1
            for i in range(data_t.shape[0]):
                G = np.dot(data_t[i], data_t[i].T)
                for j in range(n_neighbors):
                    k = knn_index[c][i, j]
                    G = G - W[c][i, j] * np.dot(data_t[k], data_t[k].T)
                J = J + np.dot(np.dot(np.dot(G, A), A.T), G)
        Z = np.zeros((F.shape[1], F.shape[1]))
        for i in range(F.shape[0]):
            S = np.dot(F[i], F[i].T)
            for j in range(F.shape[0] - 1):
                k = knn_index_F[i, j]
                S = S - B[i, j] * np.dot(F[k], F[k].T)
            Z = Z + np.dot(np.dot(np.dot(S, A), A.T), S)
        eig_values, eig_vectors = scipy.linalg.eig(a=J, b=Z)
        sort_index_ = np.argsort(np.abs(eig_values))
        eig_values = eig_values[sort_index_]
        j = 0
        while np.abs(eig_values[j]) < self.drop_tol:
            j += 1
        index_ = sort_index_[j: j + self.n_components]
        A = eig_vectors[:, index_]
        return A, np.trace(np.dot(np.dot(A.T, J), A))

    def fit(self, data, target):
        """
        训练过程
        :param data: 数据集
        :param target: 标签
        :return:
        """
        self.coms = []
        F = self.calculate_center(data=data, target=target)
        self.components_ = self.init_A(data)
        self.coms.append(self.components_)
        knn_index = []
        W = []
        if self.mode == 1:
            for t in np.unique(target):
                data_t = data[target == t]
                if data_t.shape[0] > 1:
                    _, index_, _ = self.Calculate_K_NN(data_t)
                    W_ = self.Calculate_W_first(data_t, index_)
                else:
                    W_, index_ = np.array([[0]]), np.array([[0]])
                knn_index.append(index_)
                W.append(W_)
            _, knn_index_F, _ = self.Calculate_K_NN(F)
            B = self.Calculate_W_first(F, knn_index_F)
        elif self.mode == 2:
            for t in np.unique(target):
                data_t = data[target == t]
                if data_t.shape[0] > 1:
                    W_, index_ = self.Calculate_W_second(data_t)
                else:
                    W_, index_ = np.array([[0]]), np.array([[0]])
                knn_index.append(index_)
                W.append(W_)
            B, knn_index_F = self.Calculate_W_second(F)
        epoch = 0
        while len(self.object_value) <= 1 or np.abs(self.object_value[-1] - self.object_value[-2]) > self.converged_tol:
            data = self.orthogonal_subspace(data, self.components_)
            self.components_, current_object = self.optimization(data, target, F, W, B, knn_index, knn_index_F, self.components_)
            self.coms.append(self.components_)
            epoch += 1
            if self.verbose:
                print("第{:d}次迭代的目标值：".format(epoch) + "{:.4f}".format(current_object))
            self.object_value = np.append(self.object_value, current_object)
            self.Normal_data = data
            if epoch >= self.max_epoch:
                break
        return self

    def fit_transform(self, data, target):
        """
        主函数
        :param data: 数据集
        :param target: 标签
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
        划分训练集和测试集
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
class GrassmannSNPE:
    """
    Semi-Supervised Neighborhood Preserving Embedding on Grassmann Manifold
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
            converged_tol=1.0,
            drop_tol=1e-6,
            max_epoch=20,
            mode=1,
            alpha=1.0,
            verbose=False,
            func_name="GDNPE",
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
        :param mode:                权重矩阵构建方法
        :param alpha:               GSNPE的正则化系数
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
        self.mode = mode
        self.alpha = alpha
        self.verbose = verbose
        self.func_name = "-".join([func_name, roman.toRoman(mode)])
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), self.func_name, data_name, ""]
        self.return_time = return_time
        self.time = None
        self.GD = GrassmannDistance()
        self.GK = GrassmannKernel()
        self.metric = self.GD.gdist
        self.object_value = np.array([])

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
        for x in data:
            embedding.append(np.dot(np.transpose(self.components_), x))
        return np.array(embedding)

    def Calculate_K_NN(self, data, n_neighbors=None):
        """
        训练数据集的k近邻
        :param data:
        :param n_neighbors:
        :return:
        """
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        dist = self.GD.pairwise_dist(data, self.metric)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", algorithm=self.neighbors_algorithm)
        KNN = knn.fit(dist)
        knn_dist, knn_index = KNN.kneighbors(n_neighbors=n_neighbors)
        return knn_dist, knn_index, dist

    def Calculate_W_first(self, data, knn_index, n_neighbors=None, lamda=0.01):
        """
        第一种权重矩阵构建方法
        :param data: 数据集 [N, D, p]
        :param knn_index: 邻居索引
        :param n_neighbors: 近邻数
        :param lamda: 超参数
        :return:
        """
        W = []
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        delta = self.GK.pairwise_kernel(data, self.GK.projection_kernel)
        I = np.ones((n_neighbors, 1))
        for i in range(data.shape[0]):
            dist_i = delta[knn_index[i]][:, knn_index[i]]
            U, S, V = np.linalg.svd(dist_i)
            if np.isclose(S, 0).any():
                S = S + 0.001
            Di = np.dot(np.diag(np.sqrt(S)), V)
            Ti = np.zeros((n_neighbors, 1))
            for j in range(n_neighbors):
                Ti[j] = np.square(np.linalg.norm(data[i] * data[knn_index[i, j]]))
            yi = np.dot(np.dot(np.diag(np.power(S, -0.5)), V), Ti).reshape((-1, 1))
            Hi = np.dot(Di.T - np.dot(I, yi.T), (Di.T - np.dot(I, yi.T)).T)
            ci = np.linalg.solve(Hi + lamda * np.eye(*Hi.shape), I)
            c = ci / np.dot(I.T, ci)
            W.append(c.flatten())
        return np.array(W)

    def Calculate_W_second(self, data, n_neighbors=None, miu=0.01):
        """
        第二种权重矩阵构建方法
        :param data: 数据集 [N, D, p]
        :param n_neighbors: 近邻数
        :param miu: 超参数
        :return:
        """
        W = []
        knn_index = []
        if n_neighbors is None:
            n_neighbors = data.shape[0] - 1
        delta = self.GK.pairwise_kernel(data, self.GK.projection_kernel)
        P = np.linalg.inv(delta + miu*np.eye(len(data)))
        et = np.eye(len(data))
        for i in range(data.shape[0]):
            mol = np.dot(np.dot(np.dot(et[i], P), delta[:, i]), et[:, i])
            den = np.dot(np.dot(et[i], P), et[:, i])
            c = np.dot(P, delta[:, i]-mol/den)
            indices_ = np.argpartition(c, -n_neighbors)[-n_neighbors:]
            W.append(c[indices_])
            knn_index.append(indices_)
        return np.array(W), np.array(knn_index)

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
        :param data:             训练数据
        :param target:           标签
        :param data_with_target: 有标签数据
        :param F:  中心点集
        :param W:  训练数据的权重矩阵
        :param B:  类间权重矩阵
        :param D:  类内权重矩阵
        :param knn_index: 训练数据邻居索引
        :param knn_index_F: 类间邻居索引
        :param d_index:     类内邻居索引
        :param A: 映射矩阵
        :return:
        """
        J = np.zeros((data.shape[1], data.shape[1]))
        idx = 0
        # 计算有标签数据部分
        for c, t in enumerate(np.unique(target)):
            data_t = data_with_target[target == t]
            n_neighbor = data_t.shape[0] - 1
            for i in range(data_t.shape[0]):
                G = np.dot(data_t[i], data_t[i].T)
                M = np.dot(data_t[i], data_t[i].T)
                for j in range(n_neighbor):
                    k = d_index[c][i, j]
                    G = G - D[c][i, j] * np.dot(data_t[k], data_t[k].T)
                J = J + np.dot(np.dot(np.dot(G, A), A.T), G) * self.alpha
                for p in range(self.n_neighbors):
                    k = knn_index[idx, p]
                    M = M - W[idx, p] * np.dot(data[k], data[k].T)
                J = J + np.dot(np.dot(np.dot(M, A), A.T), M) * (1 - self.alpha)
                idx += 1
        # 计算无标签数据
        for i in range(idx, len(data)):
            G = np.dot(data[i], data[i].T)
            for j in range(self.n_neighbors):
                k = knn_index[i, j]
                G = G - W[i, j] * np.dot(data[k], data[k].T)
            J = J + np.dot(np.dot(np.dot(G, A), A.T), G) * (1 - self.alpha)
        # 计算中心点集
        Z = np.zeros((F.shape[1], F.shape[1]))
        for i in range(F.shape[0]):
            S = np.dot(F[i], F[i].T)
            for j in range(F.shape[0] - 1):
                k = knn_index_F[i, j]
                S = S - B[i, j] * np.dot(F[k], F[k].T)
            Z = Z + np.dot(np.dot(np.dot(S, A), A.T), S)
        # 求解特征值问题
        eig_values, eig_vectors = scipy.linalg.eig(a=J, b=Z)
        sort_index_ = np.argsort(np.abs(eig_values))
        eig_values = eig_values[sort_index_]
        j = 0
        while np.abs(eig_values[j]) < self.drop_tol:
            j += 1
        index_ = sort_index_[j: j + self.n_components]
        A = eig_vectors[:, index_]
        return A, np.trace(np.dot(np.dot(A.T, J), A))

    def fit(self, data_with_target, data_no_data, target):
        """
        训练过程
        :param data_with_target: 标签数据
        :param data_no_data:     无标签数据
        :param target:           数据标签
        :return:
        """
        self.coms = []
        data = np.concatenate((data_with_target, data_no_data))
        F = self.calculate_center(data=data_with_target, target=target)
        self.components_ = self.init_A(data_with_target)
        self.coms.append(self.components_)
        D = []
        d_index = []
        if self.mode == 1:
            for t in np.unique(target):
                data_t = data_with_target[target == t]
                if data_t.shape[0] > 1:
                    _, d_index_, _ = self.Calculate_K_NN(data_t)
                    D_ = self.Calculate_W_first(data_t, d_index_)
                else:
                    D_, d_index_ = np.array([[0]]), np.array([[0]])
                d_index.append(d_index_)
                D.append(D_)
            _, knn_index, _ = self.Calculate_K_NN(data, n_neighbors=self.n_neighbors)
            W = self.Calculate_W_first(data, knn_index, n_neighbors=self.n_neighbors)
            _, knn_index_F, _ = self.Calculate_K_NN(F)
            B = self.Calculate_W_first(F, knn_index_F)
        elif self.mode == 2:
            for t in np.unique(target):
                data_t = data_with_target[target == t]
                if data_t.shape[0] > 1:
                    D_, d_index_ = self.Calculate_W_second(data_t)
                else:
                    D_,d_index_ = np.array([[0]]), np.array([[0]])
                d_index.append(d_index_)
                D.append(D_)
            W, knn_index = self.Calculate_W_second(data, n_neighbors=self.n_neighbors)
            B, knn_index_F = self.Calculate_W_second(F)
        epoch = 0
        while len(self.object_value) <= 1 or np.abs(self.object_value[-1] - self.object_value[-2]) > self.converged_tol:
            data = self.orthogonal_subspace(data, self.components_)
            data_with_target = self.orthogonal_subspace(data_with_target, self.components_)
            self.components_, current_object = self.optimization(data, target, data_with_target, F, W, B, D, knn_index, knn_index_F, d_index, self.components_)
            self.coms.append(self.components_)
            epoch += 1
            if self.verbose:
                print("第{:d}次迭代的目标值：".format(epoch) + "{:.4f}".format(current_object))
            self.object_value = np.append(self.object_value, current_object)
            if epoch >= self.max_epoch:
                break
        return self

    def fit_transform(self, data, target):
        """
        主函数
        :param data:   数据集 [N, D, p]
        :param target:
        :return:
        """
        self.data = data
        self.target = target
        self.split_data()
        data_train, data_val, target_train, target_val = pp().uniform_sampling(self.data_train, self.target_train, train_size=0.5)
        self.time_start = perf_counter()
        self.fit(data_train, data_val, target_train)
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
        划分训练集和测试集
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
class GrassmannFNPE:
    """
    Self-Supervised Neighborhood Preserving Embedding on Grassmann Manifold
    """
    def __init__(self):
        pass
