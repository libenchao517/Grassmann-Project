################################################################################
# 本文件用于实现Grassmann UMAP算法
################################################################################
# 导入模块
from __future__ import print_function
import locale
from warnings import warn
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from time import perf_counter
import joblib
import numpy as np
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph
import numba
import GUMAP.distances as dist
import GUMAP.sparse as sparse
from GUMAP.utils import fast_knn_indices
from GUMAP.layouts import optimize_layout_grassmann
from pynndescent import NNDescent
from pynndescent.distances import named_distances as pynn_named_distances
from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances
from Grassmann import GrassmannDistance
################################################################################
# 定义必要常量
locale.setlocale(locale.LC_NUMERIC, "C")
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf
DISCONNECTION_DISTANCES = {"correlation": 2, "cosine": 2, "hellinger": 1, "jaccard": 1, "dice": 1}
################################################################################
# 定义必要函数
def raise_disconnected_warning(edges_removed, vertices_disconnected, disconnection_distance, total_rows, threshold=0.1):
    if (vertices_disconnected > 0) & (vertices_disconnected <= threshold * total_rows):
        warn(
            f"A few of your vertices were disconnected from the manifold.  This shouldn't cause problems.\n"
            f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\n"
            f"It has only fully disconnected {vertices_disconnected} vertices.\n"
            f"Use umap.utils.disconnected_vertices() to identify them.",)
    elif vertices_disconnected > threshold * total_rows:
        warn(
            f"A large number of your vertices were disconnected from the manifold.\n"
            f"Disconnection_distance = {disconnection_distance} has removed {edges_removed} edges.\n"
            f"It has fully disconnected {vertices_disconnected} vertices.\n"
            f"You might consider using find_disconnected_points() to find and remove these points from your data.\n"
            f"Use umap.utils.disconnected_vertices() to identify them.",)

@numba.njit(locals={"psum": numba.types.float32, "lo": numba.types.float32, "mid": numba.types.float32, "hi": numba.types.float32}, fastmath=True)
def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)
    mean_distances = np.mean(distances)
    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1])
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)
        for n in range(n_iter):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0
            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0
        result[i] = mid
        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances
    return result, rho

def nearest_neighbors(X, n_neighbors, metric, metric_kwds, random_state, low_memory=True, n_jobs=-1):
    if metric == "precomputed":
        knn_indices = fast_knn_indices(X, n_neighbors)
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        disconnected_index = knn_dists == np.inf
        knn_indices[disconnected_index] = -1
        knn_search_index = None
    else:
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        knn_search_index = NNDescent(X, n_neighbors=n_neighbors, metric=metric, metric_kwds=metric_kwds, random_state=random_state, n_trees=n_trees, n_iters=n_iters, max_candidates=60, low_memory=low_memory, n_jobs=n_jobs, compressed=False)
        knn_indices, knn_dists = knn_search_index.neighbor_graph
    return knn_indices, knn_dists, knn_search_index

@numba.njit(locals={"knn_dists": numba.types.float32[:, ::1], "sigmas": numba.types.float32[::1], "rhos": numba.types.float32[::1], "val": numba.types.float32}, parallel=True, fastmath=True)
def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos, return_dists=False, bipartite=False):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None
    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]
    return rows, cols, vals, dists

def fuzzy_simplicial_set(X, n_neighbors, random_state, metric, metric_kwds={}, knn_indices=None, knn_dists=None, set_op_mix_ratio=1.0, local_connectivity=1.0, apply_set_operations=True, return_dists=None):
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(X, n_neighbors, metric, metric_kwds, random_state)
    knn_dists = knn_dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity))
    rows, cols, vals, dists = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos, return_dists)
    result = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    result.eliminate_zeros()
    if apply_set_operations:
        transpose = result.transpose()
        prod_matrix = result.multiply(transpose)
        result = (set_op_mix_ratio * (result + transpose - prod_matrix) + (1.0 - set_op_mix_ratio) * prod_matrix)
    result.eliminate_zeros()
    if return_dists is None:
        return result, sigmas, rhos
    else:
        if return_dists:
            dmat = scipy.sparse.coo_matrix((dists, (rows, cols)), shape=(X.shape[0], X.shape[0]))
            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None
        return result, sigmas, rhos, dists

@numba.njit()
def fast_intersection(rows, cols, values, target, unknown_dist=1.0, far_dist=5.0):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform Data_size fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array
        An array of the value of each non-zero in the sparse matrix
        representation.

    target: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist float (optional, default 5.0)
        The distance between unmatched labels.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        if (target[i] == -1) or (target[j] == -1):
            values[nz] *= np.exp(-unknown_dist)
        elif target[i] != target[j]:
            values[nz] *= np.exp(-far_dist)
    return

@numba.njit()
def fast_metric_intersection(rows, cols, values, discrete_space, metric, metric_args, scale):
    """Under the assumption of categorical distance for the intersecting
    simplicial set perform Data_size fast intersection.

    Parameters
    ----------
    rows: array
        An array of the row of each non-zero in the sparse matrix
        representation.

    cols: array
        An array of the column of each non-zero in the sparse matrix
        representation.

    values: array of shape
        An array of the values of each non-zero in the sparse matrix
        representation.

    discrete_space: array of shape (n_samples, n_features)
        The vectors of categorical labels to use in the intersection.

    metric: numba function
        The function used to calculate distance over the target array.

    scale: float
        A scaling to apply to the metric.

    Returns
    -------
    None
    """
    for nz in range(rows.shape[0]):
        i = rows[nz]
        j = cols[nz]
        dist = metric(discrete_space[i], discrete_space[j], *metric_args)
        values[nz] *= np.exp(-(scale * dist))
    return

@numba.njit()
def reprocess_row(probabilities, k=15, n_iters=32):
    target = np.log2(k)
    lo = 0.0
    hi = NPY_INFINITY
    mid = 1.0

    for n in range(n_iters):

        psum = 0.0
        for j in range(probabilities.shape[0]):
            psum += pow(probabilities[j], mid)

        if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
            break

        if psum < target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == NPY_INFINITY:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0

    return np.power(probabilities, mid)

@numba.njit()
def reset_local_metrics(simplicial_set_indptr, simplicial_set_data):
    for i in range(simplicial_set_indptr.shape[0] - 1):
        simplicial_set_data[
            simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]
        ] = reprocess_row(
            simplicial_set_data[simplicial_set_indptr[i] : simplicial_set_indptr[i + 1]]
        )
    return

def reset_local_connectivity(simplicial_set, reset_local_metric=False):
    """Reset the local connectivity requirement -- each data sample should
    have complete confidence in at least one 1-simplex in the simplicial set.
    We can enforce this by locally rescaling confidences, and then remerging the
    different local simplicial sets together.

    Parameters
    ----------
    simplicial_set: sparse matrix
        The simplicial set for which to recalculate with respect to local
        connectivity.

    Returns
    -------
    simplicial_set: sparse_matrix
        The recalculated simplicial set, now with the local connectivity
        assumption restored.
    """
    simplicial_set = normalize(simplicial_set, norm="max")
    if reset_local_metric:
        simplicial_set = simplicial_set.tocsr()
        reset_local_metrics(simplicial_set.indptr, simplicial_set.data)
        simplicial_set = simplicial_set.tocoo()
    transpose = simplicial_set.transpose()
    prod_matrix = simplicial_set.multiply(transpose)
    simplicial_set = simplicial_set + transpose - prod_matrix
    simplicial_set.eliminate_zeros()
    return simplicial_set

def discrete_metric_simplicial_set_intersection(simplicial_set, discrete_space, unknown_dist=1.0, far_dist=5.0, metric=None, metric_kws={}, metric_scale=1.0):
    """Combine Data_size fuzzy simplicial set with another fuzzy simplicial set
    generated from discrete metric data using discrete distances. The target
    data is assumed to be categorical label data (Data_size vector of labels),
    and this will update the fuzzy simplicial set to respect that label data.

    TODO: optional category cardinality based weighting of distance

    Parameters
    ----------
    simplicial_set: sparse matrix
        The input fuzzy simplicial set.

    discrete_space: array of shape (n_samples)
        The categorical labels to use in the intersection.

    unknown_dist: float (optional, default 1.0)
        The distance an unknown label (-1) is assumed to be from any point.

    far_dist: float (optional, default 5.0)
        The distance between unmatched labels.

    metric: str (optional, default None)
        If not None, then use this metric to determine the
        distance between values.

    metric_scale: float (optional, default 1.0)
        If using Data_size custom metric scale the distance values by
        this value -- this controls the weighting of the
        intersection. Larger values weight more toward target.

    Returns
    -------
    simplicial_set: sparse matrix
        The resulting intersected fuzzy simplicial set.
    """
    simplicial_set = simplicial_set.tocoo()

    if metric is not None:
        # We presume target is now Data_size 2d array, with each row being Data_size
        # vector of target info
        if metric in dist.named_distances:
            metric_func = dist.named_distances[metric]
        else:
            raise ValueError("Discrete intersection metric is not recognized")

        fast_metric_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            metric_func,
            tuple(metric_kws.values()),
            metric_scale,
        )
    else:
        fast_intersection(
            simplicial_set.row,
            simplicial_set.col,
            simplicial_set.data,
            discrete_space,
            unknown_dist,
            far_dist,
        )
    simplicial_set.eliminate_zeros()
    return reset_local_connectivity(simplicial_set)

def general_simplicial_set_intersection(simplicial_set1, simplicial_set2, weight=0.5, right_complement=False):
    if right_complement:
        result = simplicial_set1.tocoo()
    else:
        result = (simplicial_set1 + simplicial_set2).tocoo()
    left = simplicial_set1.tocsr()
    right = simplicial_set2.tocsr()

    sparse.general_sset_intersection(
        left.indptr,
        left.indices,
        left.data,
        right.indptr,
        right.indices,
        right.data,
        result.row,
        result.col,
        result.data,
        mix_weight=weight,
        right_complement=right_complement)
    return result

def make_epochs_per_sample(weights, n_epochs):
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result

def noisy_scale_coords(coords, random_state, max_coord=10.0, noise=0.0001):
    expansion = max_coord / np.abs(coords).max()
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(np.float32)

def simplicial_set_embedding(
        data, graph, n_components, initial_alpha, a, b, gamma,
        negative_sample_rate, n_epochs, init, random_state,
        metric, metric_kwds,
        output_metric=dist.named_distances_with_gradients["euclidean"],
        output_metric_kwds={}, euclidean_output=True, parallel=False,
        p_grassmann=10
):
    """
    优化低维嵌入
    :param data:
    :param graph:
    :param n_components: 目标维度
    :param initial_alpha:
    :param a:
    :param b:
    :param gamma:
    :param negative_sample_rate:
    :param n_epochs:
    :param init:
    :param random_state:
    :param metric:
    :param metric_kwds:
    :param output_metric:
    :param output_metric_kwds:
    :param euclidean_output:
    :param parallel:
    :param p_grassmann: 子空间阶数
    :return:
    """
    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200
    if n_epochs is None:
        n_epochs = default_epochs
    n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else n_epochs
    if n_epochs_max > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0
    graph.eliminate_zeros()
    # 初始化低维数据
    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0,
            size=(data.shape[0], n_components, p_grassmann))
        for i in range(len(embedding)):
            Q, R = np.linalg.qr(embedding[i])
            embedding[i] = Q
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs_max)
    head = graph.row
    tail = graph.col
    weight = graph.data
    embedding = embedding.astype(np.float32)
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    if euclidean_output:
        embedding = optimize_layout_grassmann(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            move_other=True)
    return embedding

def find_ab_params(spread, min_dist):
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]
################################################################################
class GrassmannUMAP(BaseEstimator):
    """
    Uniform Manifold Approximation and Projection on Grassmann Manifold
    """
    def __init__(
        self,
        n_neighbors=15,
        n_components=10,
        p_grassmann=10,
        metric="precomputed",
        metric_kwds=None,
        output_metric="euclidean",
        output_metric_kwds=None,
        n_epochs=None,
        learning_rate=1.0,
        init="random",
        min_dist=0.1,
        spread=1.0,
        low_memory=True,
        n_jobs=-1,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        a=None,
        b=None,
        random_state=None,
        target_n_neighbors=-1,
        target_metric="categorical",
        target_metric_kwds=None,
        target_weight=0.5,
        transform_mode="embedding",
        force_approximation_algorithm=False,
        disconnection_distance=None,
        precomputed_knn=(None, None, None),
        train_size=0.5,
        func_name="GUMAP",
        data_name="ETH-80",
        sec_part="Experiment",
        sec_num=1,
        return_time=True
    ):
        """
        初始化函数
        :param n_neighbors:
        :param n_components:
        :param p_grassmann: 子空间阶数
        :param metric:
        :param metric_kwds:
        :param output_metric:
        :param output_metric_kwds:
        :param n_epochs:
        :param learning_rate:
        :param init:
        :param min_dist:
        :param spread:
        :param low_memory:
        :param n_jobs:
        :param set_op_mix_ratio:
        :param local_connectivity:
        :param repulsion_strength:
        :param negative_sample_rate:
        :param a:
        :param b:
        :param random_state:
        :param target_n_neighbors:
        :param target_metric:
        :param target_metric_kwds:
        :param target_weight:
        :param transform_mode:
        :param force_approximation_algorithm:
        :param disconnection_distance:
        :param precomputed_knn:
        :param train_size:
        :param func_name:
        :param data_name:
        :param sec_part:
        :param sec_num:
        :param return_time:
        """
        self.n_neighbors = n_neighbors
        self.p_grassmann = p_grassmann
        self.n_components = np.maximum(n_components, p_grassmann)
        self.metric = metric
        self.output_metric = output_metric
        self.metric_kwds = metric_kwds
        self.output_metric_kwds = output_metric_kwds
        self.n_epochs = n_epochs
        self.init = init
        self.repulsion_strength = repulsion_strength
        self.learning_rate = learning_rate
        self.spread = spread
        self.min_dist = min_dist
        self.low_memory = low_memory
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.target_n_neighbors = target_n_neighbors
        self.target_metric = target_metric
        self.target_metric_kwds = target_metric_kwds
        self.target_weight = target_weight
        self.transform_mode = transform_mode
        self.force_approximation_algorithm = force_approximation_algorithm
        self.disconnection_distance = disconnection_distance
        self.precomputed_knn = precomputed_knn
        self.n_jobs = n_jobs
        self.a = a
        self.b = b
        self.train_size = train_size
        self.func_name = func_name
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, ""]
        self.return_time = return_time
        self.time = None
        self.GD = GrassmannDistance()

    def _validate_parameters(self):
        if self.set_op_mix_ratio < 0.0 or self.set_op_mix_ratio > 1.0:
            raise ValueError("set_op_mix_ratio must be between 0.0 and 1.0")
        if self.repulsion_strength < 0.0:
            raise ValueError("repulsion_strength cannot be negative")
        if self.min_dist > self.spread:
            raise ValueError("min_dist must be less than or equal to spread")
        if self.min_dist < 0.0:
            raise ValueError("min_dist cannot be negative")
        if not isinstance(self.init, str) and not isinstance(self.init, np.ndarray):
            raise ValueError("init must be Data_size string or ndarray")
        if isinstance(self.init, str) and self.init not in ("pca", "spectral", "random", "tswspectral"):
            raise ValueError('string init values must be one of: "pca", "tswspectral", "spectral" or "random"')
        if (isinstance(self.init, np.ndarray) and self.init.shape[1] != self.n_components):
            raise ValueError("init ndarray must match n_components value")
        if not isinstance(self.metric, str) and not callable(self.metric):
            raise ValueError("metric must be string or callable")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self._initial_alpha < 0.0:
            raise ValueError("learning_rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be Data_size whole number")
            try:
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        self.n_epochs_list = None
        if (isinstance(self.n_epochs, list) or isinstance(self.n_epochs, tuple) or isinstance(self.n_epochs, np.ndarray)):
            if not issubclass(np.array(self.n_epochs).dtype.type, np.integer) or not np.all(np.array(self.n_epochs) >= 0):
                raise ValueError("n_epochs must be Data_size nonnegative integer or Data_size list of nonnegative integers")
            self.n_epochs_list = list(self.n_epochs)
        elif self.n_epochs is not None and (self.n_epochs < 0 or not isinstance(self.n_epochs, int)):
            raise ValueError("n_epochs must be Data_size nonnegative integer or Data_size list of nonnegative integers")
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
        if self.output_metric_kwds is None:
            self._output_metric_kwds = {}
        else:
            self._output_metric_kwds = self.output_metric_kwds
        if self.target_metric_kwds is None:
            self._target_metric_kwds = {}
        else:
            self._target_metric_kwds = self.target_metric_kwds
        if scipy.sparse.isspmatrix_csr(self._raw_data):
            self._sparse_data = True
        else:
            self._sparse_data = False
        if callable(self.metric):
            in_returns_grad = self._check_custom_metric(self.metric, self._metric_kwds, self._raw_data)
            if in_returns_grad:
                _m = self.metric
                @numba.njit(fastmath=True)
                def _dist_only(x, y, *kwds):
                    return _m(x, y, *kwds)[0]
                self._input_distance_func = _dist_only
            else:
                self._input_distance_func = self.metric
        elif self.metric == "precomputed":
            self._input_distance_func = self.metric
        elif self.metric == "hellinger" and self._raw_data.min() < 0:
            raise ValueError("Metric 'hellinger' does not support negative values")
        elif self.metric in dist.named_distances:
            if self._sparse_data:
                if self.metric in sparse.sparse_named_distances:
                    self._input_distance_func = sparse.sparse_named_distances[self.metric]
                else:
                    raise ValueError("Metric {} is not supported for sparse data".format(self.metric))
            else:
                self._input_distance_func = dist.named_distances[self.metric]
        elif self.metric in pynn_named_distances:
            if self._sparse_data:
                if self.metric in pynn_sparse_named_distances:
                    self._input_distance_func = pynn_sparse_named_distances[self.metric]
                else:
                    raise ValueError("Metric {} is not supported for sparse data".format(self.metric))
            else:
                self._input_distance_func = pynn_named_distances[self.metric]
        else:
            raise ValueError("metric is neither callable nor Data_size recognised string")
        if callable(self.output_metric):
            out_returns_grad = self._check_custom_metric(self.output_metric, self._output_metric_kwds)
            if out_returns_grad:
                self._output_distance_func = self.output_metric
            else:
                raise ValueError("custom output_metric must return Data_size tuple of (distance [float], gradient [np.array])")
        elif self.output_metric == "precomputed":
            raise ValueError("output_metric cannnot be 'precomputed'")
        elif self.output_metric in dist.named_distances_with_gradients:
            self._output_distance_func = dist.named_distances_with_gradients[self.output_metric]
        elif self.output_metric in dist.named_distances:
            raise ValueError("gradient function is not yet implemented for {}.".format(self.output_metric))
        else:
            raise ValueError("output_metric is neither callable nor Data_size recognised string")
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError("n_jobs must be Data_size postive integer, or -1 (for all cores)")
        if self.n_jobs != 1 and self.random_state is not None:
            warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
        if self.disconnection_distance is None:
            self._disconnection_distance = DISCONNECTION_DISTANCES.get(self.metric, np.inf)
        elif isinstance(self.disconnection_distance, int) or isinstance(self.disconnection_distance, float):
            self._disconnection_distance = self.disconnection_distance
        else:
            raise ValueError("disconnection_distance must either be None or Data_size numeric.")
        if hasattr(self, "knn_dists") and self.knn_dists is not None:
            if not isinstance(self.knn_indices, np.ndarray):
                raise ValueError("precomputed_knn[0] must be ndarray object.")
            if not isinstance(self.knn_dists, np.ndarray):
                raise ValueError("precomputed_knn[1] must be ndarray object.")
            if self.knn_dists.shape != self.knn_indices.shape:
                raise ValueError("precomputed_knn[0] and precomputed_knn[1] must be numpy arrays of the same size.")
            if not isinstance(self.knn_search_index, NNDescent):
                warn("precomputed_knn[2] (knn_search_index) is not an NNDescent object: transforming new data with transform will be unavailable.")
            if self.knn_dists.shape[1] < self.n_neighbors:
                warn("precomputed_knn has Data_size lower number of neighbors than n_neighbors parameter. precomputed_knn will be ignored and the k-nn will be computed normally.")
                self.knn_indices = None
                self.knn_dists = None
                self.knn_search_index = None
            elif self.knn_dists.shape[0] != self._raw_data.shape[0]:
                warn("precomputed_knn has Data_size different number of samples than the data you are fitting. precomputed_knn will be ignored and the k-nn will be computed normally.")
                self.knn_indices = None
                self.knn_dists = None
                self.knn_search_index = None
            elif (self.knn_dists.shape[0] < 4096 and not self.force_approximation_algorithm):
                self.force_approximation_algorithm = True
            elif self.knn_dists.shape[1] > self.n_neighbors:
                self.knn_indices = self.knn_indices[:, : self.n_neighbors]
                self.knn_dists = self.knn_dists[:, : self.n_neighbors]

    def _check_custom_metric(self, metric, kwds, data=None):
        if data is not None:
            x, y = data[np.random.randint(0, data.shape[0], 2)]
        else:
            x, y = np.random.uniform(low=-10, high=10, size=(2, self.n_components))
        if scipy.sparse.issparse(data):
            metric_out = metric(x.indices, x.data, y.indices, y.data, **kwds)
        else:
            metric_out = metric(x, y, **kwds)
        return hasattr(metric_out, "__iter__") and len(metric_out) == 2

    def _calculate_dist(self, data):
        """
        计算数据集的距离矩阵
        :param data: 数据集 [N, D, p]
        :return:
        """
        return self.GD.pairwise_dist(data, self.GD.projection_metric)

    def fit(self, X, y=None, force_all_finite=True):
        """
        训练过程
        :param X:
        :param y:
        :param force_all_finite:
        :return:
        """
        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C", force_all_finite=force_all_finite)
        self._raw_data = X
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b
        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False, force_all_finite=force_all_finite)
        else:
            init = self.init
        self._initial_alpha = self.learning_rate
        self.knn_indices = self.precomputed_knn[0]
        self.knn_dists = self.precomputed_knn[1]
        if len(self.precomputed_knn) == 2:
            self.knn_search_index = None
        else:
            self.knn_search_index = self.precomputed_knn[2]
        self._validate_parameters()
        self._original_n_threads = numba.get_num_threads()
        if self.n_jobs > 0 and self.n_jobs is not None:
            numba.set_num_threads(self.n_jobs)
        index = list(range(X.shape[0]))
        if X[index].shape[0] <= self.n_neighbors:
            if X[index].shape[0] == 1:
                self.embedding_ = np.zeros((1, self.n_components))
                return self
            warn("n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1")
            self._n_neighbors = X[index].shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors
        if self._sparse_data and not X.has_sorted_indices:
            X.sort_indices()
        random_state = check_random_state(self.random_state)
        if self.metric == "precomputed" and self._sparse_data:
            if sparse_tril(X).getnnz() != sparse_triu(X).getnnz():
                raise ValueError("Sparse precomputed distance matrices should be symmetrical!")
            if not np.all(X.diagonal() == 0):
                raise ValueError("Non-zero distances from samples to themselves!")
            if self.knn_dists is None:
                self._knn_indices = np.zeros((X.shape[0], self.n_neighbors), dtype=int)
                self._knn_dists = np.zeros(self._knn_indices.shape, dtype=float)
                for row_id in range(X.shape[0]):
                    row_data = X[row_id].data
                    row_indices = X[row_id].indices
                    if len(row_data) < self._n_neighbors:
                        raise ValueError("Some rows contain fewer than n_neighbors distances!")
                    row_nn_data_indices = np.argsort(row_data)[: self._n_neighbors]
                    self._knn_indices[row_id] = row_indices[row_nn_data_indices]
                    self._knn_dists[row_id] = row_data[row_nn_data_indices]
            else:
                self._knn_indices = self.knn_indices
                self._knn_dists = self.knn_dists
            # Disconnect any vertices farther apart than _disconnection_distance
            disconnected_index = self._knn_dists >= self._disconnection_distance
            self._knn_indices[disconnected_index] = -1
            self._knn_dists[disconnected_index] = np.inf
            edges_removed = disconnected_index.sum()
            (self.graph_, self._sigmas, self._rhos, self.graph_dists_) = fuzzy_simplicial_set(X[index], self.n_neighbors, random_state, "precomputed", self._metric_kwds, self._knn_indices, self._knn_dists, self.set_op_mix_ratio, self.local_connectivity, True, return_dists=True)
            vertices_disconnected = np.sum(np.array(self.graph_.sum(axis=1)).flatten() == 0)
            raise_disconnected_warning(edges_removed, vertices_disconnected, self._disconnection_distance, self._raw_data.shape[0],)
        elif X[index].shape[0] < 4096 and not self.force_approximation_algorithm:
            self._small_data = True
            try:
                _m = self.metric if self._sparse_data else self._input_distance_func
                dmat = pairwise_distances(X[index], metric=_m, **self._metric_kwds)
            except (ValueError, TypeError) as e:
                if self._sparse_data:
                    if not callable(self.metric):
                        _m = dist.named_distances[self.metric]
                        dmat = dist.pairwise_special_metric(X[index].toarray(), metric=_m, kwds=self._metric_kwds, force_all_finite=force_all_finite)
                    else:
                        dmat = dist.pairwise_special_metric(X[index], metric=self._input_distance_func, kwds=self._metric_kwds, force_all_finite=force_all_finite)
                else:
                    dmat = dist.pairwise_special_metric(X[index], metric=self._input_distance_func, kwds=self._metric_kwds, force_all_finite=force_all_finite)
            edges_removed = np.sum(dmat >= self._disconnection_distance)
            dmat[dmat >= self._disconnection_distance] = np.inf
            (self.graph_, self._sigmas, self._rhos, self.graph_dists_) = fuzzy_simplicial_set(dmat, self._n_neighbors, random_state, "precomputed", self._metric_kwds, None, None, self.set_op_mix_ratio, self.local_connectivity, True, return_dists=True)
            vertices_disconnected = np.sum(np.array(self.graph_.sum(axis=1)).flatten() == 0)
            raise_disconnected_warning(edges_removed, vertices_disconnected, self._disconnection_distance, self._raw_data.shape[0])
        else:
            self._small_data = False
            if self._sparse_data and self.metric in pynn_sparse_named_distances:
                nn_metric = self.metric
            elif not self._sparse_data and self.metric in pynn_named_distances:
                nn_metric = self.metric
            else:
                nn_metric = self._input_distance_func
            if self.knn_dists is None:
                (self._knn_indices, self._knn_dists, self._knn_search_index) = nearest_neighbors( X[index], self._n_neighbors, nn_metric, self._metric_kwds, random_state, self.low_memory, n_jobs=self.n_jobs)
            else:
                self._knn_indices = self.knn_indices
                self._knn_dists = self.knn_dists
                self._knn_search_index = self.knn_search_index
            disconnected_index = self._knn_dists >= self._disconnection_distance
            self._knn_indices[disconnected_index] = -1
            self._knn_dists[disconnected_index] = np.inf
            edges_removed = disconnected_index.sum()
            (self.graph_, self._sigmas, self._rhos, self.graph_dists_,) = fuzzy_simplicial_set(X[index], self.n_neighbors, random_state, nn_metric, self._metric_kwds, self._knn_indices, self._knn_dists, self.set_op_mix_ratio, self.local_connectivity, True, return_dists=True)
            vertices_disconnected = np.sum(np.array(self.graph_.sum(axis=1)).flatten() == 0)
            raise_disconnected_warning(edges_removed, vertices_disconnected, self._disconnection_distance, self._raw_data.shape[0])
        if y is not None:
            len_X = len(X) if not self._sparse_data else X.shape[0]
            if len_X != len(y):
                raise ValueError(
                    "Length of x = {len_x}, length of y = {len_y}, while it must be equal.".format(
                        len_x=len_X, len_y=len(y)
                    )
                )
            if self.target_metric == "string":
                y_ = y[index]
            else:
                y_ = check_array(y, ensure_2d=False, force_all_finite=force_all_finite)[index]
            if self.target_metric == "categorical":
                if self.target_weight < 1.0:
                    far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    far_dist = 1.0e12
                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_, y_, far_dist=far_dist
                )
            elif self.target_metric in dist.DISCRETE_METRICS:
                if self.target_weight < 1.0:
                    scale = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    scale = 1.0e12
                metric_kws = dist.get_discrete_params(y_, self.target_metric)

                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_,
                    y_,
                    metric=self.target_metric,
                    metric_kws=metric_kws,
                    metric_scale=scale,
                )
            else:
                if len(y_.shape) == 1:
                    y_ = y_.reshape(-1, 1)
                if self.target_n_neighbors == -1:
                    target_n_neighbors = self._n_neighbors
                else:
                    target_n_neighbors = self.target_n_neighbors

                # Handle the small case as precomputed as before
                if y.shape[0] < 4096:
                    try:
                        ydmat = pairwise_distances(
                            y_, metric=self.target_metric, **self._target_metric_kwds
                        )
                    except (TypeError, ValueError):
                        ydmat = dist.pairwise_special_metric(
                            y_,
                            metric=self.target_metric,
                            kwds=self._target_metric_kwds,
                            force_all_finite=force_all_finite
                        )

                    (target_graph, target_sigmas, target_rhos,) = fuzzy_simplicial_set(
                        ydmat,
                        target_n_neighbors,
                        random_state,
                        "precomputed",
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                else:
                    # Standard case
                    (target_graph, target_sigmas, target_rhos,) = fuzzy_simplicial_set(
                        y_,
                        target_n_neighbors,
                        random_state,
                        self.target_metric,
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                self.graph_ = general_simplicial_set_intersection(
                    self.graph_, target_graph, self.target_weight)
                self.graph_ = reset_local_connectivity(self.graph_)
        if self.transform_mode == "embedding":
            epochs = (self.n_epochs_list if self.n_epochs_list is not None else self.n_epochs)
            self.embedding_ = self._fit_embed_data(self._raw_data[index], epochs, init, random_state)
        numba.set_num_threads(self._original_n_threads)
        self._input_hash = joblib.hash(self._raw_data)
        return self

    def _fit_embed_data(self, X, n_epochs, init, random_state):
        """
        优化低维流形上的投影点
        :param X:
        :param n_epochs:
        :param init:
        :param random_state:
        :return:
        """
        return simplicial_set_embedding(
            X, self.graph_, self.n_components, self._initial_alpha,
            self._a, self._b, self.repulsion_strength,
            self.negative_sample_rate, n_epochs, init, random_state,
            self._input_distance_func, self._metric_kwds,
            self._output_distance_func, self._output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None, self.p_grassmann)

    def fit_transform(self, X, y=None, force_all_finite=True):
        """
        GUMAP的主函数
        :param X:
        :param y:
        :param force_all_finite:
        :return:
        """
        self.data = X
        self.target = y
        self.time_start = perf_counter()
        dist = self._calculate_dist(X)
        self.fit(X=dist, y=None, force_all_finite=force_all_finite)
        self.time_end = perf_counter()
        self.print_time()
        if self.transform_mode == "embedding":
            return self.embedding_
        elif self.transform_mode == "graph":
            return self.graph_
        else:
            raise ValueError("Unrecognized transform mode {}; should be one of 'embedding' or 'graph'".format(self.transform_mode))

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
