################################################################################
# 本文件用于实现Grassmann UMAP算法的优化
################################################################################
# 导入模块
import numba
import numpy as np
from GUMAP.utils import tau_rand_int
from tqdm import tqdm
################################################################################
# 定义必要函数
@numba.njit(fastmath=True, cache=True)
def gdist(x, y):
    """
    F范数距离
    :param x:
    :param y:
    :return:
    """
    return np.linalg.norm(np.dot(x, x.T) - np.dot(y, y.T)) / np.sqrt(2)

@numba.njit(fastmath=True, cache=True)
def grassmann_matrix_grad(x, y):
    """
    用于计算梯度
    :param x:
    :param y:
    :return:
    """
    return (x@x.T - y@y.T)@x
################################################################################
def _optimize_layout_grassmann_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n
):
    """
    Grassmann UMAP算法的单次训练
    :param head_embedding: 低维投影
    :param tail_embedding: 与head_embedding相同
    :param head:
    :param tail:
    :param n_vertices:
    :param epochs_per_sample:
    :param a:
    :param b:
    :param rng_state:
    :param gamma:
    :param move_other:
    :param alpha:
    :param epochs_per_negative_sample:
    :param epoch_of_next_negative_sample:
    :param epoch_of_next_sample:
    :param n:
    :return:
    """
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]
            current = head_embedding[j]
            other = tail_embedding[k]
            dist_squared = gdist(current, other)
            # 吸引力
            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0
            attr_grid = grassmann_matrix_grad(current, other)
            grad_d = np.clip(grad_coeff * attr_grid, a_min=-4.0, a_max=4.0)
            current += grad_d * alpha
            if move_other:
                other += -grad_d * alpha
            epoch_of_next_sample[i] += epochs_per_sample[i]
            # 排斥力
            n_neg_samples = int((n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices
                other = tail_embedding[k]
                dist_squared = gdist(current, other)
                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0
                repul_grid = grassmann_matrix_grad(current, other)
                if grad_coeff > 0.0:
                    grad_d = np.clip(grad_coeff * repul_grid, a_min=-4.0, a_max=4.0)
                else:
                    grad_d = np.clip(grad_coeff * repul_grid, a_min=0.0, a_max=0.0)
                current += grad_d * alpha
            epoch_of_next_negative_sample[i] += (n_neg_samples * epochs_per_negative_sample[i])
################################################################################
def optimize_layout_grassmann(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    move_other=False
):
    """
    Grassmann UMAP算法的循环训练
    :param head_embedding: 初始化的低维投影       [N, d, p]
    :param tail_embedding: 与head_embedding相同  [N, d, p]
    :param head:           近邻图的行索引
    :param tail:           近邻图的列索引
    :param n_epochs:       迭代次数
    :param n_vertices:
    :param epochs_per_sample:
    :param a:
    :param b:
    :param rng_state:
    :param gamma:
    :param initial_alpha:
    :param negative_sample_rate:
    :param parallel:
    :param move_other:
    :return:
    """
    alpha = initial_alpha
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()
    optimize_fn = numba.njit(_optimize_layout_grassmann_single_epoch, fastmath=True, parallel=parallel)
    for n in tqdm(range(n_epochs)):
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            move_other,
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n
        )
        # 重新建模低维投影到格拉斯曼流形上
        for he in range(len(head_embedding)):
            Q, R = np.linalg.qr(head_embedding[he])
            head_embedding[he] = Q
        for te in range(len(tail_embedding)):
            Q, R = np.linalg.qr(tail_embedding[te])
            tail_embedding[te] = Q
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))
    return head_embedding
