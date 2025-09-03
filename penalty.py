from numpy import array, zeros, full, argmin, inf
from math import isinf
import numpy as np


def apmdtw(x, y, cost_matrix,  warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x) #assert断言语句，assert 表达式，如果表达式为真，则继续执行，如果表达式为假，则直接崩溃
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y))) #isinf()检查无穷大，返回bool型值
    assert s>0
    r, c = len(x), len(y)

    if not isinf(w): #w不为无限大的值
        D0 = full((r + 1, c + 1), inf) #numpy.full(shape, fill_value, dtype=None, order='C')[source], 返回一个根据指定的shape和type，并用fill_value填充的新数组
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else: #w为无限大的值
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf #第一行的值全部为无穷大
        D0[1:, 0] = inf #第一列的值全部为无穷大
    D1 = D0[1:, 1:]  # view
    D0[1:, 1:] = cost_matrix

    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):#w不为inf
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), D1, path



def _traceback(D):
    i, j = array(D.shape) - 2
    # i, j = cp.array(D.shape) -2
    p, q = [i], [j]
    track_verti = 0
    track_hori = 0
    while (i > 0) or (j > 0):
        # tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            track_hori = track_hori+1
            i -= 1
        else:  # (tb == 2):
            track_verti = track_verti+1
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


