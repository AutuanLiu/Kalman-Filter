"""
Email: autuanliu@163.com
Date: 2018/11/212
# !!用于网格搜索
"""

import numpy as np

from .estimator import Kalman4ARX

__all__ = ['grid_search1', 'grid_search2', 'grid_search3', 'plot_grid_search']


def grid_search1(signals, lag_range, uc_range):
    """对超参数进行网格搜索策略(policy 1)，lag 和 uc 均不固定

    Args:
        signals (np.array): the data need to process.
        lag_range (list or tuple or np.array, iterable): the range of max_lag.
        uc_range (list or tuple or np.array, iterable): the range of update coefficient.

    Returns:
        best_lag_uc
    """

    # policy 1: (max_lag, uc) 的最佳组合，使得 mse_loss 最小
    min_mse = 100
    best_lag_uc = (5, 0.001)
    for m in lag_range:
        for n in uc_range:
            kf = Kalman4ARX(signals, m, n)
            _ = kf.smoother()
            min_mse = kf.mse_loss if kf.mse_loss <= min_mse else min_mse
            print(f'max_lag={m}, uc={n}, mse_loss={kf.mse_loss}')
            best_lag_uc = (m, n)
    return best_lag_uc


def grid_search2(signals, lag_range, uc, criterion='BIC', plot=False):
    """对超参数进行网格搜索策略(policy 2)，uc 固定

    Args:
        signals (np.array): the data need to process.
        lag_range (list or tuple or np.array, iterable): the range of max_lag.
        uc (float): update coefficient.
        criterion (str): the criterion for searching max_lag.
        plot (bool): 是否绘图

    Returns:
        best_lag, (x, y)
    """

    # policy 2: max_lag的最佳值，使得 AIC 或者 BIC 最小
    min_criterion = 100
    best_lag = 1
    # 绘图时保存
    x, y = [], []
    for m in lag_range:
        kf = Kalman4ARX(signals, m, uc)
        _ = kf.smoother()
        min_criterion = getattr(kf, criterion.lower()) if getattr(kf, criterion.lower()) <= min_criterion else min_criterion
        print(f'max_lag={m}, {criterion}={getattr(kf, criterion.lower())}')
        best_lag = m
        x.append(m)
        y.append(getattr(kf, criterion.lower()))
    if plot:
        plot_grid_search(x, y, 'max_lag', criterion, f'max_lag2{criterion}')
    return best_lag, (x, y)


def grid_search3(signals, max_lag, uc_range, criterion='BIC', plot=False):
    """对超参数进行网格搜索策略(policy 3)，max_lag 固定

    Args:
        signals (np.array): the data need to process.
        max_lag (int): max_lag.
        uc_range (list or tuple or np.array, iterable): the range of update coefficient.
        criterion (str): the criterion for searching max_lag.
        plot (bool): 是否绘图

    Returns:
        best_uc, (x, y)
    """

    # policy 2: uc 的最佳值，使得 mse_loss 最小
    min_mse = 100
    best_uc = 1
    # 绘图时保存
    x, y = [], []
    for m in uc_range:
        kf = Kalman4ARX(signals, max_lag, m)
        _ = kf.smoother()
        min_mse = kf.mse_loss if kf.mse_loss <= min_mse else min_mse
        print(f'uc={m}, mse_loss={kf.mse_loss}')
        best_uc = m
        x.append(m)
        y.append(kf.mse_loss)
    if plot:
        plot_grid_search(x, y, 'uc', 'mse_loss', 'uc2mse_loss')
    return best_uc, (x, y)


def plot_grid_search(x, y, x_label, y_label, file_name):
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'b+')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # save figure
    plt.savefig(f'./kalman_filter/images/{file_name}.png')
