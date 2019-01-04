"""使用 Kalman Filter 实现对 MVAR(MultiVariate AutoRegression) 模型的系数估计。

1. https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

from pathlib import Path

import numpy as np
from core import (Kalman4ARX, Kalman4FROLS, Selector, Timer, get_mat_data, make_func4K4FROLS, make_linear_func, normalize, save_2Darray, save_3Darray,
                  torch4FROLS)

timer = Timer()
timer.start()

# !Kalman4ARX 算法
# get data
file_path = './kalman_filter/data/linear_signals5D_noise1.mat'
data = get_mat_data(file_path, 'linear_signals')

# !数据标准化要对整体数据都做
data = normalize(data)

# 网格搜索
# lag_range = np.arange(3, 6)
# uc_range = np.arange(0.001, 0.01, 0.001)
# best_lag_uc = grid_search1(data, lag_range, uc_range)
# print(f'best_lag_uc: {best_lag_uc}')
# best_lag, _ = grid_search2(data, lag_range, 0.001, criterion='AIC', plot=True)
# print(f'best_lag: {best_lag}')
# best_uc, _ = grid_search3(data, 5, uc_range, plot=True)
# print(f'best_uc: {best_uc}')

# 构造 Kalman Filter
# 初始化
max_lag = 4
n_trial = 1
y_coef, A_coef = 0, 0
for t in range(n_trial):
    timer0 = Timer()
    timer0.start()
    kf = Kalman4ARX(data, max_lag, uc=0.01)
    y, A = kf.estimate_coef(0.1)
    y_coef += y
    A_coef += A
    print(f'trial id: {t+1}', end=', ')
    timer0.stop()
y_coef /= n_trial
A_coef /= n_trial

# 保存结果2D
# Write the array to disk
file_path0 = Path('./kalman_filter/data/y_coef.txt')
save_2Darray(file_path0, y_coef)

# 保存结果3D
# Write the array to disk
file_path1 = Path('./kalman_filter/data/A_coef.txt')
save_3Darray(file_path1, A_coef)

# load data(real coef)
# file_path2 = Path('./kalman_filter/data/linear_real_coef.txt')
# real_coef = np.loadtxt(file_path2).reshape(A_coef.shape)
# print(real_coef - A_coef)

# make func
est_model = make_linear_func(A_coef, var_name='x', fname='./kalman_filter/data/linear_est_model.txt')
print(est_model)

timer.stop()

# !线性模型 Kalmal4FROLS 算法(matlab 标准化)
terms_path = './kalman_filter/data/nor_linear_terms.mat'
term = Selector(terms_path)
terms_repr = term.make_terms()

# *保存候选项集合
fname = './kalman_filter/data/linear_candidate_terms.txt'
np.savetxt(fname, terms_repr, fmt='%s')

# *selection
print(term)
normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term.make_selection()

# *构造 Kalman Filter
kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=0.01)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname='./kalman_filter/data/linear_Kalmal4FROLS_est_model.txt')
print(est_model)

# !线性模型 torch4FROLS 算法(matlab 标准化)
terms_path = './kalman_filter/data/nor_linear_terms.mat'
term = Selector(terms_path)
terms_repr = term.make_terms()

# *保存候选项集合
fname = './kalman_filter/data/linear_candidate_terms.txt'
np.savetxt(fname, terms_repr, fmt='%s')

# *selection
print(term)
normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term.make_selection()

# *构造估计器
kf = torch4FROLS(normalized_signals, Kalman_H, n_epoch=100)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname='./kalman_filter/data/linear_torch4FROLS_est_model.txt')
print(est_model)
