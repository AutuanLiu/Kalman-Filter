"""测试 Kalman Filter 估计模型系数。

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

import numpy as np

from core import (Kalman4ARX, Kalman4FROLS, Selector, get_mat_data, make_func4K4FROLS, make_linear_func, normalize, torch4FROLS)

# !Kalman4ARX 测试
# timer = Timer()
# timer.start()

# get data 线性模型
file_path = './kalman_filter/data/linear_signals5D_noise1.mat'
data = get_mat_data(file_path, 'linear_signals')

# 数据标准化
data = normalize(data)

# *构造 Kalman Filter
kf = Kalman4ARX(data, 5, uc=0.01)
y_coef, A_coef = kf.estimate_coef(0.1)
print(y_coef, A_coef)

# *估计模型生成
est_model = make_linear_func(A_coef, var_name='x', fname='./kalman_filter/data/linear_est_model.txt')
print(est_model)

# !Kalman4FROLS 测试
# !Selector 测试
# !线性模型
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
est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname='./kalman_filter/data/K4FROLS_est_model.txt')
print(est_model)

# !非线性模型
terms_path = './kalman_filter/data/nor_nonlinear_terms.mat'
term = Selector(terms_path)
terms_repr = term.make_terms()

# *保存候选项集合
fname = './kalman_filter/data/nonlinear_candidate_terms.txt'
np.savetxt(fname, terms_repr, fmt='%s')

# *selection
normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term.make_selection()

# *构造 Kalman Filter
kf = Kalman4FROLS(normalized_signals, Kalman_H, uc=0.01)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname='./kalman_filter/data/nonlinear_Kalman4FROLS_est_model.txt')
print(est_model)

# !sklearn4FROLS 测试
# terms_path = './kalman_filter/data/nonlinear_terms.mat'
# term = Selector(terms_path)
# terms_repr = term.make_terms()

# # *保存候选项集合
# fname = './kalman_filter/data/nonlinear_candidate_terms.txt'
# np.savetxt(fname, terms_repr, fmt='%s')

# # *selection
# Kalman_H, candidate_terms, terms_No, max_lag = term.make_selection()

# # *非线性数据
# file_path = './kalman_filter/data/nonlinear_signals5D_noise1.mat'
# data = get_mat_data(file_path, 'nonlinear_signals')

# # 数据标准化
# data = normalize(data)

# # *构造 估计器
# kf = sklearn4FROLS(data, Kalman_H)
# y_coef = kf.estimate_coef()
# print(y_coef)

# # *估计模型生成
# est_model = make_func4K4FROLS(y_coef, candidate_terms, terms_No, fname='./kalman_filter/data/K4FROLS_est_model.txt')
# print(est_model)

# !torch4FROLS 测试
terms_path = './kalman_filter/data/nonlinear_terms.mat'
term = Selector(terms_path)
terms_repr = term.make_terms()

# *保存候选项集合
fname = './kalman_filter/data/nonlinear_candidate_terms.txt'
np.savetxt(fname, terms_repr, fmt='%s')

# *selection
normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term.make_selection()

# *构造 估计器
kf = torch4FROLS(normalized_signals, Kalman_H, n_epoch=50)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname='./kalman_filter/data/torch4FROLS_est_model.txt')
print(est_model)
