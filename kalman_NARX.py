"""使用 Extend Kalman Filter 实现对 MVARX 模型的系数估计。

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

import numpy as np

from core import (Kalman4FROLS, Selector, get_mat_data, make_func4K4FROLS, normalize, torch4FROLS)

# !非线性模型
# *非线性数据
terms_path = './kalman_filter/data/nor_nonlinear_terms.mat'
term = Selector(terms_path)
terms_repr = term.make_terms()

# *保存候选项集合
fname = './kalman_filter/data/nonlinear_candidate_terms.txt'
np.savetxt(fname, terms_repr, fmt='%s')

# *selection
normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term.make_selection()

# *构造 Kalman Filter
kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=0.01)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname='./kalman_filter/data/nonlinear_Kalman4FROLS_est_model.txt')
print(est_model)

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
est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname='./kalman_filter/data/nonlinear_torch4FROLS_est_model.txt')
print(est_model)
