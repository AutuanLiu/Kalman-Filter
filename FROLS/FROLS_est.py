"""使用 FROLS 实现对模型的系数估计。

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/1/14
"""

import numpy as np

from kalman_estimation import get_mat_data, get_txt_data

data_types = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}


def make_FROLS_func(var_name: str = 'x', step_name: str = 't', save=True, fname='FROLS_est_model.txt', **kwargs):
    """get the coefficient of FROLS estimation.

    Args:
        var_name (str, optional): Defaults to 'x'. 变量名
        step_name (str, optional): Defaults to 't'. 时间点变量名
        save (bool, optional): Defaults to True. 是否保存结果
        fname (str, optional): Defaults to 'est_model.txt'. 保存的文件名

    Returns:
        func_repr (np.ndarray) 模型表达式
    """

    root = 'data/'
    for d_type in data_types:
        coef = get_mat_data(f'{root}FROLS_{d_type}_coef.mat', 'coef_est')
        terms = get_txt_data(f'{root}{d_type}_candidate_terms.txt', delimiter='\n', dtype=np.str)
        # 候选项的顺序是相同的，因为采用了相同的算法计算候选项
        [n_dim, n_term] = coef.shape
        func_repr = []
        for row in range(n_dim):
            y = f'{var_name}{row+1}({step_name}) = '
            for col in range(n_term):
                if abs(coef[row, col]) > 1e-5:
                    y += f'{coef[row, col]:.4f} * {terms[col]} + '
            func_repr.append(y + f'e{row+1}({step_name})')
        func_repr = np.array(func_repr).reshape(-1, 1)
        fname = f'{root}FROLS_{d_type}_est_model.txt'
        if save:
            np.savetxt(fname, func_repr, fmt='%s', **kwargs)


make_FROLS_func()
