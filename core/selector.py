"""基于 FROLS 算法的候选项选择器算法(通用算法，无关线性或者非线性)

**参考文献：
1. Billings S A, Chen S, Korenberg M J. Identification of MIMO non-linear systems using a forward-regression orthogonal estimator[J]. International Journal of Control, 1989, 49(6):2157-2189.
2. Billings S A. Nonlinear system identification : NARMAX methods in the time, frequency, and spatio-temporal domains[M]. Wiley, 2013.

**Notes: 参看 matlab 相关实现以及其保存数据

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/12/10
"""

import numpy as np

from .utils import get_mat_data

__all__ = ['Selector']


class Selector:
    """基于 FROLS 算法的候选项选择器算法(通用算法，无关线性或者非线性)

    Attributes:
        norder (int): 非线性次数
        max_lag (int): max lag.
        n_dim (int): the channel or dim of signals.
        Kalman_H (np.array): 供 kalman 滤波器使用的候选项矩阵, 参看 matlab 代码实现
        normalized_signals (np.array): 标准化之后的信号数据, 参看 matlab 代码实现
        Hv (np.array): 基于 base(linear terms) 的候选项组合, 参看 matlab 代码实现
        S_No (np.array): sparse matrix, 表示选择候选项的下标或索引, 参看 matlab 代码实现
        candidate_terms (np.array): 候选项集合
        Kalman_S_No (np.array): 和 Kalman_H 相匹配的候选项选择下标
        n_term (int): FROLS 的阈值或者需要选择的候选项个数
    """

    def __init__(self, terms_path):
        """基于 FROLS 算法的候选项选择器算法

        Args:
            terms_path (str): term selector(matlab) 程序的结果路径
        """

        for key in ['normalized_signals', 'Hv', 'Kalman_H', 'S_No']:
            setattr(self, key, get_mat_data(terms_path, key))
        self.n_dim, n_point, self.n_term = self.Kalman_H.shape
        self.norder = self.Hv.shape[0]
        self.max_lag = self.normalized_signals.shape[0] - n_point
        self.candidate_terms = None
        # !这里注意消除 matlab 和 python 之间的索引差异
        self.Kalman_S_No = self.S_No[:, :self.n_term] - 1

    def make_terms(self, var_name: str = 'x', step_name: str = 't'):
        """生成模型候选项表达式

        Args:
            var_name (str, optional): Defaults to 'x'. 使用的变量名
            step_name (str, optional): Defaults to 't'. 时间点变量名

        Returns:
            terms_repr (np.array) 模型表达式
        """

        terms_repr = []
        base = []
        nonlinear = []
        for var in range(self.n_dim):
            for lag in range(self.max_lag):
                base.append(f'{var_name}{var+1}({step_name}-{lag+1})')
        base = np.asarray(base)
        if self.norder == 1:
            terms_repr = base
        else:
            for ord in range(2, self.norder + 1):
                Hv_tmp = self.Hv[ord - 1, 0]
                n_row, n_var = Hv_tmp.shape
                for m in range(n_row):
                    term = ''
                    for n in range(n_var):
                        # !这里注意消除 matlab 和 python 之间的索引差异
                        term += f'{base[Hv_tmp[m, n]-1]}*'    # *matlab 索引从1开始
                    nonlinear.append(term[:-1])
            nonlinear = np.asarray(nonlinear)
            tmp = list(base)
            tmp.extend(list(nonlinear))
            terms_repr = np.asarray(tmp)
        self.candidate_terms = terms_repr    # 更新 candidate_terms
        return terms_repr

    def make_selection(self):
        """生成与模型候选项选择有关的数据执行此方法前，先执行 make_terms 方法

        Args:
            var_name (str, optional): Defaults to 'x'. 使用的变量名
            step_name (str, optional): Defaults to 't'. 时间点变量名

        Returns:
            normalized_signals (np.array): 标准化之后的信号数据
            Kalman_H (np.array): 供 kalman 滤波器使用的候选项矩阵
            candidate_terms (np.array): 候选项集合
            Kalman_S_No (np.array): 和 Kalman_H 相匹配的候选项选择下标
        """

        return self.normalized_signals, self.Kalman_H, self.candidate_terms, self.Kalman_S_No

    def __repr__(self):
        return f'Selector(norder={self.norder}, max_lag={self.max_lag}, n_dim={self.n_dim}, n_term={self.n_term})'
