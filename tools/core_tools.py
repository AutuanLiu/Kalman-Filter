"""重要工具包

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/1/14
"""

import numpy as np

from kalman_estimation import (get_mat_data, get_terms_matrix, get_txt_data, plot_term_ERR, update_condidate_terms)


def update_terms(data_root='data/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}):
    """update condidate terms

    Args:
        data_root (str): 数据存储根目录
        data_type_set (set, optional): 数据类型集合，Defaults to {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}.
    """

    for data_type in data_type_set:
        update_condidate_terms(f'{data_root}{data_type}_terms.mat', f'{data_root}{data_type}_candidate_terms.txt')


def plot_FROLS_term(data_root='data/', img_root='images/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}):
    """可视化 FROLS 算法中 选择的候选项和 ERR 之间的关系, 图片质量(size: 10*30, dpi=300), 3000 x 9000

    Args:
        data_root (str): 数据存储根目录, Defaults to 'data/'.
        img_root (str): [description], Defaults to 'images/'.
        data_type_set (set, optional): 数据类型集合，Defaults to {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}.
    """

    for data_type in data_type_set:
        terms = get_txt_data(f'{data_root}{data_type}_candidate_terms.txt', delimiter='\n', dtype=np.str)
        ERRs = get_mat_data(f'{data_root}FROLS_{data_type}_est.mat', 'ERRs')
        terms_No = get_mat_data(f'{data_root}FROLS_{data_type}_est.mat', 'terms_chosen')
        terms_matrix = get_terms_matrix(terms, terms_No)
        plot_term_ERR(terms_matrix, ERRs, f'{img_root}FROLS_{data_type}.png')


def plot_Kalman_term(data_root='data/', img_root='images/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}):
    """可视化 Kalman 算法中 选择的候选项和 ERR 之间的关系, 图片质量(size: 10*30, dpi=300), 3000 x 9000

    Args:
        data_root (str): 数据存储根目录, Defaults to 'data/'.
        img_root (str): [description], Defaults to 'images/'.
        data_type_set (set, optional): 数据类型集合，Defaults to {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}.
    """

    for data_type in data_type_set:
        terms = get_txt_data(f'{data_root}{data_type}_candidate_terms.txt', delimiter='\n', dtype=np.str)
        ERRs = get_mat_data(f'{data_root}{data_type}_terms.mat', 'ERRs')
        terms_No = get_mat_data(f'{data_root}{data_type}_terms.mat', 'terms_chosen')
        terms_matrix = get_terms_matrix(terms, terms_No)
        plot_term_ERR(terms_matrix, ERRs, f'{img_root}Kalman_{data_type}.png')
