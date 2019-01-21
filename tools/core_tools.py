"""重要工具包

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/1/14
"""

import numpy as np

from kalman_estimation import (Kalman4ARX, Kalman4FROLS, Selector,
                               get_mat_data, get_terms_matrix, get_txt_data,
                               make_func4K4FROLS, make_linear_func,
                               plot_term_ERR, save_3Darray, torch4FROLS,
                               update_condidate_terms)


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


def get_json_data(fname):
    """获取 JSON 数据

    Args:
        fname (str): 存储 JSON 数据的文件路径和文件名
    """

    import ujson
    return ujson.load(open(fname, 'r'))


def kalman_pipeline(configs, n_trial=1):
    """基于 Kalman 滤波器的各个算法的 pipeline

    Args:
        configs (dict): 配置字典
        n_trial (int): 试验次数
    """

    for data_type in configs.keys():
        config = configs[data_type]
        term_selector = Selector(f"{config['data_root']}{data_type}{config['term_path']}")
        terms_set = term_selector.make_terms()
        # get data
        normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term_selector.make_selection()
        # 构造 Kalman Filter
        for algo_type in config['algorithm']:
            y_coef100 = np.zeros((n_trial+2, 5, 5))
            for trial in range(n_trial):
                print(f"{data_type} {algo_type} ### {trial+1}")
                fname = f"{config['data_root']}{data_type}_{algo_type}_{config['est_fname']}"
                if algo_type == 'Kalman4ARX':
                    y_coef100 = np.zeros((n_trial + 2, 5, 25))
                    kf = Kalman4ARX(normalized_signals, config['max_lag'], uc=config['uc'])
                    # 估计系数
                    y_coef, A_coef = kf.estimate_coef(config['threshold'])
                    # 计算模型表达式并保存
                    est_model = make_linear_func(A_coef, fname=fname)
                    # 保存平均结果
                    y_coef100[trial] = y_coef
                elif algo_type == 'Kalman4FROLS':
                    kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=config['uc'])
                    y_coef = kf.estimate_coef()
                    est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname=fname)
                    # 保存平均结果
                    y_coef100[trial] = y_coef
                elif algo_type == 'torch4FROLS':
                    kf = torch4FROLS(normalized_signals, Kalman_H=Kalman_H, n_epoch=config['n_epoch'])
                    y_coef = kf.estimate_coef()
                    est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname=fname)
                    # 保存平均结果
                    y_coef100[trial] = y_coef
                else:
                    print('!Not Defined!')
                y_coef100[trial+1] = np.mean(y_coef, 0)
                y_coef100[trial+2] = np.std(y_coef, 0)
                print(f"\n{data_type}_{algo_type} est model saved!\n")
                fname = f"{config['data_root']}{data_type}_{algo_type}_coef100.txt"
                save_3Darray(fname, y_coef100)

            # 输出结果
            print(est_model)
