"""重要工具包

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/2/20
"""

import numpy as np

from kalman_estimation import (Kalman4ARX, Kalman4FROLS, Selector, get_mat_data, get_terms_matrix, get_txt_data, make_func4K4FROLS, make_linear_func,
                               plot_term_ERR, save_3Darray, torch4FROLS, update_condidate_terms, save_2Darray)


def update_terms(data_root='data/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}):
    """update condidate terms

    Args:
        data_root (str): 数据存储根目录
        data_type_set (set, optional): 数据类型集合，Defaults to {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}.
    """

    for data_type in data_type_set:
        update_condidate_terms(f'{data_root}{data_type}_terms.mat', f'{data_root}{data_type}_candidate_terms.txt')


def get_json_data(fname):
    """获取 JSON 数据

    Args:
        fname (str): 存储 JSON 数据的文件路径和文件名
    """

    import ujson
    return ujson.load(open(fname, 'r'))


def kalman4ARX_pipeline(data_type, configs, n_trial):
    """基于 Kalman 滤波器的各个算法的 pipeline

    types = ['linear', 'longlag_linear']

    Args:
        data_type: 数据类型
        configs (dict): 配置字典
        n_trial: 试验次数
    """

    config = configs[data_type]
    if data_type == 'linear':
        y_coef100 = np.zeros((100, 5, 25))
    else:
        y_coef100 = np.zeros((100, 5, 50))
    for trial in range(n_trial):
        term_selector = Selector(f"{config['data_root']}{data_type}{config['term_path']}{trial+1}.mat")
        terms_set = term_selector.make_terms()
        # get data
        normalized_signals = term_selector.make_selection()[0]
        fname = f"{config['data_root']}{data_type}_kalman4ARX100_{config['est_fname']}{trial+1}.txt"
        print(f'data_type: {data_type}, trial: ### {trial+1}')
        # 构造 Kalman Filter
        kf = Kalman4ARX(normalized_signals, config['max_lag'], uc=config['uc'])
        # 估计系数
        y_coef, A_coef = kf.estimate_coef(config['threshold'])
        y_coef100[trial] = y_coef
        # 计算模型表达式并保存
        est_model = make_linear_func(A_coef, fname=fname)
        # 输出结果
        # print(est_model)
    fname1 = f"{config['data_root']}{data_type}_kalman4ARX100_{config['est_fname']}log.txt"
    save_3Darray(fname1, y_coef100)
    mean_y = np.mean(y_coef100, 0)
    var_y = np.var(y_coef100, 0)
    print(mean_y, var_y, sep='\n')
    fname1 = f"{config['data_root']}{data_type}_kalman4ARX100_{config['est_fname']}log100.txt"
    save_3Darray(fname1, np.array([mean_y, var_y]))


def kalman4FROLS_pipeline(data_type, configs, n_trial, id_correct, n_correct):
    """基于 Kalman 滤波器的各个算法的 pipeline

    Args:
        data_type: 数据类型
        configs (dict): 配置字典
        n_trial: 试验次数
    """

    config = configs[data_type]
    y_coef100 = np.zeros((100, 5, 5))
    y_coef9 = np.zeros((100, 9))
    for trial in range(n_trial):
        fname = f"{config['data_root']}{data_type}_kalman4FROLS100_{config['est_fname']}{trial+1}.txt"
        term_selector = Selector(f"{config['data_root']}{data_type}{config['term_path']}{trial+1}.mat")
        terms_set = term_selector.make_terms()
        # get data
        normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term_selector.make_selection()
        print(f'data_type: {data_type}, trial: ### {trial+1}')
        # 构造 Kalman Filter
        kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=config['uc'])
        y_coef = kf.estimate_coef()
        y_coef100[trial] = y_coef
        est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname=fname)
        coef9 = []
        Kalman_S_No_order = np.sort(Kalman_S_No)
        for row in range(5):
            for t in range(n_correct[row]):
                idx = np.argwhere(Kalman_S_No_order[row, :] == id_correct[data_type][row][t])
                value = y_coef[row, idx]
                coef9.append(value[0, 0])
        y_coef9[trial] = np.array(coef9)
        # 输出结果
        # print(est_model)
    fname1 = f"{config['data_root']}{data_type}_kalman4FROLS100_{config['est_fname']}log.txt"
    save_3Darray(fname1, y_coef100)
    mean_y = np.mean(y_coef9, 0)
    var_y = np.var(y_coef9, 0)
    print(mean_y, var_y, sep='\n')
    fname1 = f"{config['data_root']}{data_type}_kalman4FROLS100_{config['est_fname']}log100.txt"
    save_2Darray(fname1, np.array([mean_y, var_y]).T)


def torch4FROLS_pipeline(data_type, configs, n_trial, id_correct, n_correct):
    """基于 Kalman 滤波器的各个算法的 pipeline

    Args:
        data_type: 数据类型
        configs (dict): 配置字典
        n_trial: 试验次数
    """

    config = configs[data_type]
    y_coef100 = np.zeros((100, 5, 5))
    y_coef9 = np.zeros((100, 9))
    for trial in range(n_trial):
        fname = f"{config['data_root']}{data_type}_torch4FROLS100_{config['est_fname']}{trial+1}.txt"
        term_selector = Selector(f"{config['data_root']}{data_type}{config['term_path']}{trial+1}.mat")
        terms_set = term_selector.make_terms()
        # get data
        normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term_selector.make_selection()
        print(f'data_type: {data_type}, trial: ### {trial+1}')
        kf = torch4FROLS(normalized_signals, Kalman_H=Kalman_H, n_epoch=config['n_epoch'])
        y_coef = kf.estimate_coef()
        y_coef100[trial] = y_coef
        est_model = make_func4K4FROLS(y_coef, candidate_terms, Kalman_S_No, fname=fname)
        coef9 = []
        Kalman_S_No_order = np.sort(Kalman_S_No)
        for row in range(5):
            for t in range(n_correct[row]):
                idx = np.argwhere(Kalman_S_No_order[row, :] == id_correct[data_type][row][t])
                value = y_coef[row, idx]
                coef9.append(value[0, 0])
        y_coef9[trial] = np.array(coef9)

        # 输出结果
        # print(est_model)
    fname1 = f"{config['data_root']}{data_type}_torch4FROLS100_{config['est_fname']}log.txt"
    save_3Darray(fname1, y_coef100)
    mean_y = np.mean(y_coef9, 0)
    var_y = np.var(y_coef9, 0)
    print(mean_y, var_y, sep='\n')
    fname1 = f"{config['data_root']}{data_type}_torch4FROLS100_{config['est_fname']}log100.txt"
    save_2Darray(fname1, np.array([mean_y, var_y]).T)
