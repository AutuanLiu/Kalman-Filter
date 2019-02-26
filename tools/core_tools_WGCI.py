"""重要工具包 计算 WGCI

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/2/22
"""

import numpy as np
import scipy.io as sio

from kalman_estimation import (Kalman4ARX, Kalman4FROLS, Selector, get_terms_matrix, get_txt_data, make_func4K4FROLS, make_linear_func, plot_term_ERR,
                               save_3Darray, torch4FROLS, save_2Darray)


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
    WGCI100 = []
    for trial in range(n_trial):
        # 计算总体情况
        term_selector = Selector(f"{config['data_root']}{data_type}{config['term_path']}{trial+1}.mat")
        # get data
        normalized_signals = term_selector.make_selection()[0]
        print(f'data_type: {data_type}, trial: ### {trial+1}')
        # 构造 Kalman Filter
        kf = Kalman4ARX(normalized_signals, config['max_lag'], uc=config['uc'])
        # 估计系数
        y_coef, A_coef = kf.estimate_coef(config['threshold'])
        # 总体误差
        whole_y_error = np.var(kf.y_error, 0)

        # 子模型情况
        terms_mat = sio.loadmat(f"{config['data_root']}{data_type}{config['term_path']}_WGCI{trial+1}.mat")
        sub1 = []
        for ch in range(5):
            data_set = {
                'normalized_signals': terms_mat['normalized_signals'],
                'Hv': terms_mat['Hv'],
                'Kalman_H': terms_mat['Kalman_H'][0, ch],
                'terms_chosen': terms_mat['terms_chosen'][0, ch]
            }
            term_selector = Selector(data_set)
            # get data
            normalized_signals = term_selector.make_selection()[0]
            # 构造 Kalman Filter
            kf = Kalman4ARX(normalized_signals, config['max_lag'], uc=config['uc'])
            # 估计系数
            y_coef, A_coef = kf.estimate_coef(config['threshold'])
            # 误差
            sub_y_error = np.var(kf.y_error, 0)
            sub1.append(np.log(sub_y_error / whole_y_error))
        WGCI100.append(np.asarray(sub1).T)

    mean_WGCI = np.mean(WGCI100, 0)
    var_WGCI = np.var(WGCI100, 0)
    print(f"mean_WGCI = {mean_WGCI}, var_WGCI = {var_WGCI}")
    fname1 = f"{config['data_root']}{data_type}_kalman4ARX100_{config['est_fname']}WGCI100.txt"
    save_3Darray(fname1, np.array([mean_WGCI * (mean_WGCI > 0.01), var_WGCI * (var_WGCI > 0.01)]))


def kalman4FROLS_pipeline(data_type, configs, n_trial):
    """基于 Kalman 滤波器的各个算法的 pipeline

    Args:
        data_type: 数据类型
        configs (dict): 配置字典
        n_trial: 试验次数
    """

    config = configs[data_type]
    WGCI100 = []
    for trial in range(n_trial):
        term_selector = Selector(f"{config['data_root']}{data_type}{config['term_path']}{trial+1}.mat")
        # get data
        normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term_selector.make_selection()
        print(f'data_type: {data_type}, trial: ### {trial+1}')
        # 构造 Kalman Filter
        kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=config['uc'])
        y_coef = kf.estimate_coef()
        print(y_coef)
        # 总体误差

        whole_y_error = np.var(kf.y_error, 0)

        # 子模型情况
        terms_mat = sio.loadmat(f"{config['data_root']}{data_type}{config['term_path']}_WGCI{trial+1}.mat")
        sub1 = []
        for ch in range(5):
            data_set = {
                'normalized_signals': terms_mat['normalized_signals'],
                'Hv': terms_mat['Hv'],
                'Kalman_H': terms_mat['Kalman_H'][0, ch],
                'terms_chosen': terms_mat['terms_chosen'][0, ch]
            }
            term_selector = Selector(data_set)
            # get data
            normalized_signals = term_selector.make_selection()[0]
            # 构造 Kalman Filter
            kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=config['uc'])
            # 估计系数
            y_coef = kf.estimate_coef()
            # 误差
            sub_y_error = np.var(kf.y_error, 0)
            sub1.append(np.log(sub_y_error / whole_y_error))
        WGCI100.append(np.asarray(sub1).T)

    mean_WGCI = np.mean(WGCI100, 0)
    var_WGCI = np.var(WGCI100, 0)
    print(f"mean_WGCI = {mean_WGCI}, var_WGCI = {var_WGCI}")
    fname1 = f"{config['data_root']}{data_type}_kalman4FROLS100_{config['est_fname']}WGCI100.txt"
    save_3Darray(fname1, np.array([mean_WGCI * (mean_WGCI > 0.01), var_WGCI * (var_WGCI > 0.01)]))


def torch4FROLS_pipeline(data_type, configs, n_trial):
    """基于 Kalman 滤波器的各个算法的 pipeline

    Args:
        data_type: 数据类型
        configs (dict): 配置字典
        n_trial: 试验次数
    """

    config = configs[data_type]
    WGCI100 = []
    for trial in range(n_trial):
        term_selector = Selector(f"{config['data_root']}{data_type}{config['term_path']}{trial+1}.mat")
        # get data
        normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term_selector.make_selection()
        print(f'data_type: {data_type}, trial: ### {trial+1}')
        kf = torch4FROLS(normalized_signals, Kalman_H=Kalman_H, n_epoch=config['n_epoch'])
        y_coef = kf.estimate_coef()
        # 总体误差
        whole_y_error = np.var(kf.y_error, 0)

        # 子模型情况
        terms_mat = sio.loadmat(f"{config['data_root']}{data_type}{config['term_path']}_WGCI{trial+1}.mat")
        sub1 = []
        for ch in range(5):
            data_set = {
                'normalized_signals': terms_mat['normalized_signals'],
                'Hv': terms_mat['Hv'],
                'Kalman_H': terms_mat['Kalman_H'][0, ch],
                'terms_chosen': terms_mat['terms_chosen'][0, ch]
            }
            term_selector = Selector(data_set)
            # get data
            normalized_signals = term_selector.make_selection()[0]
            # 构造 Kalman Filter
            kf = torch4FROLS(normalized_signals, Kalman_H=Kalman_H, n_epoch=config['n_epoch'])
            # 估计系数
            y_coef = kf.estimate_coef()
            # 误差
            sub_y_error = np.var(kf.y_error, 0)
            sub1.append(np.log(sub_y_error / whole_y_error))
        WGCI100.append(np.asarray(sub1).T)

    mean_WGCI = np.mean(WGCI100, 0)
    var_WGCI = np.var(WGCI100, 0)
    print(f"mean_WGCI = {mean_WGCI}, var_WGCI = {var_WGCI}")
    fname1 = f"{config['data_root']}{data_type}_kalman4FROLS100_{config['est_fname']}WGCI100.txt"
    save_3Darray(fname1, np.array([mean_WGCI * (mean_WGCI > 0.01), var_WGCI * (var_WGCI > 0.01)]))
