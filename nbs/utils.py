import numpy as np
from kalman_estimation import Kalman4FROLS, Selector, get_mat_data
from tqdm import trange, tqdm, tqdm_notebook
import matplotlib.pyplot as plt


def corr_term(y_coef, terms_set, Kalman_S_No, var_name: str = 'x', step_name: str = 't'):
    n_dim, n_term = y_coef.shape
    func_repr = []
    for var in range(n_dim):
        y = {}
        for term in range(n_term):
            y[terms_set[Kalman_S_No[var, term]]] = y_coef[var, term]
        func_repr.append(y)
    return func_repr


def frokf(noise_var, ndim, dtype, terms, length, root='../data/', trials=100, uc=0.01):
    assert dtype in ['linear', 'nonlinear'], 'type not support!'
    ax = []
    for trial in tqdm(range(1, trials + 1)):
        terms_path = root + f'{dtype}_terms{ndim}D_{noise_var:2.2f}trial{trial}.mat'
        term = Selector(terms_path)
        _ = term.make_terms()
        normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term.make_selection()
        Kalman_S_No = np.sort(Kalman_S_No)
        kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=uc)
        y_coef = kf.estimate_coef()
        terms_set = corr_term(y_coef, candidate_terms, Kalman_S_No)
        flatten_coef, t = [], 0
        for i in range(ndim):
            tmp = []
            for k in terms[t:t+length[i]]:
                tmp.append(terms_set[i][k] if k in terms_set[i] else np.nan)
            flatten_coef.extend(tmp)
            t += length[i]
        ax.append(flatten_coef)
    return np.stack(ax)


def frols(noise_var, ndim, dtype, terms, length, root='../data/', trials=100):
    assert dtype in ['linear', 'nonlinear'], 'type not support!'
    terms_path = root + f'FROLS_{ndim}{dtype}_est100_{noise_var:2.2f}.mat'
    terms_pathx = root + f'{dtype}_terms{ndim}D_0.50trial1.mat'
    term = Selector(terms_pathx)
    candidate_terms = term.make_terms()
    y_coef = get_mat_data(terms_path, 'coef_est100')
    Kalman_S_No = get_mat_data(terms_path, 'terms_chosen100')
    flatten_coef = []
    for trial in trange(trials):
        ax, t, S_No = [], 0, Kalman_S_No[trial] - 1
        for i in range(ndim):
            terms_set = corr_term(y_coef[trial], candidate_terms, S_No)
            tmp = []
            for k in terms[t:t+length[i]]:
                tmp.append(terms_set[i][k] if k in terms_set[i] else np.nan)
            ax.extend(tmp)
            t += length[i]
        flatten_coef.append(ax)
    return np.stack(flatten_coef)


def frokf_sta(dtype, ndim, noise_var, root='../data/'):
    name = f"{root}FROKF_{dtype}{ndim}D_{noise_var:2.2f}"
    coef = get_mat_data(name, 'frokf_coef')
    return coef.mean(0), coef.var(0)


def frols_sta(dtype, ndim, noise_var, root='../data/'):
    name = f"{root}FROLS_{dtype}{ndim}D_{noise_var:2.2f}"
    coef = get_mat_data(name, 'frols_coef')
    return coef.mean(0), coef.var(0)


def get_term_dict(dtype, dim, root='../data/'):
    terms_pathx = root + f'{dtype}_terms{dim}D_0.50trial1.mat'
    term = Selector(terms_pathx)
    candidate_terms = term.make_terms()
    dict1 = {t:i for i, t in enumerate(candidate_terms)}
    dict2 = {i:t for i, t in enumerate(candidate_terms)}
    return dict1, dict2
