import numpy as np
from kalman_estimation import Kalman4FROLS, Selector, get_mat_data


def corr_term(y_coef, terms_set, Kalman_S_No, var_name: str = 'x', step_name: str = 't'):
    n_dim, n_term = y_coef.shape
    Kalman_S_No = np.sort(Kalman_S_No)
    func_repr = []
    for var in range(n_dim):
        y = {}
        for term in range(n_term):
            y[terms_set[Kalman_S_No[var, term]]] = y_coef[var, term]
        func_repr.append(y)
    return func_repr


def frokf(noise_var, trial, ndim, dtype, terms, length, root='../data/', uc=0.01):
    assert dtype in ['linear', 'nonlinear'], 'type not support!'
    terms_path = root + f'{dtype}_terms{ndim}D_{noise_var:2.2f}trial{trial}.mat'
    term = Selector(terms_path)
    _ = term.make_terms()
    normalized_signals, Kalman_H, candidate_terms, Kalman_S_No = term.make_selection()
    kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=uc)
    y_coef = kf.estimate_coef()
    terms_set = corr_term(y_coef, candidate_terms, Kalman_S_No)
    flatten_coef, t = [], 0
    for i in range(ndim):
        flatten_coef.extend([terms_set[i][k] for k in terms[t:t+length[i]]])
        t += length[i]
    return flatten_coef
