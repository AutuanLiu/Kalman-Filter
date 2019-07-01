import numpy as np
from tqdm import trange, tqdm
from utils import frokf
import scipy.io as sio


## 配置
con_terms_linear5 = ['x1(t-1)', 'x1(t-2)', 'x1(t-2)', 'x1(t-3)', 'x1(t-2)', 'x4(t-1)', 'x5(t-1)', 'x4(t-1)', 'x5(t-1)']  # 9
con_terms_nonlinear5 = ['x1(t-1)', 'x1(t-2)', 'x1(t-2)*x1(t-2)', 'x1(t-3)', 'x1(t-2)*x1(t-2)', 'x4(t-1)', 'x5(t-1)', 'x4(t-1)', 'x5(t-1)']  # 9
true_coefs5 = [0.95*np.sqrt(2), -0.9025, 0.5, -0.4, -0.5, 0.25*np.sqrt(2), 0.25*np.sqrt(2), -0.25*np.sqrt(2), 0.25*np.sqrt(2)]  # 9
con_terms_linear10 = ['x1(t-1)', 'x1(t-2)', 'x1(t-2)', 'x2(t-3)', 'x1(t-2)', 'x4(t-4)', 'x9(t-2)', 'x4(t-4)', 'x1(t-1)', 'x1(t-2)', 'x7(t-2)', 
                      'x8(t-3)', 'x9(t-3)', 'x8(t-3)', 'x9(t-3)', 'x7(t-4)']  # 16
con_terms_nonlinear10 = ['x1(t-1)', 'x1(t-2)', 'x1(t-2)*x1(t-10)', 'x2(t-3)', 'x1(t-2)', 'x4(t-4)', 'x9(t-2)', 'x4(t-4)', 'x1(t-1)*x1(t-10)', 'x1(t-2)', 'x7(t-2)', 
                      'x8(t-3)', 'x9(t-3)', 'x8(t-3)', 'x9(t-3)', 'x7(t-4)']  # 16
true_coefs10 = [0.95*np.sqrt(2), -0.9025, 0.5, 0.9, -0.5, 0.8, -0.4, -0.8, 0.4, -0.4, -0.9, 0.4, 0.3, -0.3, 0.4, -0.75]  # 16
noises = np.linspace(0.5, 4, 8)
con_terms5 = [2, 1, 1, 3, 2]
con_terms10 = [2, 1, 1, 1, 2, 1, 2, 3, 2, 1]
root = '../data/'

## 批量计算保存
ret = []
for dtype in tqdm(['linear', 'nonlinear']):
    for ndim in tqdm([5, 10]):
        for noise_id in trange(8):
            noise_var = noises[noise_id]
            ret = []
            for trial in trange(1, 101):
                ret.append(frokf(noise_var, trial, ndim, dtype, eval(f"con_terms_{dtype}{ndim}"), eval(f"con_terms{ndim}")))
            sio.savemat(f"{root}FROKF_{dtype}{ndim}D_{noise_var:2.2f}", {'frokf_coef': np.stack(ret)})
