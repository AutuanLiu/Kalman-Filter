"""
Email: autuanliu@163.com
Date: 2018/12/12
"""

from .estimator import *
from .regression import *
from .selector import *
from .tools import *
from .utils import *

__all__ = [
    'Kalman4ARX', 'Kalman4FROLS', 'torch4FROLS', 'regression4torch', 'regression4sklearn', 'TermsData', 'Selector', 'grid_search1', 'grid_search2',
    'grid_search3', 'plot_grid_search', 'get_mat_data', 'normalize', 'save_2Darray', 'save_3Darray', 'Timer', 'make_linear_func', 'make_func4K4FROLS',
    'make_dataset4SK', 'Linear', 'get_txt_data', 'update_condidate_terms', 'get_terms_matrix', 'plot_term_ERR'
]

__version__ = '0.6.0'
