import unittest

import numpy as np

from kalman_estimation import (Kalman4ARX, Kalman4FROLS, Selector, get_mat_data, torch4FROLS)


class Test_kalman_estimation(unittest.TestCase):
    def test_Kalman4ARX(self):
        file_path = 'test_data/linear_signals5D_noise1.mat'
        data = get_mat_data(file_path, 'linear_signals')
        kf = Kalman4ARX(data, 4, uc=0.01)
        y, A = kf.estimate_coef(0.1)
        self.assertTrue(isinstance(kf, Kalman4ARX))
        self.assertTrue(isinstance(A, np.ndarray))

    def test_Kalman4FROLS(self):
        terms_path = 'test_data/linear_terms.mat'
        term = Selector(terms_path)
        normalized_signals, Kalman_H, _, _ = term.make_selection()
        kf = Kalman4FROLS(normalized_signals, Kalman_H=Kalman_H, uc=0.01)
        y_coef = kf.estimate_coef()
        self.assertTrue(isinstance(kf, Kalman4FROLS))
        self.assertTrue(isinstance(y_coef, np.ndarray))

    def test_Selector(self):
        terms_path = 'test_data/linear_terms.mat'
        term = Selector(terms_path)
        terms_repr = term.make_terms()
        self.assertTrue(isinstance(term, Selector))
        self.assertTrue(isinstance(terms_repr, np.ndarray))

    def test_torch4FROLS(self):
        terms_path = 'test_data/linear_terms.mat'
        term = Selector(terms_path)
        normalized_signals, Kalman_H, _, _ = term.make_selection()
        kf = torch4FROLS(normalized_signals, Kalman_H, n_epoch=100)
        y_coef = kf.estimate_coef()
        self.assertTrue(isinstance(kf, torch4FROLS))
        self.assertTrue(isinstance(y_coef, np.ndarray))


if __name__ == '__main__':
    unittest.main()
