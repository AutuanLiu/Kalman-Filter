"""
Email: autuanliu@163.com
Date: 2018/12/12
"""

import math

import numpy as np
import torch
from torch.nn import Module, Parameter, init
from torch.utils.data import Dataset

__all__ = ['Linear', 'regression4torch', 'regression4sklearn', 'TermsData', 'make_dataset4SK']


class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = x@A + b`

    REF: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L11

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
            
    Attributes:
        weight: the learnable weights of the module
        bias:   the learnable bias of the module
    """

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is not None:
            return input @ self.weight + self.bias
        else:
            return input @ self.weight

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def regression4torch(in_dim, out_dim):
    """基于 PyTorch 的线性回归模型 Y = X*W +b
    
    原有的 nn.Linear() 不适用于这里的问题，所以定义 Linear 如上

    Args:
        in_dim (int): 输入维度
        out_dim (int): 输出维度
    """

    return Linear(in_dim, out_dim, bias=False)


def regression4sklearn():
    """基于 scikit-learn 的线性回归模型
    """

    from sklearn.linear_model import LinearRegression
    return LinearRegression(fit_intercept=False)


class TermsData(Dataset):
    """构造关于候选项的数据集
    
    Attributes:
        data (torch.Tensor): 数据
        target (torch.Tensor): 目标值
    """

    def __init__(self, signals, Kalman_H):
        """构造函数
        
        Args:
            signals (np.array): 原始信号数据
            Kalman_H (np.array): Kalman 候选项矩阵
        """

        super().__init__()
        n_dim, n_point, _ = Kalman_H.shape
        max_lag = signals.shape[0] - Kalman_H.shape[1]
        data = []
        for t in range(n_point):
            Cn = np.kron(np.eye(n_dim), Kalman_H[:, t, :])
            data.append(Cn[slice(0, Cn.shape[0], n_dim + 1)])
        self.data = torch.as_tensor(np.array(data)).float()
        self.target = torch.as_tensor(signals[max_lag:]).float()

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.size(0)


def make_dataset4SK(signals, Kalman_H):
    """构造用于 scikit-learn 使用的数据集
        
    Args:
        signals (np.array): 原始信号数据
        Kalman_H (np.array): Kalman 候选项矩阵
    """

    n_dim, n_point, _ = Kalman_H.shape
    max_lag = signals.shape[0] - Kalman_H.shape[1]
    data = []
    for t in range(n_point):
        Cn = np.kron(np.eye(n_dim), Kalman_H[:, t, :])
        data.append(Cn[slice(0, Cn.shape[0], n_dim + 1)])
    return np.array(data), signals[max_lag:]
