

# Kalman filter estimation

```bash
Email: autuanliu@163.com
```

**！！！本库的所有文件，作者保留一切版权，在未经作者许可，请不要擅自使用或者发表！！！**

[![Build Status](https://dev.azure.com/AutuanLiu/Kalman%20Filter/_apis/build/status/AutuanLiu.Kalman-Filter?branchName=master&jobName=Test&configuration=Python36)](https://dev.azure.com/AutuanLiu/Kalman%20Filter/_build/latest?definitionId=1?branchName=master)
![](https://img.shields.io/pypi/dm/kalman-estimation.svg)
![](https://img.shields.io/github/repo-size/badges/shields.svg)
![](https://img.shields.io/github/license/AutuanLiu/Kalman-Filter.svg)
![](https://img.shields.io/github/release/AutuanLiu/Kalman-Filter.svg)
![](https://img.shields.io/pypi/v/kalman-estimation.svg)
![](https://img.shields.io/pypi/pyversions/kalman-estimation.svg)
![](https://img.shields.io/github/release-date/AutuanLiu/Kalman-Filter.svg)
![](https://img.shields.io/github/languages/top/AutuanLiu/Kalman-Filter.svg)
[![](https://img.shields.io/badge/Chinese_Docs-pass-brightgreen.svg)](https://www.yuque.com/xk6dxn/drboi7)
[]()
[![](https://img.shields.io/badge/Wiki-pass-brightgreen.svg)](https://dev.azure.com/AutuanLiu/Kalman%20Filter/_wiki/wikis/Kalman-Filter.wiki?wikiVersion=GBwikiMaster&pagePath=%2FKalman%20filter%20estimation)
[![Sourcegraph](https://sourcegraph.com/github.com/AutuanLiu/Kalman-Filter/-/badge.svg?style=flat)](https://sourcegraph.com/github.com/AutuanLiu/Kalman-Filter?badge)

* Docs
  - [中文文档](https://www.yuque.com/xk6dxn/drboi7)
  - [Wiki](https://dev.azure.com/AutuanLiu/Kalman%20Filter/_wiki/wikis/Kalman-Filter.wiki?wikiVersion=GBwikiMaster&pagePath=%2FKalman%20filter%20estimation)
  - [Sourcegraph](https://sourcegraph.com/github.com/AutuanLiu/Kalman-Filter)

- [Kalman filter estimation](#Kalman-filter-estimation)
  - [1 Theory](#1-Theory)
    - [1.1 线性一维系统](#11-%E7%BA%BF%E6%80%A7%E4%B8%80%E7%BB%B4%E7%B3%BB%E7%BB%9F)
      - [1.1.1 系统表示](#111-%E7%B3%BB%E7%BB%9F%E8%A1%A8%E7%A4%BA)
      - [1.1.2 计算过程](#112-%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B)
    - [1.2 线性多维系统](#12-%E7%BA%BF%E6%80%A7%E5%A4%9A%E7%BB%B4%E7%B3%BB%E7%BB%9F)
      - [1.2.1 系统表示](#121-%E7%B3%BB%E7%BB%9F%E8%A1%A8%E7%A4%BA)
      - [1.2.2 计算过程](#122-%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B)
    - [1.3 非线性多维系统](#13-%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%A4%9A%E7%BB%B4%E7%B3%BB%E7%BB%9F)
      - [1.3.1 系统表示](#131-%E7%B3%BB%E7%BB%9F%E8%A1%A8%E7%A4%BA)
      - [1.3.2 计算过程](#132-%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B)
  - [Reference](#Reference)
  - [Info](#Info)

Notes: **所有的原始数据文件可以使用data目录下的matlab代码生成**


* 本库包含四种自回归模型系数估计的算法
  * FROLS
  * bi-KF
  * FROKF（暂未发表，但可用, 引用请联系作者）
  * bi-KF-SGD（暂未发表，但可用, 引用请联系作者）

* 主题

  * 卡尔曼滤波器
  * 自回归模型
  * 系数估计
  * 格兰杰因果
  * FROLS（Forward-Regression Orthogonal Least Square）
  * FROKF（暂未发表，但可用, 引用请联系作者）
  * SGD

* 编程语言
  * Matlab
  * Python

* FROKF 系数估计

  ![FROKF](./images/steps.png)

* FROKF 效果示意

  * 估计系数的均值比较
  
   ![估计系数的均值比较](./images/coef_com_mean.png)

  * 估计系数的方差比较

    ![估计系数的方差比较](./images/coef_com_var.png)

  * 估计系数的误差比较

    ![估计系数的误差比较](./images/coef_com_err.png)

* 安装
```bash
pip install kalman-estimation
```

## 1 Theory

### 1.1 线性一维系统

#### 1.1.1 系统表示

$$x_k=ax_{k-1}+bu_k+w_k$$

$$z_k=cx_k+v_k$$

$$p(w)\sim\mathcal{N}(0, Q)$$

$$p(v)\sim\mathcal{N}(0, R)$$

#### 1.1.2 计算过程

- step 1: Predict

$$\hat{{x}_k}=a\hat{{x}_{k-1}}+bu_k$$

$$p_k=ap_{k-1}a + Q$$

- step 2: Update

$$g_k=p_k c/(cp_k c+r)$$

$$\hat{x}_k\leftarrow \hat{x}_k+g_k(z_k-c\hat{x}_k)$$

$$p_k\leftarrow (1-g_k c)p_k$$
**以上的过程(step1 && step2)是在观测序列上递归计算的。以上为离散版本(一维)的kalman滤波。**

### 1.2 线性多维系统

#### 1.2.1 系统表示

$$x_k=Ax_{k-1}+Bu_k+w_k$$

$$z_k=Cx_k+v_k$$

#### 1.2.2 计算过程

- step 1: Predict

$$\hat{{x}_k}=A\hat{{x}_{k-1}}+Bu_k$$

$$P_k=AP_{k-1}A^T+Q$$

- step 2: Update

$$G_k=P_k C^T(C{P_k} C^T+R)^{-1}$$

$$\hat{x}_k\leftarrow \hat{x}_k+G_k(z_k-C\hat{x}_k)$$

$$P_k\leftarrow (I-G_k C)P_k$$

这里的 $x$ 可以是向量 $\vec{x}$，用来表示多个信号。

### 1.3 非线性多维系统

#### 1.3.1 系统表示

$$x_k=f(x_{k-1},u_k)+w_k$$

$$z_k=h(x_k)+v_k$$

#### 1.3.2 计算过程

- step 1: Predict

$$\hat{{x}_k}=f(\hat{{x}_{k-1}},u_k)$$

$$P_k=F_{k-1}P_{k-1}F_{k-1}^T+Q_{k-1}$$

- step 2: Update

$$G_k=P_k H_k^T(H_k{P_k} H_k^T+R)^{-1}$$

$$\hat{x}_k\leftarrow \hat{x}_k+G_k(z_k-h(\hat{x}_k))$$

$$P_k\leftarrow (I-G_k H_k)P_k$$

这里 $F_{k-1}$, $H_k$ 分别表示非线性函数 $f$, $h$ 的雅克比矩阵。

## Reference

1. [An implementation of kalman-filters for multivariate time-series in PyTorch](https://github.com/strongio/torch-kalman)
2. [Analysis of financial time series using Kalman filter.](https://github.com/noureldien/TimeSeriesAnalysis)
3. [Self-Driving Car Nanodegree Program Starter Code for the Extended Kalman Filter Project](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project)
4. [Python Kalman filtering and optimal estimation library. Implements Kalman filter, particle filter, Extended Kalman filter, Unscented Kalman filter, g-h (alpha-beta), least squares, H Infinity, smoothers, and more. Has companion book 'Kalman and Bayesian Filters in Python'.](https://github.com/rlabbe/filterpy)
5. [FilterPy — FilterPy 1.4.4 documentation](https://filterpy.readthedocs.io/en/latest/)
6. [Kalman Filter book using Jupyter Notebook. Focuses on building intuition and experience, not formal proofs. Includes Kalman filters,extended Kalman filters, unscented Kalman filters, particle filters, and more. All exercises include solutions.](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
7. [Header-only C++11 Kalman Filtering Library (EKF, UKF) based on Eigen3](https://github.com/mherb/kalman)
8. [**The Extended Kalman Filter: An Interactive Tutorial**](https://home.wlu.edu/~levys/kalman_tutorial/)
9. [Lightweight C/C++ Extended Kalman Filter with Python for prototyping](https://github.com/simondlevy/TinyEKF)
10. [CoursePack.book](http://www.cs.unc.edu/~tracker/media/pdf/SIGGRAPH2001_CoursePack_08.pdf)
11. [Kalman Filter: An Algorithm for making sense from the insights of various sensors fused together.](https://towardsdatascience.com/kalman-filter-an-algorithm-for-making-sense-from-the-insights-of-various-sensors-fused-together-ddf67597f35e)
12. [kalman_intro_chinese.pdf](https://www.cs.unc.edu/~welch/kalman/media/pdf/kalman_intro_chinese.pdf)
13. [autoregressive model - Different state-space representations for Auto-Regression and Kalman filter - Signal Processing Stack Exchange](https://dsp.stackexchange.com/questions/2197/different-state-space-representations-for-auto-regression-and-kalman-filter)
14. [14_state_space.pdf](http://www-stat.wharton.upenn.edu/~stine/stat910/lectures/14_state_space.pdf)
15. [kalman.pdf](https://eml.berkeley.edu/~rothenbe/Fall2007/kalman.pdf)
16. Linderman, S. W., Johnson, M. J., Miller, A. C., Adams, R. P., Blei, D. M., & Paninski, L. (2017). Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, 54, 914–922.
17. Sieb, M., Schultheis, M., & Szelag, S. (2018). Probabilistic Trajectory Segmentation by Means of Hierarchical Dirichlet Process Switching Linear Dynamical Systems. Retrieved from http://arxiv.org/abs/1806.06063
18. E. B. Fox, “Bayesian nonparametric learning of com- plex dynamical phenomena,” 2009
19. A. Fasoula, Y. Attal, and D. Schwartz, “Comparative performance evaluation of data-driven causality measures applied to brain networks,” J. Neurosci. Methods, vol. 215, no. 2, pp. 170–189, 2013.

## Info

```bash
$ cloc .
      95 text files.
      91 unique files.                              
      37 files ignored.

github.com/AlDanial/cloc v 1.74  T=54.92 s (1.1 files/s, 98.8 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
MATLAB                          27            200            668           1336
Python                          22            502            813           1157
Markdown                         6            152              1            404
JSON                             3              0              0            131
YAML                             2             11              5             43
Bourne Shell                     1              0              0              3
-------------------------------------------------------------------------------
SUM:                            61            865           1487           3074
-------------------------------------------------------------------------------
```
