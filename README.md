# Kalman filter estimation

```bash
Email: autuanliu@163.com
```

- [Kalman filter estimation](#kalman-filter-estimation)
  - [1 Theory](#1-theory)
    - [1.1 线性一维系统](#11-%E7%BA%BF%E6%80%A7%E4%B8%80%E7%BB%B4%E7%B3%BB%E7%BB%9F)
      - [1.1.1 系统表示](#111-%E7%B3%BB%E7%BB%9F%E8%A1%A8%E7%A4%BA)
      - [1.1.2 计算过程](#112-%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B)
    - [1.2 线性多维系统](#12-%E7%BA%BF%E6%80%A7%E5%A4%9A%E7%BB%B4%E7%B3%BB%E7%BB%9F)
      - [1.2.1 系统表示](#121-%E7%B3%BB%E7%BB%9F%E8%A1%A8%E7%A4%BA)
      - [1.2.2 计算过程](#122-%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B)
    - [1.3 非线性多维系统](#13-%E9%9D%9E%E7%BA%BF%E6%80%A7%E5%A4%9A%E7%BB%B4%E7%B3%BB%E7%BB%9F)
      - [1.3.1 系统表示](#131-%E7%B3%BB%E7%BB%9F%E8%A1%A8%E7%A4%BA)
      - [1.3.2 计算过程](#132-%E8%AE%A1%E7%AE%97%E8%BF%87%E7%A8%8B)
  - [Reference](#reference)
  - [Info](#info)

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

## Info

```bash
$ cloc .
      35 text files.
      35 unique files.
      22 files ignored.

github.com/AlDanial/cloc v 1.80  T=0.50 s (40.0 files/s, 4424.0 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                           9            301            506            575
MATLAB                           8             49            153            289
Markdown                         3             98              0            241
-------------------------------------------------------------------------------
SUM:                            20            448            659           1105
-------------------------------------------------------------------------------
```
