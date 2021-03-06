# 代码逻辑

```bash
Email: autuanliu@163.com
```

[文档下载](https://workuse.nos-eastchina1.126.net/Docs/Kalman%E6%BB%A4%E6%B3%A2%E4%BC%B0%E8%AE%A1%E7%BA%BF%E6%80%A7ARX%E6%A8%A1%E5%9E%8B%E7%B3%BB%E6%95%B0.pdf)

- [代码逻辑](#%E4%BB%A3%E7%A0%81%E9%80%BB%E8%BE%91)
  - [核心组件](#%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6)
  - [关键步骤](#%E5%85%B3%E9%94%AE%E6%AD%A5%E9%AA%A4)
    - [全局步骤](#%E5%85%A8%E5%B1%80%E6%AD%A5%E9%AA%A4)
    - [更新候选项集合](#%E6%9B%B4%E6%96%B0%E5%80%99%E9%80%89%E9%A1%B9%E9%9B%86%E5%90%88)
    - [FROLS 算法实验](#frols-%E7%AE%97%E6%B3%95%E5%AE%9E%E9%AA%8C)
    - [FROLS 算法实验可视化分析](#frols-%E7%AE%97%E6%B3%95%E5%AE%9E%E9%AA%8C%E5%8F%AF%E8%A7%86%E5%8C%96%E5%88%86%E6%9E%90)
    - [Kalman Filter 算法实验](#kalman-filter-%E7%AE%97%E6%B3%95%E5%AE%9E%E9%AA%8C)
    - [Kalman Filter 算法实验可视化分析](#kalman-filter-%E7%AE%97%E6%B3%95%E5%AE%9E%E9%AA%8C%E5%8F%AF%E8%A7%86%E5%8C%96%E5%88%86%E6%9E%90)
  - [关键点](#%E5%85%B3%E9%94%AE%E7%82%B9)
  - [结果示例](#%E7%BB%93%E6%9E%9C%E7%A4%BA%E4%BE%8B)
    - [FROLS](#frols)
    - [Kalman](#kalman)

## 核心组件

组件名称 | 组件作用
--- | ---
kalman_estimation | 基于 Kalman Filter 的系数估计器
FROLS | FROLS 算法
term_selector | 候选项选择器(matlab)
tools | 核心工具包，主要用于各种系数估计算法的比较与可视化
data | 生成仿真数据与存储实验结果
images | 存储信号信息可视化结果和各种算法实验结果的可视化
docs | 保存文档
examples | 测试各种系数估计算法
kalman-estimation.Selector | 候选项选择器, 生成候选项表达式
kalman_estimation.tools | 网格搜索与可视化工具包
kalman_estimation.utils | 数据导入、数据标准化、数据保存、计时器、算法估计表达式计算
kalman_estimation.regression | 回归算法、候选项数据集设计
kalman_estimation.estimator | 基于 Kalman Filter 的系数估计器算法，如 Kalman4ARX, Kalman4FROLS, torch4FROLS
term_selector.terms_maker.m | 数据标准化、计算候选项
tools.core_tools | 为实验设计的核心工具
tools.update_terms_main.py | 更新候选项
tools.visualization_main.py | 可视化操作
tools.kalman_main.py | 基于 Kalman 滤波器的算法流程 flow, pipeline
FROLS.FROLS_estimator.m | 使用 FROLS 算法估计系数并保存结果
FROLS.FROLS_est.py | 将 FROLS 算法估计的结果保存成表达式并保存结果

## 关键步骤

**Notes:** 以下各个分支步骤存在先后依赖关系，具体细节：

1. 先执行 **全局步骤**
2. *可视化* 前必须执行 **更新候选项集合**

### 全局步骤

1. make_sim_data.m
2. terms_makers.m

### 更新候选项集合

1. update_terms(update_terms_main.py)

### FROLS 算法实验

1. FROLS_estimation.m
2. FROLS_est.py

### FROLS 算法实验可视化分析

1. plot_Kalman_term(visualization_main.py)

### Kalman Filter 算法实验

1. 导入数据
2. kalman-estimation.Selector.make_selection()
3. 构造估计器

    ```python
    kf = torch4FROLS(normalized_signals, Kalman_H, n_epoch=100)
    ```

4. 估计系数

    ```python
    y_coef = kf.estimate_coef()
    ```

5. 生成表达式

    ```python
    make_func4K4FROLS
    ```

### Kalman Filter 算法实验可视化分析

1. plot_Kalman_term(visualization_main.py)

## 关键点

1. 一定要注意代码的运行顺序
2. 一定要主要保持各个代码中 max lag 的设置的一致性
3. 运行 matlab 文件时，请将工作目录添加到路径中，另外运行某个文件时，请切换到可执行文件的路径处
4. 运行 python 文件时，直接在工作目录处运行即可

## 结果示例

### FROLS

![](https://workuse.nos-eastchina1.126.net/Github/Images/FROLS_linear.png)

### Kalman

![](https://workuse.nos-eastchina1.126.net/Github/Images/Kalman_linear.png)
