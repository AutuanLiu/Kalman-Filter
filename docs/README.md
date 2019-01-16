# 代码逻辑

```bash
Email: autuanliu@163.com
```

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

1. update_terms

### FROLS 算法实验

1. FROLS_estimation.m

### FROLS 算法实验可视化分析

1. plot_Kalman_term

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

1. plot_Kalman_term

## 关键点

1. 一定要注意代码的运行顺序
2. 一定要主要保持各个代码中 max lag 的设置的一致性
