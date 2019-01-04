# 估计器使用指北

```python
Email: autuanliu@163.com
Date: 2018/12/13
```

所有的计算都使用经过 **标准化** 的数据，使得系数估计不受不同通道信号之间的数值差异的影响。

## [Kalman4ARX 估计器](./kalman_ARX.py)

**思路**：使用 Linear Kalman Filter 对 ARX 模型进行系数估计

- 基于 Python3.6.6 开发

1. step 1
   - 数据导入与标准化

2. step 2
   - 构造估计器并评估系数

3. 可选功能
   - 计时
      ```python
      from core import Timer
      timer = Timer()
      timer.start()
      # some steps
      timer.stop()
      ```
   - 保存结果
      ```python
      from pathlib import Path
      from core import save_2Darray, save_3Darray
      # 保存结果2D
      # Write the array to disk
      file_path0 = Path('./kalman_filter/data/y_coef.txt')
      save_2Darray(file_path0, y_coef)

      # 保存结果3D
      # Write the array to disk
      file_path1 = Path('./kalman_filter/data/A_coef.txt')
      save_3Darray(file_path1, A_coef)
      ```
   - 显示估计的函数表达式
      ```python
      from core import make_linear_func
      # make func
      est_model = make_linear_func(A_coef, var_name='x', fname='./kalman_filter/data/linear_est_model.txt')
      print(est_model) # 也可以选择保存
      ```

## [Kalman4FROLS 估计器](./kalman_NARX.py)

**思路**：使用 Linear Kalman Filter 和基于 FROLS 的模型候选项选择器 对 ARX or NARX 模型进行系数估计

- 基于 Python3.6.6 + MATLAB 开发

1. step 1
   - 数据导入(基于 matlab 代码的运行结果, 数据已经经过标准化)

2. step 2
   - 构造 Selector 实例，用于生成估计器所需要的数据 **(sparse -> dense)**

3. step 3
   - 估计器构建与估计系数

## [torch4FROLS 估计器](./kalman_NARX.py)

**思路**：使用 Linear Regression 和基于 FROLS 的模型候选项选择器 对 ARX or NARX 模型进行系数估计，可以看做是 FROLS 的拆分版

- 基于 Python3.6.6 + MATLAB + PyTorch 实现
- 可以设置训练次数和学习率

1. step 1
   - 数据导入(基于 matlab 代码的运行结果, 数据已经经过标准化)

2. step 2
   - 构造 Selector 实例，用于生成估计器所需要的数据 **(sparse -> dense)**

3. step 3
   - 估计器构建与估计系数

## terms_maker

1. step 1
   - buildH
    ```matlab
    [H, Hv] = buildH(normalized_signals, norder, max_lag);
    ```

2. step 2
   - term_selector
    ```matlab
    [Kalman_H, sparse_H, S, S_No] = term_selector(normalized_signals, norder, max_lag, H, threshold);
    ```

Notes: 示例代码 [example code](./test.py)
