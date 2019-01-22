"""重要工具包

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/1/16
"""

from kalman_estimation import get_mat_data, make_linear_func, Timer
from core_tools import get_json_data, kalman_pipeline, kalman4ARX_pipeline, kalman4FROLS_pipeline, torch4FROLS_pipeline

data_type_set = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}

# !step 1: import data
# !step 2: 创建估计器
# !step 3: 估计系数
# !step 4: 计算估计的模型表达式并保存
# !step 5: 结果可视化

timer = Timer()
timer.start()

configs = get_json_data('tools/config.json')
# kalman_pipeline(configs)

# 多次实验 kalman4ARX
n_trial = 100
A_coef = 0
for data_type in ['linear', 'longlag_linear']:
    kalman4ARX_pipeline(data_type, configs, n_trial)

# 多次实验 kalman4FROLS_pipeline
for data_type in configs.keys():
    kalman4FROLS_pipeline(data_type, configs, n_trial)

# 多次实验 kalman4FROLS_pipeline
for data_type in configs.keys():
    torch4FROLS_pipeline(data_type, configs, n_trial)

timer.stop()
