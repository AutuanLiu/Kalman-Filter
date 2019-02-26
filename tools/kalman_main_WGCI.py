"""重要工具包 WGCI version

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/2/23
"""

from core_tools_WGCI import (get_json_data, kalman4ARX_pipeline, kalman4FROLS_pipeline, torch4FROLS_pipeline)
from kalman_estimation import Timer, get_mat_data, make_linear_func

data_type_set = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}

# !step 1: import data
# !step 2: 创建估计器
# !step 3: 估计系数
# !step 4: 计算估计的模型表达式并保存
# !step 5: 结果可视化

timer = Timer()
timer.start()

configs = get_json_data('tools/config1.json')

# 多次实验 kalman4ARX
for data_type in ['linear', 'longlag_linear']:
    kalman4ARX_pipeline(data_type, configs, configs[data_type]['n_trial'])

# 多次实验 kalman4FROLS_pipeline  3分钟左右
for data_type in configs.keys():
    kalman4FROLS_pipeline(data_type, configs, configs[data_type]['n_trial'])

# 多次实验 torch4FROLS_pipeline
# for data_type in configs.keys():
#     torch4FROLS_pipeline(data_type, configs, configs[data_type]['n_trial'])

timer.stop()
