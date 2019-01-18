"""重要工具包

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/1/16
"""

from kalman_estimation import get_mat_data
from core_tools import get_json_data, kalman_pipeline

data_type_set = {'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'}

# !step 1: import data
# !step 2: 创建估计器
# !step 3: 估计系数
# !step 4: 计算估计的模型表达式并保存
# !step 5: 结果可视化

configs = get_json_data('tools/config.json')
kalman_pipeline(configs)
