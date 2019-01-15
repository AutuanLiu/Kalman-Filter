"""重要工具包之工具使用

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/1/14
"""

from core_tools import update_terms, plot_FROLS_term, plot_Kalman_term

# !更新候选项
update_terms(data_root='data/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'})

# !可视化 FROLS 算法中 选择的候选项和 ERR 之间的关系
plot_FROLS_term(data_root='data/', img_root='images/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'})

# !可视化 Kalman 算法中 选择的候选项和 ERR 之间的关系
plot_Kalman_term(data_root='data/', img_root='images/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'})
