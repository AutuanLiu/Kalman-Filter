"""重要工具包之工具使用

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/1/14
"""

from core_tools import update_terms

# !更新候选项
update_terms(data_root='data/', data_type_set={'linear', 'nonlinear', 'longlag_linear', 'longlag_nonlinear'})
