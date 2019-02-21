"""结果可视化比较

Copyright:
----------
    Author: AutuanLiu
    Date: 2019/2/21
"""

import numpy as np
from kalman_estimation import get_mat_data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

data_path = './tools/result.mat'
result = get_mat_data(data_path, 'data')

