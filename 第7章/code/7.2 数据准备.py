# 代码 7-1

import numpy as np
import pandas as pd

data = pd.read_csv('../data/data.csv')  # 读取数据
# 保留两位小数，并将结果保存为’.csv’文件
np.round(data.corr(method = 'pearson'), 2).to_csv('../tmp/data_cor.csv')
print('相关系数矩阵为：\n', np.round(data.corr(method = 'pearson'), 2))
