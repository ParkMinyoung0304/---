# 代码 7-2

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

data = pd.read_csv('../data/data.csv')  # 读取数据
# 调用Lasso()函数，设置λ的值为1000
lasso = Lasso(1000)
lasso.fit(data.iloc[:, 0:13], data['y'])
print('相关系数为：', np.round(lasso.coef_, 5))  # 输出结果，保留五位小数

print('相关系数非零个数为：', np.sum(lasso.coef_ != 0))  # 计算相关系数非零的个数

# 返回一个相关系数是否为零的布尔数组
mask = lasso.coef_ != 0
print('相关系数是否为零：', mask)

data = data.iloc[:, 0:13]
new_reg_data = data.iloc[:, mask]  # 返回相关系数非零的数据
new_reg_data.to_csv('../tmp/new_reg_data.csv')  # 存储数据
print('输出数据的维度为：', new_reg_data.shape)  # 查看输出数据的维度
