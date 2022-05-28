# 代码9-2

import pandas as pd
import numpy as np

airline = pd.read_csv('../data/air_data.csv')
# 选取需求特征
airline_selection = airline[['FFP_DATE', 'LOAD_TIME', 'FLIGHT_COUNT', 
                             'LAST_TO_END', 'avg_discount', 'SEG_KM_SUM']]
# 构建L特征
L = pd.to_datetime(airline_selection['LOAD_TIME']) - \
pd.to_datetime(airline_selection['FFP_DATE'])
L = L.astype('str').str.split().str[0]
L = L.astype('int') / 30
# 合并特征
airline_features1 = pd.concat([L, airline_selection.iloc[:, 2:]], axis = 1)
airline_features1.columns = ['L', 'F', 'R', 'C', 'M']
airline_features = pd.DataFrame(np.zeros([len(airline_features1), 5]), 
                                columns = ['L', 'R', 'F', 'M', 'C'])
for i in range(len(airline_features.columns)):
    airline_features.ix[:, airline_features.columns[i]] = \
    list(airline_features1.ix[:, airline_features.columns[i]])
print('构建的L、R、F、M、C特征前5行为：\n', airline_features.head())




# 代码9-3

# 查看特征取值范围
explore = airline_features.describe(percentiles = [], include = 'all')
explore = explore.ix[['min', 'max'], :]
print('L、R、F、M、C 5个特征取值范围：\n', explore )




# 代码9-4

# 数据标准化
from sklearn.preprocessing import StandardScaler
air_scale = StandardScaler().fit_transform(airline_features)
np.savez('../tmp/airline_scale.npz', air_scale)
print('标准化后L、R、F、M、C 5个特征为：\n', air_scale[:5, :])