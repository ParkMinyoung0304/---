# 代码9-1

import pandas as pd
airline_data = pd.read_csv('../data/air_data.csv')
print('原始数据的形状为：', airline_data.shape)
# 去除票价为空的记录
index_not_na1 = airline_data['SUM_YR_1'].notnull() 
index_not_na2 = airline_data['SUM_YR_2'].notnull()
index_not_na = index_not_na1 & index_not_na2
airline_notnull = airline_data.loc[index_not_na, :]
print('删除缺失记录后数据的形状为：', airline_notnull.shape)

# 丢弃票价为0，或平均折扣率为0，或总飞行公里数为0的记录 
index1 = airline_notnull['SUM_YR_1'] == 0
index2 = airline_notnull['SUM_YR_2'] == 0
index3 = (airline_notnull['SEG_KM_SUM']== 0) | \
    (airline_notnull['avg_discount'] == 0)  
index_drop = airline_notnull.index[(index1 & index2) | index3]
airline = airline_notnull.drop(index_drop, axis=0)
airline.to_csv('../tmp/air_data_clean.csv')
print('删除异常记录后数据的形状为：', airline.shape)
