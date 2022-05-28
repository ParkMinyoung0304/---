# 代码2-11

from scipy.interpolate import lagrange  # 导入拉格朗日插值函数
import pandas as pd
data = pd.read_excel('../data/null.xlsx', index_col = 0)
print('原始数据缺失值个数为：', sum(data['x'].isnull()))
print('使用fillna插补后的数据为：\n', data.fillna(0))
print('使用dropna删除空值后的数据为：\n', data.dropna())
# s为列向量，n为缺失值位置，取缺失值前后k个数据，默认为5
def ployinterp_column(s, n, k=5):
  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))]  # 取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n)  # 插值并返回插值结果
# 逐个元素判断是否需要插值
for j in data.index:
    if (data['x'].isnull())[j]: #如果为空即插值。
        data['x'][j] = ployinterp_column(data['x'], j)
        print('插补值为：', data['x'][j])
print('拉格朗日插值插补后缺失值个数为：', sum(data['x'].isnull()))



