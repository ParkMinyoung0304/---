# 代码 6-1

import pandas as pd
from apriori import *  # 导入自行编写的apriori函数
# 读入数据
data = pd.read_excel('../data/menu_orders.xls', header=None)
data.head()

print('\n转换原始数据至0-1矩阵...')
ct = lambda x: pd.Series(1, index=x[pd.notnull(x)])  # 转换0-1矩阵的过渡函数
b = map(ct, data.as_matrix())  # 用map方式执行
data = pd.DataFrame(list(b)).fillna(0)  # 实现矩阵转换，空值用0填充
print('\n转换完毕')

data.head()

support = 0.2  # 最小支持度
confidence = 0.5  # 最小置信度
# 连接符，默认'--'，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符
ms = '---'
find_rule(data, support, confidence, ms).to_excel('../tmp/outputfile.xls', encoding='utf-8')  # 保存结果



# 代码 6-2

import fpGrowth

# 读取数据并转换格式
newsdata = [line.split() for line in open('../data/kosarak.dat').readlines()]
indataset = fpGrowth.createInitSet(newsdata)
# 构建树寻找其中浏览次数在5万次以上的新闻
news_fptree, news_headertab = fpGrowth.createTree(indataset, 50000)
# 创建空列表用于保持频繁项集
newslist = []
fpGrowth.mineTree(news_fptree, news_headertab, 50000, set([]), newslist)
# 查看结果
print('浏览次数在5万次以上的新闻报导集合个数：', len(newslist))
print('浏览次数在5万次以上的新闻：\n', newslist)
