# 代码 6-3

# 使用基于UBCF算法对电影进行推荐
import pandas as pd
from recommender import recomm  # 加载自编推荐函数

# 读入数据
traindata = pd.read_csv('../data/u1.base', sep='\t', header=None, index_col=None)
testdata = pd.read_csv('../data/u1.test', sep='\t', header=None, index_col=None)
# 删除时间标签列
traindata.drop(3, axis=1, inplace=True)
testdata.drop(3, axis=1, inplace=True)
# 行与列重新命名
traindata.rename(columns={0: 'userid', 1: 'movid', 2: 'rat'}, inplace=True)
testdata.rename(columns={0: 'userid', 1: 'movid', 2: 'rat'}, inplace=True)
traindf = traindata.pivot(index='userid', columns='movid', values='rat')
testdf = testdata.pivot(index='userid', columns='movid', values='rat')
traindf.rename(index={i: 'usr%d' % (i) for i in traindf.index}, inplace=True)
traindf.rename(columns={i: 'mov%d' % (i) for i in traindf.columns}, inplace=True)
testdf.rename(index={i: 'usr%d' % (i) for i in testdf.index}, inplace=True)
testdf.rename(columns={i: 'mov%d' % (i) for i in testdf.columns}, inplace=True)
userdf = traindf.loc[testdf.index]
# 获取预测评分和推荐列表
trainrats, trainrecomm = recomm(traindf, userdf)
print('用户预测评分的前5行：\n', trainrats.head())

# 保存预测的评分
trainrats.to_csv('../tmp/movie_comm.csv', index=False, encoding='utf-8')
print('用户推荐列表的前5行：\n', trainrecomm[:5])



# 代码 6-4

import pandas as pd

# 读入数据
traindata = pd.read_csv('../data/u1.base', sep='\t', header=None, index_col=None)
testdata = pd.read_csv('../data/u1.test', sep='\t', header=None, index_col=None)
# 删除时间标签列
traindata.drop(3, axis=1, inplace=True)
testdata.drop(3, axis=1, inplace=True)
# 行与列重新命名
traindata.rename(columns={0: 'userid', 1: 'movid', 2: 'rat'}, inplace=True)
testdata.rename(columns={0: 'userid', 1: 'movid', 2: 'rat'}, inplace=True)
# 构建训练集数据
user_tr = traindata.iloc[:, 0]  # 训练集用户id
mov_tr = traindata.iloc[:, 1]  # 训练集电影id
user_tr = list(set(user_tr))  # 去重处理
mov_tr = list(set(mov_tr))  # 去重处理
print('训练集电影数：', len(mov_tr))

# 利用训练集数据构建模型
ui_matrix_tr = pd.DataFrame(0, index=user_tr, columns=mov_tr)
# 求用户－物品矩阵
for i in traindata.index:
    ui_matrix_tr.loc[traindata.loc[i, 'userid'], traindata.loc[i, 'movid']] = 1
print('训练集用户观影次数：', sum(ui_matrix_tr.sum(axis=1)))

# 求物品相似度矩阵（因计算量较大，需要耗费的时间较久）
item_matrix_tr = pd.DataFrame(0, index=mov_tr, columns=mov_tr)
for i in item_matrix_tr.index:
    for j in item_matrix_tr.index:
        a = sum(ui_matrix_tr.loc[:, [i, j]].sum(axis=1) == 2)
        b = sum(ui_matrix_tr.loc[:, [i, j]].sum(axis=1) != 0)
        item_matrix_tr.loc[i, j] = a / b
# 将物品相似度矩阵对角线处理为零
for i in item_matrix_tr.index:
    item_matrix_tr.loc[i, i] = 0
# 利用测试集数据对模型评价
user_te = testdata.iloc[:, 0]
mov_te = testdata.iloc[:, 1]
user_te = list(set(user_te))
mov_te = list(set(mov_te))
# 测试集数据用户物品矩阵
ui_matrix_te = pd.DataFrame(0, index=user_te, columns=mov_te)
for i in testdata.index:
    ui_matrix_te.loc[testdata.loc[i, 'userid'], testdata.loc[i, 'movid']] = 1
# 对测试集用户进行推荐
res = pd.DataFrame('NaN', index=testdata.index, columns=['User', '已观看电影', '推荐电影', 'T/F'])
res.loc[:, 'User'] = list(testdata.iloc[:, 0])
res.loc[:, '已观看电影'] = list(testdata.iloc[:, 1])
# 开始推荐
for i in res.index:
    if res.loc[i, '已观看电影'] in list(item_matrix_tr.index):
        res.loc[i, '推荐电影'] = item_matrix_tr.loc[res.loc[i, '已观看电影'], :].argmax()
        if res.loc[i, '推荐电影'] in mov_te:
            res.loc[i, 'T/F'] = ui_matrix_te.loc[res.loc[i, 'User'], res.loc[i, '推荐电影']] == 1
        else:
            res.loc[i, 'T/F'] = False
# 保存推荐结果
res.to_csv('../tmp/res_mov.csv', index=False, encoding='utf8')
print('推荐结果前5行: \n', res.head())
