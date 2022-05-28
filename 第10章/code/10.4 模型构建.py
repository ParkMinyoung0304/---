# 代码10-16

import pandas as pd
media = pd.read_csv('../tmp/media4.csv', header='infer')

# 协同过滤算法
m = media.loc[:, ['phone_no', 'program_title']]
n = 500000
media2 = m.iloc[:n, :]
media2['value'] = 1
from sklearn.model_selection import train_test_split
# 将数据划分为训练集测试集
media_train, media_test = train_test_split(media2, test_size=0.2, random_state=123)

# 长表转宽表，即用户-物品矩阵
mat1 = media_train.pivot_table(index='phone_no', columns='program_title') # 透视表
mat1.columns = [i[1] for i in mat1.columns]
mat1.fillna(0, inplace=True) # 0填充
df_matrix1 = mat1
df_matrix1 = df_matrix1 / df_matrix1.sum(axis=0) * 5     # 0-5之间

from sklearn.metrics.pairwise import pairwise_distances
# 计算余弦相似性
# metric可以设置欧式距离/曼哈顿距离/余弦夹角（euclidean/manhattan/cosine）
item_similarity = 1 - pairwise_distances(df_matrix1.T, metric='cosine')

# 对角线设为0
a = range(item_similarity.shape[0])
item_similarity[a, a] = 0
item_similarity = pd.DataFrame(item_similarity)
item_similarity.index = item_similarity.columns = df_matrix1.columns

# 测试集上预测评分高的前五推荐
phone_test = media_test.phone_no.unique()
result1 = pd.DataFrame()
for i in range(len(phone_test)):
    res1 = pd.DataFrame({'phone':[phone_test[i]] * 5,
                       'program':(df_matrix1.iloc[i] * item_similarity).sum(axis=1)
                       .sort_values(ascending=False).index[:5].tolist()})
    result1 = result1.append(res1)
print('预测评分高的前五名推荐\n', result1)


# 代码10-17

import pandas as pd
import re
# 读取爬取的数据
cctv_5 = pd.read_csv('../tmp/cctv5_spider.csv', sep=',')
# 根据数据的特征修改类型
cctv_5.game_class.unique()
cctv_5.game_class = [re.sub('花样滑冰', '冰上运动', str(i)) for i in cctv_5.game_class]
cctv_5.game_class = [re.sub('体育 生活', '生活', str(i)) for i in cctv_5.game_class]
cctv_5.game_class = [re.sub('体育 新闻', '体育新闻', str(i)) for i in cctv_5.game_class]
cctv_5.game_class = [re.sub('其他', '射击', str(i)) for i in cctv_5.game_class]
cctv_5.game_class = [re.sub('拳击', '格斗', str(i)) for i in cctv_5.game_class]

# Simple TagBased TF-IDF算法的模型
# 提取出收视表中有体育类标签的数据
media1 = pd.read_csv('../tmp/media4.csv', header='infer')
sports_ziduan = ['足球|英超|欧足|德甲|欧冠|国足|中超|西甲|亚冠|法甲|杰出球胜|女足|\
                 十分好球|亚足|意甲|中甲|足协|足总杯', '保龄球', 'KHL|NHL|冰壶|冰球|\
                 冬奥会|花滑|滑冰|滑雪|速滑', 'LPGA|OHL|PGA锦标赛|高尔夫\
                 |欧巡总决赛', '搏击|格斗|昆仑决|拳击|拳王', 'CBA|NBA|篮球|龙狮时刻\
                 |男篮|女篮','女排|排球|男排', '乒超|乒乓|乒联、乒羽', '赛马', 
                 '车生活|劲速天地|赛车', '斯诺克|台球', '体操', 
                 '今日睇弹|竞赛快讯|世界体育|体坛点击|体坛快讯|体育晨报|体育世界\
                 |体育新闻', 'ATP|澳网|费德勒|美网|纳达尔|网球|中网', 
                 '象棋', '泳联|游泳|跳水', '羽超|羽联|羽毛球|羽乐无限', '自行车', 
                 'NFL|超级碗|橄榄球', '马拉松', '飞镖|射击']
sports_mingzi = ['足球', '保龄球', '冰上运动', '高尔夫', '格斗', '篮球', '排球'
                 , '乒乓球', '赛马', '赛车', '台球', '体操', '体育新闻', '网球',
                 '象棋', '游泳', '羽毛球', '自行车', '橄榄球', '马拉松', '射击']
sports_data = pd.DataFrame()
for k in range(len(sports_mingzi)):
    sports = media1.ix[[re.search(sports_ziduan[k], str(i)) != None for i in \
                        media1.ix[:, 'program_title']], :]
    sports['wat_time'] = sports['wat_time'] / 3600
    sports['体育偏好'] = sports_mingzi[k]
    sports_data = pd.concat([sports_data, sports], axis=0)

# 计算各用户每个体育偏好标签的时长
group1 = pd.DataFrame(sports_data['wat_time'].groupby(
        [sports_data['phone_no'], sports_data['体育偏好']]).sum())
group1 = group1.reset_index()
# 计算各用户每个体育偏好标签的次数
sports_data['counts'] = 1
group2 = pd.DataFrame(sports_data['counts'].groupby(
        [sports_data['phone_no'], sports_data['体育偏好']]).sum())
group2 = group2.reset_index()
# 将数据合并
group = pd.merge(group1, group2, left_on=['phone_no', '体育偏好'], 
                 right_on =['phone_no', '体育偏好'], how='left')
# 对用户观看时间进行加权
group['weight_counts'] = group.counts * round(group.wat_time, 2)
# 对用户观看具有该标签节目的次数进行加权
import math
label_c = group.体育偏好.value_counts() # 每个用户拥有的标签数
label_c = label_c.reset_index()
new_group = pd.DataFrame()
for i in label_c.index:
    no = label_c.iloc[i, 0]
    weight = math.log(1 + label_c.iloc[i, 1])
    g = group.loc[group.体育偏好 == no, :]
    g['weight_label1'] = g.weight_counts / weight
    new_group = pd.concat([new_group, g], axis=0)
# 将每个用户按偏好标签的指数排序
new_ = new_group.sort_values(['phone_no', 'weight_label1'], ascending=False)

guess = 0
while True:
    input_no = int(input('Please input one phone_no that is in group:'))
    guess += 1
    recommend_list = []
    if input_no in list(new_.phone_no):     # 检查是否为已存在的用户号
        n = new_.loc[new_.phone_no == input_no, '体育偏好']
        for k in n.values:
            recommend_list.extend(cctv_5.loc[cctv_5['game_class'] == k, 'tv_name'])
        print('It is only %d,phone_no is %d. \nRecommend_list is \n' % (guess, input_no), 
              pd.DataFrame(recommend_list[:20], columns=['program']))
    elif input_no == 0:
        print('Stop recommend!')
        break
    else:
        print('Please input phone_no that is in group.')
'''
当输入16899545095时，即可为用户名为16899545095的用户推荐推荐指数排名前20的节目
当输入0时，即可结束为用户进行推荐
'''


# 代码10-18

import pandas as pd
media1 = pd.read_csv('../tmp/media4.csv', header='infer')

# 流行度模型
from sklearn.model_selection import train_test_split
# 将数据划分为训练集测试集
media1_train, media1_test = train_test_split(media1, test_size=0.2, random_state=1234)

# 将节目按热度排名
program = media1_train.program_title.value_counts()
program = program.reset_index()
program.columns = ['program', 'counts']

recommend_dataframe = pd.DataFrame
m = 3000
while True:
    input_no = int(input('Please input one phone_no that is not in group:'))
    if input_no == 0:
        print('Stop recommend!')
        break
    else:
        recommend_dataframe = pd.DataFrame(program.iloc[:m, 0], columns=['program'])
        print('Phone_no is %d. \nRecommend_list is \n' % (input_no), 
              recommend_dataframe)
'''
当输入16801274792时，即可为用户名为16801274792的用户推荐推荐最热门的前N个节目
当输入0时，即可结束为用户进行推荐
'''