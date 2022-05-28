# 代码10-9

import pandas as pd
import numpy as np
media3 = pd.read_csv('../tmp/media3.csv', header='infer', error_bad_lines=False)

# 构建家庭成员标签
live_label = pd.read_csv('../data/table_livelabel.csv', encoding='gbk')
# 时间列存在很多种写法，而且存在隔天的情况
live_label.开始时间 = pd.to_datetime(live_label.开始时间)
# 将时间列变成datetime类型，好比较大小
live_label.结束时间 = pd.to_datetime(live_label.结束时间)
live_label['origin_time1'] = live_label.开始时间.apply(lambda x: 
    x.second + x.minute * 60 + x.hour * 3600)
live_label['end_time1'] = live_label.结束时间.apply(lambda x: 
    x.second + x.minute * 60 + x.hour * 3600)
print('查看星期:', live_label.星期.unique())
# 有些节目跨夜，需进行隔夜处理
def geyechuli_xingqi(x):
    dic = {'星期一':'星期二', '星期二':'星期三', '星期三':'星期四', '星期四':'星期五',
           '星期五':'星期六', '星期六':'星期日', '星期日':'星期一'}
    return x.apply(lambda y: dic[y.星期], axis=1)
ind1 = live_label.结束时间 < live_label.开始时间
label1 = live_label.loc[ind1, :].copy()
# 日期可以变，后面以end_time比较
live_label.loc[ind1, '结束时间'] = pd.Timestamp('2018-06-07 23:59:59')
live_label.loc[ind1, 'end_time1'] = 24 * 3600
label1.iloc[:, 1] = pd.Timestamp('2018-06-07 00:00:00')
label1.iloc[:, -2] = 0
label1.iloc[:, 0] = geyechuli_xingqi(label1)
label = pd.concat([live_label, label1])
label = label.reset_index(drop = True)  # 恢复默认索引

data_pindao = media3.copy()
label_ = label.loc[:, ['星期', 'origin_time1', 'end_time1', '频道', '适用人群']]
label_.columns = ['星期', 'origin_time1', 'end_time1', 'station_name', '适用人群']
media_ = data_pindao.loc[:, ['phone_no', '星期', 'origin_time1', 
                             'end_time1', 'station_name', ]]
family_ = pd.merge(media_, label_, how = 'left', on=['星期', 'station_name'])
f = np.array(family_.loc[:, ['origin_time1_x', 'end_time1_x', 
                             'origin_time1_y', 'end_time1_y']])

# lebel中的栏目记录分为四类：一类是只看了后半截，一类是全部都看了，
# 一类是只看了前半截，一类是看了中间一截
n1 = np.apply_along_axis(lambda x:
    (x[0] > x[2])&(x[0] < x[3])&(x[1] >= x[3]) , 1, f)  # 1是行，2是列
n2 = np.apply_along_axis(lambda x:
    ((x[0] <= x[2])&(x[1] >= x[3])) , 1, f)
n3 = np.apply_along_axis(lambda x:
    ((x[1] > x[2])&(x[1] < x[3])&(x[0] <=x [2])), 1, f)
n4 = np.apply_along_axis(lambda x:
    ((x[0] > x[2])&(x[1] < x[3])), 1, f)
da1 = family_.loc[n1, :].copy()
da1['wat_time'] = da1.end_time1_y - da1.origin_time1_x
da2 = family_.loc[n2, :].copy()
da2['wat_time'] = da2.end_time1_y - da2.origin_time1_y
da3 = family_.loc[n3, :].copy()
da3['wat_time'] = da3.end_time1_x - da3.origin_time1_y
da4= family_.loc[n4, :].copy()
da4['wat_time'] = da4.end_time1_x - da4.origin_time1_x
sd = pd.concat([da1, da2, da3, da4])
grouped = pd.DataFrame(sd['wat_time'].groupby([sd['phone_no'], sd['适用人群']]).sum())
grouped1 = pd.DataFrame(data_pindao['wat_time'].groupby([data_pindao['phone_no']]).sum())
phone_no = []
for i in range(len(grouped)):
    id = grouped.index[i][0]
    if id in grouped1.index.unique():
        shang = grouped['wat_time'][i] / grouped1[grouped1.index==id]
        if shang.values > 0.16:
            phone_no.append(grouped.index[i][0])
    else:
        continue
grouped2 = grouped.reset_index()

# 找出满足0.16标准的用户的家庭成员
aaa = pd.DataFrame(np.zeros([0, 3]), columns = grouped2.columns)
for k in phone_no:
    aaa = pd.concat([aaa, grouped2.ix[grouped2.iloc[:, 0]== k, :]], axis=0)
a = [aaa.ix[aaa['phone_no'] == k, '适用人群'].tolist() for k in aaa['phone_no'].unique()]
a = pd.Series([pd.Series(a[i]).unique() for i in range(len(a))])
a = pd.DataFrame(a)
b = pd.DataFrame(aaa['phone_no'].unique())
c = pd.concat([a, b], axis=1)
c.columns = ['家庭成员', 'phone_no']
grouped1 = grouped1.reset_index()
users_label = pd.merge(grouped1, c, left_on='phone_no', right_on ='phone_no', how='left')

# 构建电视依赖度标签
di = media3.phone_no.value_counts().values < 10
users_label['电视依赖度'] = 0
users_label.loc[di, '电视依赖度'] = '低'
zhong_gao = [i for i in users_label.index if i not in di]
num = media3.phone_no.value_counts()
for i in zhong_gao:
    if (users_label.loc[i, 'wat_time'] / num.iloc[i]) <= 3000:
        users_label.loc[i, '电视依赖度'] = '中'
users_label.loc[users_label.电视依赖度 == 0, '电视依赖度'] = '高'

# 构建机顶盒名称标签
jidinghe = media3.ix[media3['res_type'] == 1, :]
jdh = jidinghe.res_name.groupby(jidinghe.phone_no).unique()
jdh = jdh.reset_index()
jdh.columns = ['phone_no', '机顶盒名称']
users_label = pd.merge(users_label, jdh, left_on='phone_no', right_on ='phone_no', how='left')

# 观看时间偏好（周末）
media_watch = media3.loc[:, ['phone_no', 'origin_time', 'end_time', 'res_type',
                             '星期', 'wat_time']]
media_f1 = media_watch.ix[media_watch['星期'] == '星期六', :]
media_f2 = media_watch.ix[media_watch['星期'] == '星期日', :]
media_freeday = pd.concat([media_f1, media_f2], axis=0)
media_freeday = media_freeday.reset_index(drop = True)  # 恢复默认索引
'''
由于观看时间段偏好（工作日）与观看时间偏好（周末）的计算方式相似，
所以此处不在列出观看时间段偏好（工作日）的计算代码
'''

# 分割日期和时间，按空格号分开
T1 = [str(media_freeday.ix[i, 1]).split(' ') for i in list(media_freeday.index)]
# T1是列表，time[i] = T1[[i]][2]表示T1中第i个列表的第二列赋值给time的第i个
media_freeday['origin_time'] = [' '.join(['2018/06/09', T1[i][1]]) for i in media_freeday.index]
media_freeday['origin_time'] = pd.to_datetime(media_freeday['origin_time'], 
             format = '%Y/%m/%d %H:%M')
point = ['2018/06/09 00:00:00', '2018/06/09 06:00:00', '2018/06/09 09:00:00', 
         '2018/06/09 11:00:00', '2018/06/09 14:00:00', '2018/06/09 16:00:00',
         '2018/06/09 18:00:00', '2018/06/09 22:00:00', '2018/06/09 23:59:59']
lab = ['凌晨', '早晨', '上午', '中午', '下午', '傍晚', '晚上', '深夜']
sjd_num = pd.DataFrame()
for k in range(0, 8):
    kk = (media_freeday['origin_time'] >= point[k]) & \
    (media_freeday['origin_time'] < point[k+1])
    sjd = media_freeday.ix[kk==True, ['phone_no', 'wat_time']]
    sjd_new = sjd.groupby('phone_no').sum().sort_values('wat_time')
    sjd_new['时间段偏好（周末）'] = lab[k]
    sjd_num = pd.concat([sjd_num, sjd_new], axis=0)
sjd_num = sjd_num.reset_index()  # 增加索引
sjd_num = sjd_num.sort_values('phone_no')  # 以用户号排序
sjd_num = sjd_num.reset_index(drop = True)  # 增加默认索引
# 保留前3的标签
users = sjd_num['phone_no'].unique()
sjd_num_new = pd.DataFrame()
for m in users:
    gd = sjd_num.ix[sjd_num['phone_no'] == m, :]
    if len(gd)>3:
        gd = gd.sort_values('wat_time').iloc[::-1, :]
        gd = gd.iloc[:3, :]
    else:
        continue
    sjd_num_new = pd.concat([sjd_num_new, gd], axis=0)
sjd_label = sjd_num_new['时间段偏好（周末）'].groupby(sjd_num_new['phone_no']).sum()
sjd_label = sjd_label.reset_index()  # 增加索引
users_label = pd.merge(users_label, sjd_label, left_on='phone_no', 
                       right_on ='phone_no', how='left')

# 构建付费频道月均收视时长标签
import re
ffpd_ind =[re.search('付费', str(i))!=None for i in media3.ix[:, 'station_name']]
media_ffpd = media3.ix[ffpd_ind, :]
ffpd = media_ffpd['wat_time'].groupby(media_ffpd['phone_no']).sum()
ffpd = ffpd.reset_index()  # 增加索引
ffpd['付费频道月均收视时长'] = 0
for i in range(len(ffpd)):
    if ffpd.iloc[i, 1] < 3600:
        ffpd.iloc[i, 2] = '付费频道月均收视时长短'
    elif 3600 <= ffpd.iloc[i, 1] <= 7200:
        ffpd.iloc[i, 2] = '付费频道月均收视时长中'
    else:
        ffpd.iloc[i, 2] = '付费频道月均收视时长长'
ffpd = ffpd.loc[:, ['phone_no', '付费频道月均收视时长']]
users_label = pd.merge(users_label, ffpd, left_on='phone_no', 
                       right_on ='phone_no', how='left')
ffpd_ind = [str(users_label.iloc[i, 6]) == 'nan' for i in users_label.index]
users_label.ix[ffpd_ind, 6] = '付费频道无收视'

# 构建点播回看月均收视时长标签
media_res = media3.ix[media3['res_type'] == 1, :]
res = media_res['wat_time'].groupby(media_res['phone_no']).sum()
res = res.reset_index()  # 增加索引
res['点播回看月均收视时长'] = 0
for i in range(len(res)):
    if res.iloc[i, 1] < 10800:
        res.iloc[i, 2] = '点播回看月均收视时长短'
    elif 10800 <= res.iloc[i, 1] <= 36000:
        res.iloc[i, 2] = '点播回看月均收视时长中'
    else:
        res.iloc[i, 2] = '点播回看月均收视时长长'
res = res.loc[:, ['phone_no', '点播回看月均收视时长']]
users_label = pd.merge(users_label, res, left_on='phone_no', 
                       right_on ='phone_no', how='left')
res_ind = [str(users_label.iloc[i, 7]) == 'nan' for i in users_label.index]
users_label.ix[res_ind, 7] = '点播回看无收视'

# 体育偏好
media3.loc[media3['program_title'] == 'a', 'program_title'] = \
media3.loc[media3['program_title']=='a', 'vod_title']
program = [re.sub('\(.*', '', i) for i in media3['program_title']]  # 去除集数
program = [re.sub('.*月.*日', '', str(i)) for i in program]  # 去除日期
program = [re.sub('^ ', '', str(i)) for i in program]  # 前面的空格
program = [re.sub('\\d+$', '', i) for i in program]  # 去除结尾数字
program = [re.sub('【.*】', '', i) for i in program]  # 去除方括号内容
program = [re.sub('第.*季.*', '', i) for i in program]  # 去除季数
program = [re.sub('广告|剧场', '', i) for i in program]  # 去除广告、剧场字段
media3['program_title'] = program
ind = [media3.loc[i, 'program_title'] != '' for i in media3.index]
media_ = media3.loc[ind, :]
media_ = media_.drop_duplicates()  # 去重
media_.to_csv('../tmp/media4.csv', na_rep='NaN', header=True, index=False)

sports_ziduan = ['足球|英超|欧足|德甲|欧冠|国足|中超|西甲|亚冠|法甲|杰出球胜\
                 |女足|十分好球|亚足|意甲|中甲|足协|足总杯', '保龄球', 
                 'KHL|NHL|冰壶|冰球|冬奥会|花滑|滑冰|滑雪|速滑', 
                 'LPGA|OHL|PGA锦标赛|高尔夫|欧巡总决赛', '搏击|格斗|昆仑决|拳击\
                 |拳王','CBA|NBA|篮球|龙狮时刻|男篮|女篮', '女排|排球|男排', 
                 '乒超|乒乓|乒联、乒羽', '赛马', '车生活|劲速天地|赛车', 
                 '斯诺克|台球', '体操', '今日睇弹|竞赛快讯|世界体育|体坛点击|\
                 体坛快讯|体育晨报|体育世界|体育新闻',
                 'ATP|澳网|费德勒|美网|纳达尔|网球|中网', '象棋', '泳联|游泳|跳水', 
                 '羽超|羽联|羽毛球|羽乐无限', '自行车', 'NFL|超级碗|橄榄球', 
                 '马拉松', '飞镖|射击']
sports_mingzi = ['足球', '保龄球', '冰上运动', '高尔夫', '格斗', '篮球', '排球', 
                 '乒乓球', '赛马', '赛车', '台球', '体操', '体育新闻', '网球', 
                 '象棋', '游泳', '羽毛球', '自行车', '橄榄球', '马拉松', '射击']
sports_yuzhi = [1, 0.01, 0.58, 0.05, 0.08, 0.05, 0.04, 0.1, 0.02, 0.04, 0.07,
                0.01, 0.43, 0.13, 0.01, 0.02, 0.13, 0.01, 0.01, 0.01, 0.01]
sports_label = pd.DataFrame()
for k in range(len(sports_yuzhi)):
    sports = media_.ix[[re.search(sports_ziduan[k], 
                                  str(i))!=None for i in media_.ix[:, 'program_title']], :]
    sports['wat_time'] = sports['wat_time']/3600
    sports1 = sports['wat_time'].groupby(sports['phone_no']).sum()
    sports1 = sports1.reset_index()  # 增加索引
    sports1['体育偏好'] = 0
    for x in range(len(sports1)):
        if sports1.iloc[x, 1] >= sports_yuzhi[k]:
            sports1.iloc[x, 2] = sports_mingzi[k]
        else:
            continue
    sports_label = pd.concat([sports_label, sports1], axis=0)
sports_label = sports_label.ix[sports_label['体育偏好'] != 0, :]
sports_label_new = sports_label['体育偏好'].groupby(sports_label['phone_no']).sum()
sports_label_new = sports_label_new.reset_index()  # 增加索引
users_label = pd.merge(users_label, sports_label_new, left_on='phone_no', 
                       right_on ='phone_no', how='left')

# 剧场偏好
table_TV = pd.read_csv('../data/table_TV.csv', encoding='utf-8')
table_TV.columns = ['program_title', '剧场类型']
media_TV = media_.loc[:, ['phone_no', 'program_title', 'wat_time']]
TV_merge = pd.merge(media_TV, table_TV, left_on='program_title', 
                    right_on ='program_title', how='left')
TV_merge = TV_merge.ix[[str(TV_merge['剧场类型'][i]) != 'nan' \
                        for i in range(len(TV_merge))], :]
TV_merge['wat_time'] = TV_merge['wat_time']/3600
TV_merge1 = pd.DataFrame(TV_merge['wat_time'].groupby([
        TV_merge['phone_no'], TV_merge['剧场类型']]).sum())
TV_merge1 = TV_merge1.reset_index()  # 增加索引
TV_merge1['剧场偏好'] = 0
for x in range(len(TV_merge1)):
    if TV_merge1.iloc[x, 2] >= 1.15:
        TV_merge1.iloc[x, 3] = TV_merge1.iloc[x, 1]
    else:
        continue
TV_label = TV_merge1.ix[TV_merge1['剧场偏好'] != 0, :]
TV_label1 = TV_label['剧场偏好'].groupby([TV_label['phone_no']]).sum()
TV_label1 = TV_label1.reset_index()  # 增加索引
users_label = pd.merge(users_label, TV_label1, left_on='phone_no', 
                       right_on ='phone_no', how='left')
users_label.to_csv('../tmp/users_label1.csv', na_rep='NaN', 
                   header=True, index=False)
'''
由于财经爱好、生活爱好、电影爱好、娱乐爱好、教育爱好、新闻爱好和剧场爱好的计算方式有所相似，
所以在此处不在列出财经爱好、生活爱好、电影爱好、娱乐爱好、教育爱好、新闻爱好的计算代码
'''



# 代码10-10

users_label = pd.read_csv('../tmp/users_label1.csv', header='infer')
billevents2 = pd.read_csv('../tmp/billevents2.csv', header='infer')
# 消费内容
# 基于消费内容和用户号，对数据去重
print('查看费用类型：', billevents2.fee_code.unique())
fee1 = ['0B', '0T', '0D', '0H', '0X', '0R']
fee2 = ['基本收视维护费', '节目费', '互动电视点播费', '回看费', '互动电视信息费',
        '基本收视维护费']
billevents3 = pd.DataFrame()
for m in range(6):
    fee_gd = billevents2.ix[[billevents2.fee_code[i] == fee1[m] for i in billevents2.index], :]
    fee_gd['fee_code'] = fee2[m]
    billevents3 = pd.concat([billevents3, fee_gd], axis = 0)
isduplicated = billevents3.duplicated(['fee_code', 'phone_no'], keep='first')
billevents4 = [billevents3.ix[i, ] for i in list(billevents3.index) if isduplicated[i]==False]
billevents4 = pd.DataFrame(billevents4)       # 转为数据框
billevents_label = pd.DataFrame(billevents4['fee_code'].
                                groupby(billevents4['phone_no']).sum())
billevents_label = billevents_label.reset_index() # 增加索引
billevents_label.columns = ['phone_no', '消费内容']
users_label = pd.merge(users_label, billevents_label, left_on='phone_no', 
                       right_on ='phone_no', how='left')

# 家庭消费水平
bill_family = billevents2.copy()
bill_family['fee_pay'] = bill_family.should_pay - bill_family.favour_fee
family1 = pd.DataFrame(bill_family['fee_pay'].groupby([bill_family['phone_no'],
                                                      bill_family['terminal_no']]).sum())
family1 = family1.reset_index() # 增加索引
family1['家庭消费水平'] = 0
for i in range(len(family1)):
    if family1.iloc[i, 2] < 100:
        family1.iloc[i, 3] = '家庭消费水平低'
    elif 100 <= family1.iloc[i, 2] <= 220:
        family1.iloc[i, 3] = '家庭消费水平中'
    else:
        family1.iloc[i, 3] = '家庭消费水平高'
family2 = family1.loc[:, ['phone_no', '家庭消费水平']]
users_label = pd.merge(users_label, family2, left_on='phone_no', 
                       right_on ='phone_no', how='left')

# 电视消费水平
# 由于phone_no与terminal_no一一对应，所以家庭消费水平与电视消费水平一样
family1['电视消费水平'] = 0
for i in range(len(family1)):
    if (family1.iloc[i, 2]/3) < 26.5:
        family1.iloc[i, 4] = '电视消费水平超低'
    elif 26.5 <= (family1.iloc[i, 2]/3) < 46.5:
        family1.iloc[i, 4] = '电视消费水平低'
    elif 46.5 <= (family1.iloc[i, 2]/3) < 66.5:
        family1.iloc[i, 4] = '电视消费水平中'
    else:
        family1.iloc[i, 4] = '电视消费水平高'
family3 = family1.loc[:, ['phone_no', '电视消费水平']]
users_label = pd.merge(users_label, family3, left_on='phone_no', 
                       right_on ='phone_no', how='left')

# 电视爱好类别
bill_family = bill_family.reset_index(drop = True)
bill_dianbo = bill_family.ix[[bill_family.fee_code[i] == '0D' for i in bill_family.index], :]
fee_dianbo = bill_dianbo['fee_pay'].groupby(bill_dianbo['phone_no']).sum()
fee_dianbo = fee_dianbo.reset_index()
bill_huikan = bill_family.ix[[bill_family.fee_code[i] == '0H' for i in bill_family.index], :]
fee_huikan = bill_huikan['fee_pay'].groupby(bill_huikan['phone_no']).sum()
fee_huikan = fee_huikan.reset_index()
family1 = family1.sort_values(by=['phone_no']) # 以用户号排序
family4 = pd.merge(family1, fee_dianbo, left_on='phone_no', right_on ='phone_no', how='left')
family5 = pd.merge(family1, fee_huikan, left_on='phone_no', right_on ='phone_no', how='left')
family4['电视爱好类别'] = ''
family4.ix[family4.fee_pay_y/family4.fee_pay_x > 0.2, '电视爱好类别'] = '点播爱好者'
family5['电视爱好类别'] = ''
family5.ix[family5.fee_pay_y/family5.fee_pay_x > 0.2, '电视爱好类别'] = '回看爱好者'
family6 = pd.concat([family4, family5], axis=0)
family7 = family6['电视爱好类别'].groupby(family6['phone_no']).sum()
family7 = family7.reset_index()
users_label = pd.merge(users_label, family7.loc[:, ['phone_no', '电视爱好类别']], 
                       left_on='phone_no', right_on ='phone_no', how='left')

# 电视消费趋势
bill_xf = billevents2.copy()
bill_xf['fee_pay'] = bill_xf.should_pay - bill_xf.favour_fee
bill_xf1 = pd.DataFrame(bill_xf['fee_pay'].groupby(bill_xf['phone_no']).sum())
bill_xf1['电视消费趋势'] = ''
for i in bill_xf1.index:
    a = bill_xf.ix[bill_xf['phone_no'] == i, :]
    a['year_month'] = a['year_month'].astype('datetime64')
    b = [a['year_month'].iloc[n].month for n in range(len(a))]
    c1 = a.ix[[b[m] == 4 for m in range(len(b))], :]
    c2 = a.ix[[b[m] == 5 for m in range(len(b))], :]
    c3 = a.ix[[b[m] == 6 for m in range(len(b))], :]
    d1 = c1['fee_pay'].groupby(c1['phone_no']).sum()
    d2 = c2['fee_pay'].groupby(c2['phone_no']).sum()
    d3 = c3['fee_pay'].groupby(c3['phone_no']).sum()
    if (d1.values>=d2.values)&(d2.values>=d3.values):
        bill_xf1.loc[i, '电视消费趋势'] = '费用递减'
    elif (d1.values<=d2.values)&(d2.values<=d3.values):
        bill_xf1.loc[i, '电视消费趋势'] = '费用递增'
    else:
        bill_xf1.loc[i, '电视消费趋势'] = '费用不稳定'
bill_xf1 = bill_xf1.reset_index()
users_label = pd.merge(users_label, bill_xf1.loc[:, ['phone_no', '电视消费趋势']], 
                       left_on='phone_no', right_on ='phone_no', how='left')
users_label.to_csv('../tmp/users_label2.csv', na_rep='NaN', header=True, index=False)



# 代码10-11

import pandas as pd
users_label = pd.read_csv('../tmp/users_label2.csv', sep=',', header='infer')
order2 = pd.read_csv('../tmp/order2.csv', sep=',', header='infer', error_bad_lines=False)
# 销售品内容
order_offername = pd.DataFrame(order2['offername'].groupby(order2['phone_no']).last())
order_offername = order_offername.reset_index()
order_offername.columns = ['phone_no', '销售品名称']
users_label = pd.merge(users_label, order_offername, left_on='phone_no', 
                       right_on ='phone_no', how='left')

# 电视当前方式
import re
users_label['电视当前方式'] = '单一'
ind1 = [re.search('互动\+', str(i)) != None for i in users_label['销售品名称']]
users_label.ix[ind1, '电视当前方式'] = '套餐'
users_label.to_csv('../tmp/users_label3.csv', na_rep='NaN', header=True, index=False)
print('查看订单数据相关标签:\n', users_label[['销售品名称', '电视当前方式']][0:5])



# 代码10-12

import pandas as pd
users_label = pd.read_csv('../tmp/users_label3.csv', header='infer')
payevents = pd.read_csv('../tmp/payevents2.csv', header='infer')
# 最近支付方式
payevents_pay = pd.DataFrame(payevents['payment_name'].
                             groupby(payevents['phone_no']).last())
payevents_pay = payevents_pay.reset_index()
payevents_pay.columns = ['phone_no', '最近支付方式']
users_label = pd.merge(users_label, payevents_pay, left_on='phone_no', 
                       right_on ='phone_no', how='left')

# 最近缴费渠道
payevents_login = pd.DataFrame(payevents['login_group_name'].
                               groupby(payevents['phone_no']).last())
payevents_login = payevents_login.reset_index()
payevents_login.columns = ['phone_no', '最近缴费渠道']
users_label = pd.merge(users_label, payevents_login, left_on='phone_no', 
                       right_on ='phone_no', how='left')
users_label.to_csv('../tmp/users_label4.csv', na_rep='NaN', header=True, index=False)
print('查看收费数据中相关标签:\n', users_label[['最近支付方式', '最近缴费渠道']][0:5])



# 代码10-13

import pandas as pd
import time
users_label = pd.read_csv('../tmp/users_label4.csv', header='infer')
userevents = pd.read_csv('../tmp/userevents2.csv', header='infer')
# 电视入网时长
user_time = pd.DataFrame(userevents['run_time'].groupby(userevents['phone_no']).first())
user_time = user_time.reset_index()
user_time['run_time'] = pd.to_datetime(user_time.run_time)
sj = time.ctime()
sj = pd.to_datetime(sj)
user_time['电视入网时长'] = ''
for i in user_time.index:
    sc = sj.year - user_time.ix[i, 'run_time'].year
    if sc >= 6:
        user_time.ix[i, '电视入网时长'] = '老用户'
    elif 3 < sc < 6:
        user_time.ix[i, '电视入网时长'] = '中等用户'
    else:
        user_time.ix[i, '电视入网时长'] = '新用户'
users_label = pd.merge(users_label, user_time.loc[:, ['phone_no', '电视入网时长']],
                       left_on='phone_no', right_on ='phone_no', how='left')

# 业务品牌
user_sm = pd.DataFrame(userevents['sm_name'].groupby(userevents['phone_no']).last())
user_sm = user_sm.reset_index()
users_label = pd.merge(users_label, user_sm, left_on='phone_no', 
                       right_on='phone_no', how='left')
users_label.to_csv('../tmp/users_label5.csv', na_rep='NaN', header=True, index=False)
print('查看用户状态数据中相关标签:\n', users_label[['电视入网时长', 'sm_name']][0:5])



# 代码10-14

import pandas as pd
import time
media3 = pd.read_csv('../tmp/media3.csv', header='infer', error_bad_lines=False)
billevents2 = pd.read_csv('../tmp/billevents2.csv', header='infer', error_bad_lines=False)
userevents2 = pd.read_csv('../tmp/userevents2.csv', header='infer', error_bad_lines=False)

# 构造特征
# 观看总次数F
media_f = media3['phone_no'].value_counts()
media_f = media_f.reset_index()
media_f.columns = ['phone_no', '观看总次数']
# 观看总时长C
media_c = media3['wat_time'].groupby(media3['phone_no']).sum()
media_c = media_c.reset_index()
media_rfm = pd.merge(media_f, media_c, left_on='phone_no', right_on ='phone_no', 
                     how='left')
# 距最近观看时间R
media_r = media3['origin_time'].groupby(media3['phone_no']).last()
media_r = media_r.reset_index()
sj = time.ctime()
sj = pd.to_datetime(sj)
r = pd.Series([sj - pd.to_datetime(media_r.iloc[i, 1]) for i in media_r.index])
r = pd.to_datetime(r)
r = r.apply(lambda x: x.second + x.minute * 60 + x.hour * 3600 + x.day * 3600 * 24)
media_rfm = pd.concat([media_rfm, r], axis=1)
# 入网时长L
media_l = userevents2['run_time'].groupby(userevents2['phone_no']).first()
media_l = media_l.reset_index()
l = pd.Series([sj - pd.to_datetime(media_l.iloc[i, 1]) for i in media_l.index])
l = pd.to_datetime(l)
l = l.apply(lambda x: x.second + x.minute * 60 + x.hour * 3600 + x.day * 3600 * 24)
media_rfm = pd.concat([media_rfm, l], axis=1)
# 消费总金额M
billevents2['fee_pay'] = billevents2.should_pay - billevents2.favour_fee
media_m = pd.DataFrame(billevents2['fee_pay'].groupby(billevents2['phone_no']).sum())
media_m = media_m.reset_index()
media_rfm = pd.merge(media_rfm, media_m, left_on='phone_no', right_on ='phone_no',
                     how='left')

media_rfm = media_rfm.dropna() # 去除任何有空值的行
# 标准化
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
stdScaler = StandardScaler().fit(media_rfm.iloc[:, 1:6])
rfm_std = stdScaler.transform(media_rfm.iloc[:, 1:6])

# K-Means聚类，5类
kmeans = KMeans(n_clusters=5, random_state=123).fit(rfm_std) # 构建并训练模型

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 使用TSNE进行数据降维, 降成两维
tsne = TSNE(n_components=2, init='random', random_state=177).fit(rfm_std)
df = pd.DataFrame(tsne.embedding_) # 将原始数据转换为DataFrame
df['labels'] = kmeans.labels_ # 将聚类结果存储进df数据表
# 提取不同标签的数据
df1 = df[df['labels'] == 0]
df2 = df[df['labels'] == 1]
df3 = df[df['labels'] == 2]
df4 = df[df['labels'] == 3]
df5 = df[df['labels'] == 4]
# 绘制图形
fig = plt.figure(figsize=(9, 6)) # 设定空白画布，并制定大小
# 用不同的颜色表示不同数据
plt.plot(df1[0], df1[1], 'bo', df2[0], df2[1], 'r*', df3[0], df3[1], 'gD',
         df4[0], df4[1], 'kH', df5[0], df5[1], 'y+')
plt.legend([0,1,2,3,4])
plt.show() ##显示图片

# 绘制雷达图
import numpy as np
N = len(kmeans.cluster_centers_[0])
# 设置雷达图的角度，用于平分切开一个圆面
angles = np.linspace(0, 2*np.pi, N, endpoint=False) 
angles = np.concatenate((angles, [angles[0]])) # 为了使雷达图一圈封闭起来
# 绘图
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, polar=True) # 这里一定要设置为极坐标格式
sam = ['r-', 'o-', 'g-', 'b-', 'p-'] # 样式
lstype = ['-',':','--','-.','-.']
lab = []
for i in range(len(kmeans.cluster_centers_)):
    values = kmeans.cluster_centers_[i]
    feature = ['F', 'C', 'R', 'L', 'M']
    # 为了使雷达图一圈封闭起来，需要下面的步骤
    values = np.concatenate((values, [values[0]]))
    ax.plot(angles, values, sam[i], linestyle=lstype[i], linewidth=2) # 绘制折线图
    ax.fill(angles, values, alpha=0.25) # 填充颜色
    ax.set_thetagrids(angles * 180 / np.pi, feature) # 添加每个特征的标签
    ax.set_ylim(-2, 6) # 设置雷达图的范围
    plt.title('客户群特征分布图') # 添加标题
    ax.grid(True) # 添加网格线
    lab.append('客户群' + str(i))
plt.legend(lab)
plt.show() # 显示图形

# 加入到标签表
media_rfm = pd.concat([media_rfm, df['labels']], axis=1)
kmeans.cluster_centers_ # K-Means的聚类结果
media_rfm.ix[media_rfm['labels'] == 0, 'labels'] = '重要发展用户'
media_rfm.ix[media_rfm['labels'] == 1, 'labels'] = '低价值用户'
media_rfm.ix[media_rfm['labels'] == 2, 'labels'] = '重要挽留用户'
media_rfm.ix[media_rfm['labels'] == 3, 'labels'] = '重要保持用户'
media_rfm.ix[media_rfm['labels'] == 4, 'labels'] = '一般用户'
users_label = pd.read_csv('../tmp/users_label5.csv', header='infer')
users_label = pd.merge(users_label, media_rfm.loc[:, ['phone_no', 'labels']], 
                       left_on='phone_no', right_on ='phone_no', how='left')
users_label.to_csv('../tmp/users_label6.csv', na_rep='NaN', header=True, index=False)



# 代码10-15

from bs4 import BeautifulSoup
import requests
import time
# 准备翻页处理
url = 'https://www.tvmao.com/program/CCTV-CCTV5-w1.html'
# 根据url的特点构造连续7页的CCTV5的URL
urls = ['https://www.tvmao.com/program/CCTV-CCTV5-w{}.html'.format(str(i)) \
        for i in range(1, 8)]

st_times = []
tv_names = []
links_l = []
header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 \
          (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36 LBBROWSER'}
for url in urls: 
    time.sleep(3) # 设置间隔3秒请求一次浏览器
    wb_data = requests.get(url, headers=header) # 请求服务器
    soup = BeautifulSoup(wb_data.text, 'html.parser') # 解析网页
    # 电视开始的时间
    st_times += [st_time.get_text() for st_time in soup.select('div.over_hide span.am')]
    # 电视的节目名称
    tv_names += [tv_name.get_text() for tv_name in soup.select('div.over_hide span.p_show')]
    links_l += ['https://www.tvmao.com' + link.get('href') for link in \
                soup.select('div.over_hide span.p_show a')]

# 通过循环获取跳转页面里面的节目分类和简介
game_class = []
game_intro = []
for in_url in links_l:
    in_wb_data = requests.get(in_url, headers = header)
    in_soup = BeautifulSoup(in_wb_data.text, 'html.parser')
    if in_url.__contains__('tvcolumn'):# 判断
        game_class.append(in_soup.select('tr td')[3].get_text())
        game_intro.append(in_soup.select('div.lessmore.clear p')[0].get_text())
    else:
        game_class.append(in_soup.select('tr td span')[1].get_text())
        game_intro.append(in_soup.select('article p')[0].get_text())
# 创建数据框
import pandas as pd
data = {
    'st_time' : st_times,
    'tv_name' : tv_names,
    'game_class' : game_class,
    'game_intro' : game_intro
}
df = pd.DataFrame(data)
df.to_csv('../tmp/cctv5_spider.csv', index=False, sep=',') # 保存数据
print('查看获取的前5条数据:\n', df.ix[:, :5])
