# 代码10-1

import pandas as pd

# 处理收视行为信息数（media-index）
media = pd.read_csv('../data/media_index.csv', encoding='gbk', 
                    header='infer', error_bad_lines=False)

# 将高清替换为空
media['station_name'] = media['station_name'].str.replace('-高清', '')

# 过滤特殊线路、政企用户
media = media.ix[(media.owner_code != 2)&(media.owner_code != 9)&
                 (media.owner_code != 10), :]
print('查看过滤后的特殊线路的用户:', media.owner_code.unique())
  # 查看是否去除完成
media = media.ix[(media.owner_name !='EA级')&(media.owner_name !='EB级')&
                 (media.owner_name !='EC级')&(media.owner_name !='ED级')&
                 (media.owner_name !='EE级'), :]
print('查看过滤后的政企用户:', media.owner_name.unique())

# 对开始时间进行拆分
# 检查数据类型
type(media.ix[0, 'origin_time'])
# 转化为时间类型
media['end_time'] = pd.to_datetime(media['end_time'])
media['origin_time'] = pd.to_datetime(media['origin_time'])
# 提取秒
media['origin_second']=media['origin_time'].dt.second
media['end_second'] = media['end_time'].dt.second
# 筛选数据
ind1 = (media['origin_second']==0) & (media['end_second'] == 0)
media1 = media.ix[~ind1, :]
# 基于开始时间和结束时间的记录去重
media1.end_time = pd.to_datetime(media1.end_time)
media1.origin_time = pd.to_datetime(media1.origin_time)
media1 = media1.drop_duplicates(['origin_time', 'end_time'])

# 隔夜处理
# 去除开始时间，结束时间为空值的数据
media1 = media1.loc[media1.origin_time.dropna().index, :]
media1 = media1.loc[media1.end_time.dropna().index, :]
# 创建星期特征列
media1['星期'] = media1.origin_time.apply(lambda x: x.weekday()+1)
dic = {1:'星期一', 2:'星期二', 3:'星期三', 4:'星期四', 5:'星期五', 6:'星期六', 7:'星期日'}
for i in range(1, 8):
    ind = media1.loc[media1['星期']==i, :].index
    media1.loc[ind, '星期'] = dic[i]
# 查看有多少观看记录是隔天的，隔天的进行隔天处理
a = media1.origin_time.apply(lambda x :x.day)
b = media1.end_time.apply(lambda x :x.day)
sum(a != b)
media2 = media1.loc[a!=b, :].copy()  # 需要做隔天处理的数据
def geyechuli_xingqi(x):
    dic = {'星期一':'星期二','星期二':'星期三','星期三':'星期四','星期四':'星期五',
           '星期五':'星期六','星期六':'星期日','星期日':'星期一'}
    return x.apply(lambda y: dic[y.星期], axis=1)
media1.loc[a!=b, 'end_time'] = media1.loc[a!=b, 'end_time'].apply(lambda x:
    pd.to_datetime('%d-%d-%d 23:59:59'%(x.year, x.month, x.day)))
media2.loc[:, 'origin_time'] = pd.to_datetime(media2.end_time.apply(lambda x:
    '%d-%d-%d 00:00:01'%(x.year, x.month, x.day)))
media2.loc[:, '星期'] = geyechuli_xingqi(media2)
media3 = pd.concat([media1, media2])
media3['origin_time1'] = media3.origin_time.apply(lambda x:
    x.second + x.minute * 60 + x.hour * 3600)
media3['end_time1'] = media3.end_time.apply(lambda x: 
    x.second + x.minute * 60 + x.hour * 3600)
media3['wat_time'] = media3.end_time1 - media3.origin_time1  # 构建观看总时长特征

# 清洗时长不符合的数据
# 剔除下次观看的开始时间小于上一次观看的结束时间的记录
media3 = media3.sort_values(['phone_no', 'origin_time'])
media3 = media3.reset_index(drop=True)
a = [media3.ix[i+1, 'origin_time'] < media3.ix[i, 'end_time'] for i in range(len(media3)-1)]
a.append(False)
aa = pd.Series(a)
media3 = media3.loc[~aa, :]
# 去除小于4S的记录
media3 = media3.loc[media3['wat_time']> 4, :] 
media3.to_csv('../tmp/media3.csv', na_rep='NaN', header=True, index=False)

# 查看连续观看同一频道的时长是否大于3h，发现这2000个用户不存在连续观看大于3h的情况
media3['date'] = media3.end_time.apply(lambda x :x.date())
media_group = media3['wat_time'].groupby([media3['phone_no'],
                                         media3['date'],
                                         media3['station_name']]).sum()
media_group = media_group.reset_index()
media_g = media_group.loc[media_group['wat_time'] >= 10800, ]
media_g['time_label'] = 1
o = pd.merge(media3, media_g, left_on=['phone_no', 'date', 'station_name'],
             right_on =['phone_no', 'date', 'station_name'], how='left')
oo = o.loc[o['time_label']==1, :]



# 代码10-2

# 处理账单数据（mmconsume-billevents）与收费数据（mmconsume-payevents）
billevents = pd.read_csv('../data/mmconsume_billevents.csv', encoding='gbk', header='infer')
billevents.columns = ['phone_no', 'fee_code', 'year_month', 'owner_name', 
                      'owner_code', 'sm_name', 'should_pay', 'favour_fee', 
                      'terminal_no']
# 基于处理账单数据过滤特殊线路、政企用户
billevents = billevents.ix[(billevents.owner_code != 2)&
                           (billevents.owner_code != 9)&
                           (billevents.owner_code != 10), :]
print('查看过滤后的特殊线路的用户:', billevents.owner_code.unique())
billevents = billevents.loc[(billevents.owner_name !='EA级')&
                            (billevents.owner_name !='EB级')&
                            (billevents.owner_name !='EC级')&
                            (billevents.owner_name !='ED级')&
                            (billevents.owner_name !='EE级'), :]
print('查看过滤后的政企用户:', billevents.owner_name.unique())
billevents.to_csv('../tmp/billevents2.csv', na_rep='NaN', header=True, index=False)

payevents = pd.read_csv('../data/mmconsume_payevents.csv', sep=',', 
                        encoding='gbk', header='infer')
payevents.columns = ['phone_no', 'owner_name', 'event_time', 'payment_name', 
                     'login_group_name', 'owner_code']

# 基于消费数据过滤特殊线路、政企用户
payevents = payevents.ix[(payevents.owner_code != 2
                          )&(payevents.owner_code != 9
                          )&(payevents.owner_code != 10), :] # 去除特殊线路数据
payevents.owner_code.unique() #查看是否去除完成
payevents = payevents.loc[(payevents.owner_name != "EA级"
                           )&(payevents.owner_name!= "EB级"
                           )&(payevents.owner_name != "EC级"
                           )&(payevents.owner_name!= "ED级"
                           )&(payevents.owner_name != "EE级"), :]
payevents.owner_name.unique() # 查看是否去除完成
payevents.to_csv('../tmp/payevents2.csv', na_rep='NaN', header=True, index=False)


# 代码10-3

# 处理订单数据（order_index）
order = pd.read_csv('../data/order_index.csv', encoding='gbk', 
                    header='infer', error_bad_lines=False)
# 过滤特殊线路、政企用户
order = order.ix[(order.owner_code != 2)&(order.owner_code != 9)&
                 (order.owner_code != 10), :] 
print('查看过滤后的特殊线路的用户:', order.owner_code.unique())
order = order.loc[(order.owner_name !='EA级')&(order.owner_name !='EB级')&
                  (order.owner_name !='EC级')&(order.owner_name !='ED级')&
                  (order.owner_name !='EE级'), :]
print('查看过滤后的政企用户:', order.owner_name.unique())

# 用户状态应只保留（正常，主动暂停，欠费暂停，主动销户）4个用户状态
order = order.loc[(order.business_name == '正常状态')|
        (order.business_name == '主动暂停')|
        (order.business_name == '欠费暂停状态')|
        (order.business_name == '主动销户'), :]
order.business_name.unique()
# 取optdate最大且effdate<=当前时间<=expdate的数据
order['optdate'] = pd.to_datetime(order.optdate)
order['effdate'] = pd.to_datetime(order.effdate)
order['expdate'] = pd.to_datetime(order.expdate)
import time
sj = time.ctime()
sj = pd.to_datetime(sj)
order1 = order.ix[(order['effdate']<=sj)&(order['expdate']>=sj), :]
order1 = order1.sort_values(by=['phone_no', 'optdate'])  # 以用户号及订单时间排序
# 根据字段phone_no, offername来去重
isduplicated = order1.duplicated(['phone_no', 'optdate', 'offername'], keep='last')
order2 = [order1.ix[i, ] for i in list(order1.index) if isduplicated[i] == False]
order2 = pd.DataFrame(order2)  # 转为数据框
order2.to_csv('../tmp/order2.csv', na_rep='NaN', header=True, index=False)



# 代码10-4

# 处理用户状态数据（mediamatch-userevents）
userevents = pd.read_csv('../data/mediamatch_userevents.csv', 
                         encoding='gbk', header='infer')
userevents.columns = ['phone_no', 'owner_name', 'run_name', 'run_time', 
                      'sm_name', 'owner_code']
# 过滤特殊线路、政企用户
userevents = userevents.ix[(userevents.owner_code != 2)&
                           (userevents.owner_code != 9)&
                           (userevents.owner_code != 10), :]
print('查看过滤后的特殊线路的用户:', userevents.owner_code.unique())
userevents = userevents.loc[(userevents.owner_name != 'EA级')&
                            (userevents.owner_name != 'EB级')&
                            (userevents.owner_name != 'EC级')&
                            (userevents.owner_name != 'ED级')
&(userevents.owner_name !='EE级'), :]
print('查看过滤后的政企用户:', userevents.owner_name.unique())

# 用户状态应只保留（正常，主动暂停，欠费暂停，主动销户）4个用户状态
userevents = userevents.loc[(userevents.run_name !='创建'), :]
userevents.run_name.unique()
userevents.to_csv('../tmp/userevents2.csv', na_rep='NaN', header=True, index=False)



# 代码10-5

import pandas as pd
import matplotlib.pyplot as plt
media3 = pd.read_csv('../tmp/media3.csv', header='infer')
# 用户观看总时长
m = pd.DataFrame(media3['wat_time'].groupby([media3['phone_no']]).sum())
m = m.sort_values(['wat_time'])
m = m.reset_index()
m['wat_time'] = m['wat_time'] / 3600
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.figure(figsize=(8, 4))
plt.bar(m.index,m.iloc[:,1])
plt.xlabel('观看用户（排序后）')
plt.ylabel('观看时长（单位：小时）')
plt.title('用户观看总时长')
plt.show()



# 代码10-6

import re
# 周观看时长分布
n = pd.DataFrame(media3['wat_time'].groupby([media3['星期']]).sum())
n = n.reset_index()
n = n.loc[[0, 2, 1, 5, 3, 4, 6], :]
n['wat_time'] = n['wat_time'] / 3600
plt.figure(figsize=(8, 4))
plt.plot(range(7), n.iloc[:, 1])
plt.xticks([0, 1, 2, 3, 4, 5, 6], 
           ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'])
plt.xlabel('星期')
plt.ylabel('观看时长（单位：小时）')
plt.title('周观看时长分布')
plt.show()

# 付费频道与点播回看的周观看时长分布
media_res = media3.ix[media3['res_type'] == 1, :]
ffpd_ind =[re.search('付费', str(i))!=None for i in media3.ix[:, 'station_name']]
media_ffpd = media3.ix[ffpd_ind, :]
z = pd.concat([media_res, media_ffpd], axis=0)
z = z['wat_time'].groupby(z['星期']).sum()
z = z.reset_index()
z = z.loc[[0, 2, 1, 5, 3, 4, 6], :]
z['wat_time'] = z['wat_time'] / 3600
plt.figure(figsize=(8, 4))
plt.plot(range(7), z.iloc[:, 1])
plt.xticks([0, 1, 2, 3, 4, 5, 6], 
           ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'])
plt.xlabel('星期')
plt.ylabel('观看时长（单位：小时）')
plt.title('付费频道与点播回看的周观看时长分布')
plt.show()



# 代码10-7

# 工作日与周末的观看时长比例
ind = [re.search('星期六|星期日', str(i)) != None for i in media3['星期']]
freeday = media3.ix[ind, :]
workday = media3.ix[[ind[i]==False for i in range(len(ind))], :]
m1 = pd.DataFrame(freeday['wat_time'].groupby([freeday['phone_no']]).sum())
m1 = m1.sort_values(['wat_time'])
m1 = m1.reset_index()
m1['wat_time'] = m1['wat_time'] / 3600
m2 = pd.DataFrame(workday['wat_time'].groupby([workday['phone_no']]).sum())
m2 = m2.sort_values(['wat_time'])
m2 = m2.reset_index()
m2['wat_time'] = m2['wat_time'] / 3600
w = sum(m2['wat_time']) / 5
f = sum(m1['wat_time']) / 2
plt.figure(figsize=(6, 6))
plt.pie([w, f], labels=['工作日', '周末'], explode=[0.1, 0.1], autopct='%1.1f%%')
plt.title('工作日与周末观看时长比例图')
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(121) # 参数为：行，列，第几项  subplot(numRows, numCols, plotNum)
plt.bar(m1.index, m1.iloc[:, 1])
plt.xlabel('观看用户（排序后）')
plt.ylabel('观看时长（单位：小时）')
plt.title('周末用户观看总时长')
plt.subplot(122)
plt.bar(m2.index, m2.iloc[:, 1])
plt.xlabel('观看用户（排序后）')
plt.ylabel('观看时长（单位：小时）')
plt.title('工作日用户观看总时长')
plt.show()



# 代码10-8

# 所有收视频道名称的观看时长与观看次数
media3.station_name.unique()
pindao = pd.DataFrame(media3['wat_time'].groupby([media3.station_name]).sum())
pindao = pindao.sort_values(['wat_time'])
pindao = pindao.reset_index()
pindao['wat_time'] = pindao['wat_time'] / 3600
pindao_n = media3['station_name'].value_counts()
pindao_n = pindao_n.reset_index()
pindao_n.columns=['station_name', 'counts']
a = pd.merge(pindao, pindao_n, left_on='station_name', right_on ='station_name', how='left')
fig, left_axis=plt.subplots()
right_axis = left_axis.twinx()
left_axis.bar(a.index, a.iloc[:, 1])
right_axis.plot(a.index, a.iloc[:, 2], 'r.-')
left_axis.set_ylabel('观看时长（单位：小时）')
right_axis.set_ylabel('观看次数')
left_axis.set_xlabel('频道号（排序后）')
plt.xticks([])
plt.title('所有收视频道名称的观看时长与观看次数')
plt.tight_layout()
plt.show()

# 收视前15频道名称的观看时长
plt.figure(figsize=(15, 8))
plt.bar(range(15), pindao.iloc[124:139, 1], width=0.5)
plt.xticks(range(15), pindao.iloc[124:139, 0])
plt.xlabel('频道名称')
plt.ylabel('观看时长（单位：小时）')
plt.title('收视前15频道名称的观看时长')
plt.show()