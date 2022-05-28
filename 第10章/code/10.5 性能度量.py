# 代码10-19

# 接着代码10-16
mat2 = media_test.pivot_table(index='phone_no', columns='program_title') # 透视表
mat2.columns = [i[1] for i in mat2.columns]
df_matrix2 = mat2
df_matrix2 = df_matrix2 / df_matrix2.sum(axis=0) * 5     # 0-5之间

# 计算推荐准确率
# 在df_matrix2中只要不为0则是观看过该节目
result1['real_program'] = 0
result1['T/F'] = 'NaN'
for i in range(df_matrix2.shape[0]):
    if df_matrix2.index[i] in list(result1['phone']):
        index = df_matrix2.iloc[i, :].notnull()
        index = index.reset_index()
        wp = index.loc[index.iloc[:, 1], 'index']
        wp1 = result1.loc[result1['phone'] == df_matrix2.index[i], :].program
        cunzai = wp1.isin(list(wp))
        result1.loc[result1['phone'] == df_matrix2.index[i], 'real_program'] = wp1[cunzai]
        result1.loc[result1['phone'] == df_matrix2.index[i], 'T/F'] = cunzai
    else:
        continue

precesion = sum(result1.loc[:, 'T/F'] == True) / len(result1)

# 计算召回率
program = pd.DataFrame()
for i in range(len(phone_test)):
    program1 = media.loc[media.phone_no == phone_test[i], 'program_title']
    program = pd.concat([program, program1])
# 推荐个数为5个，会影响召回率
recall = sum(result1.loc[:, 'T/F'] == True) / len(program.iloc[:, 0].unique())
print('协同过滤的准确率为：', precesion, '\n', '协同过滤的召回率为：', recall)



# 代码10-20

# 接着代码10-18
recommend_dataframe = recommend_dataframe
import numpy as np
phone_no = media1_test['phone_no'].unique()
real_dataframe = pd.DataFrame()
pre = pd.DataFrame(np.zeros((len(phone_no), 3)), columns=['phone_no', 'pre_num', 're_num'])
for i in range(len(phone_no)):
    real = media1_test.loc[media1_test['phone_no'] == phone_no[i], 'program_title']
    a = recommend_dataframe['program'].isin(real)
    pre.iloc[i, 0] = phone_no[i]
    pre.iloc[i, 1] = sum(a)
    pre.iloc[i, 2] = len(real)
    real_dataframe = pd.concat([real_dataframe, real])

real_program = np.unique(real_dataframe.iloc[:, 0])
# 计算推荐准确率
precesion = (sum(pre['pre_num'] / m)) / len(pre) # m为推荐个数，为3000

# 计算召回率
recall = (sum(pre['pre_num'] / pre['re_num'])) / len(pre)
print('流行度推荐的准确率为：', precesion, '\n', '流行度推荐的召回率为：', recall)