import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('I:/电力分项计量案例/基于非侵入式负荷监测与分解的电力数据挖掘（胡杰）')
filename = os.listdir("I:/电力分项计量案例/基于非侵入式负荷监测与分解的电力数据挖掘（胡杰）/data/附件1")  # 得到文件夹下的所有文件名称
n_filename = len(filename)
# 给设备数据添加操作信息，画出各特征轨迹图并保存
def fun(a):
    save_name = ['YD1','YD10','YD11','YD2','YD3','YD4',
           'YD5','YD6','YD7','YD8','YD9']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号   
    for i in range(a):
        Sb = pd.read_excel("data/附件1/" + filename[i],'设备数据',index_col = None)
        Cz = pd.read_excel("data/附件1/" + filename[i],'操作记录',index_col = 0)
        Xb = pd.read_excel("data/附件1/" + filename[i],'谐波数据',index_col = None)
        Zb = pd.read_excel("data/附件1/" + filename[i],'周波数据',index_col = None)
        # 电流轨迹图
        plt.plot(Sb['IC'])
        plt.title(save_name[i]+'-IC')
        plt.show()
        # 电压轨迹图
        plt.plot(Sb['UC'])
        plt.title(save_name[i] + '-UC')
        plt.show()
        # 有功功率和总有功功率
        plt.plot(Sb[['PC','P']])
        plt.title(save_name[i] + '-P')
        plt.show()
        # 无功功率和总无功功率
        plt.plot(Sb[['QC','Q']])
        plt.title(save_name[i] + '-Q')
        plt.show()
        # 功率因数和总功率因数
        plt.plot(Sb[['PFC','PF']])
        plt.title(save_name[i] + '-PF')
        plt.show()
        # 谐波电压
        plt.plot(Xb.loc[:,'UC02':].T)
        plt.title(save_name[i] + '-谐波电压')
        plt.show()
        # 周波数据
        plt.plot(Zb.loc[:,'IC001':].T)
        plt.title(save_name[i] + '-周波数据')
        plt.show()

fun(n_filename)
