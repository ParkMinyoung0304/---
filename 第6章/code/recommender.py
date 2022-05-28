import numpy as np
import pandas as pd
import math
def prediction(df,userdf,Nn=15):#Nn邻居个数
    corr=df.corr();
    rats=userdf.copy()
    for itemid in userdf.columns:
        dfnull=userdf.loc[:,itemid][userdf.loc[:,itemid].isnull()]
        itemv=df.loc[:,itemid].mean()#评价平均值
        for usrid in dfnull.index:
            nft=(df.loc[usrid]).notnull()
            #获取该用户评过分的物品中最相似邻居物品列表，且取其中Nn个，如果不够Nn个则全取
            nlist=df.loc[usrid][nft]
            nlist=nlist[(corr.loc[itemid,nlist.index][corr.loc[itemid,nlist.index].notnull()].sort_values(ascending=False)).index]
            if(Nn<=len(nlist)):
                nlist=(df.loc[usrid][nft])[:Nn]
            else:
                nlist=df.loc[usrid][nft][:len(nlist)]
            nratsum=0
            corsum=0
            if(0!=nlist.size):
                nv=df.loc[:,nlist.index].mean()#邻居评价平均值
                for index in nlist.index:
                    ncor=corr.loc[itemid,index]
                    nratsum+=ncor*(df.loc[usrid][index]-nv[index])
                    corsum+=abs(ncor)
                if(corsum!=0):
                    rats.at[usrid,itemid]= itemv + nratsum/corsum
                else:
                    rats.at[usrid,itemid]= itemv
            else:
                rats.at[usrid,itemid]= None
    return rats
def recomm(df,userdf,Nn=15,TopN=3):
    ratings=prediction(df,userdf,Nn)#获取预测评分
    recomm=[]#存放推荐结果
    for usrid in userdf.index:
        #获取按NA值获取未评分项
        ratft=userdf.loc[usrid].isnull()
        ratnull=ratings.loc[usrid][ratft]
        #对预测评分进行排序
        if(len(ratnull)>=TopN):
            sortlist=(ratnull.sort_values(ascending=False)).index[:TopN]
        else:
            sortlist=ratnull.sort_values(ascending=False).index[:len(ratnull)]
        recomm.append(sortlist)
    return ratings,recomm


