
# coding: utf-8

# In[1]:

import pandas as pd
import os,sys
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import urllib
import sqlite3
import json
import time
import ssl
import pandas as pd
import numpy as np
import re as re
from matplotlib import *
from pylab import *
rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams.update({'font.size': 12})
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'semibold'
# In[2]:

'''
This file does analysis of data
'''

train=pd.read_csv('train_mrt.csv.gz', header=0)
train=train.sort_values(by='month').reset_index()
target='resale_price'
predictors=[x for x in  list(train.keys()) if x!=target]
train['sql_name']=train['street_name']+' '+train['block']+'  SINGAPORE'


# In[3]:

'''

'''
#set up missing values with mean~0.8km
x=train[train['dist_nearestMRT'].isnull()]['dist_nearestMRT'].index
train.set_value(x,'dist_nearestMRT',0.8)
train.set_value(x,'nearestMRT','REDHILL MRT')

# mrt station built date file
a=pd.read_csv('mrt_date.csv')
print a.head() # i know the formatting has problem, but watever


# In[4]:


stn_names=map(lambda x : str(x) , a['Station name ']) 
stn_times=map(lambda x : str(x).split() , a['Abbreviation']) # Abbreviation gives dates
dictt_temp = {'February': 2, 'October': 10, 'March': 3, 'May': 5,
              'December': 12, 'June': 6, 'April': 4, 'January': 1,
              'November': 11, 'July': 7}

dict_startMRT = {}
for i in range(len(stn_names)):
    if len(stn_times[i]) >= 3:
        stn_times[i][1]=dictt_temp[stn_times[i][1]]
        dict_startMRT[stn_names[i].upper()+'MRT']= round(int(stn_times[i][2])                                                +1.0*stn_times[i][1]/12,2)

# get mrt built date from nearest mrt
train['MRTbuilt'] = train['nearestMRT'].map(dict_startMRT)



# In[5]:

# get unique values for each predictors
target='resale_price'
predictors=[x for x in  list(train.keys()) if x!=target]
dictt={}
for i in predictors :
    dictt[i]=list(train[i].unique())


# In[6]:

'''
Next few lines change string catagorical values to integers, since xgboost cannot handle string
'''
### Changing month to float
dictt_month = {}
for i in dictt['month']:
    dictt_month[i]=round(int(i[0:4])+float(i[-2:])*1.0/12,1)
train['month']=train['month'].map(dictt_month).astype(np.float32)
train['Time_sinceMRTbuilt']=train['month']-train['MRTbuilt']
#train.set_value(train[train['Time_sinceMRTbuilt']<0].index,'Time_sinceMRTbuilt',-10)

#### nearest mrt_station to integer
dictt_nearestMRT={}
counter=0
for i in dictt['nearestMRT']:
    dictt_nearestMRT[i]=counter
    counter +=1
train['nearestMRT']=train['nearestMRT'].map(dictt_nearestMRT)


### Changing flat_type
dictt_flattype={}

dictt_flattype={'3 ROOM': 3, '4 ROOM': 4, '1 ROOM': 1,
                'MULTI GENERATION': 7, 'MULTI-GENERATION':
                8, 'EXECUTIVE': 6,
                '5 ROOM': 5, '2 ROOM': 2}
train['flat_type']=train['flat_type'].map(dictt_flattype).astype(np.int16)
train['storey_range']=train['storey_range'].map(lambda x:0.5*int(x[0:2])+0.5*int(x[-2:]))


dictt_street_name={}
counter=0
for i in sorted(train.street_name.unique()):
    dictt_street_name[i]=counter
    counter += 1
train['street_name']=train['street_name'].map(dictt_street_name)

dictt_block={}
import string
for i in sorted(train.block.unique()):
    dictt_block[i]=int(i.translate(None,string.letters)) #keep digits only
train['block']=train['block'].map(dictt_block)

# In[7]:

'''
This does three things
1. adds monthly mean of prices as an additional explanatory variable. This models in external factors like economic growth, inflation and others into the model implicitly, and this number can be estimated with the help of a professional real estate agent for future modelling
2. adds normalized target variable compared to montly mean. To model the relative prices to other houses. This is a rough measure, can use stratified mean instaed
'''
### Sorting towns by prices , then giving them number
dictt_town = {}
for i in sorted(train['month'].unique()):
	dictt_month[i]= np.median(train[train['month']==i][target]) #median isntad of mean
train['month_mean']=train['month'].map(dictt_month).astype(np.int64)
train['nor'+target]=1.0*train[target]/train['month_mean']


# In[10]:

''' getting the price per square meter, normalized by mean of houses sold that month. Gives a guage of how expensive the house is relative to other places in singapore
'''

train['price_sqm_unNorm']=train[target]/(train['month_mean']*train['floor_area_sqm']) #normalized by monthly mean
train['price_sqm']=0
for x in train.month.unique(): #normalized to one using median of month as base
    idx = train[train['month']==x].index
    temp=train['price_sqm_unNorm'][train['month']==x]/np.median(train[train['month']==x]['price_sqm_unNorm'])
    temp=train.set_value(idx,'price_sqm',temp)  
train['lease_length']=train.month - train.lease_commence_date

x=[]
for i in list(train['town'].unique()):
    x+= [[ i,np.mean(train[train['town']==i]['nor'+target])],]
counter=0
for i in sorted(x,key=lambda x:x[0]):
    dictt_town[i[0]]=counter
    counter+=1
train['town']=train['town'].map(dictt_town).astype(np.int32)
predictors+=['month_mean', 'nor'+target]
for i in sorted(train['flat_type'].unique()):
    print i, np.mean(train[train['flat_type']==i]['nor'+target])
    
dictt_flat_model={}
counter=0

xxx=[]
for i in list(train['flat_model'].unique()):
    xxx+= [[ i,np.mean(train[train['flat_model']==i]['nor'+target])],]
counter=0
for i in sorted(xxx,key=lambda x:x[1]):
    dictt_flat_model[i[0]]=counter
    counter+=1
train['flat_model']=train['flat_model'].map(dictt_flat_model)
for i in sorted(train['flat_model'].unique()):
    print i, np.mean(train[train['flat_model']==i]['nor'+target])
# In[8]:

# predicts on future data from 2013
predictors = [ 'town', 'flat_type',  'storey_range', \
              'floor_area_sqm','lease_commence_date', 'lat', 'lng', \
              'nearestMRT', 'dist_nearestMRT', 'MRTbuilt', 'Time_sinceMRTbuilt', \
              'month_mean', 'lease_length']
target='resale_price'
train[target]=np.log10(train[target])
dtrain=train[0:int(len(train)*.90)]
dtest=train[int(len(train)*.90):]
train=train[train['Time_sinceMRTbuilt'].isnull() == 0]
xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values,missing=np.NAN,feature_names=predictors)
xgtest = xgb.DMatrix(dtest[predictors].values, label=dtest[target].values,missing=np.NAN,feature_names=predictors)
 
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.05
params["min_child_weight"] = 1
params["subsample"] = 0.7
params["colsample_bytree"] = 0.7
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 12
#params["nthread"] = 6
params["nthread"] = 6
#params["reg_alpha"] = 3
#params["reg_lamb'lda"] = 3
plst = list(params.items())
watchlist  = [ (xgtrain,'train'),(xgtest,'CV')]
a={}
 
model=xgb.train(plst,xgtrain,500,watchlist,early_stopping_rounds=30,evals_result=a,maximize=0)
z= model.get_fscore()
print sorted([ [z[i],i] for i in z ],reverse=1)[0:5]
dtest['predict']=10**(model.predict(xgtest))
dtest['mae']=dtest['predict']- 10**(dtest[target])
dtrain['predict']=10**(model.predict(xgtrain))
dtrain['mae']=dtrain['predict']- 10**(dtrain[target])
print np.mean((dtrain['mae']**2)**.5);print np.mean((dtest['mae']**2)**.5)


dtest1=dtest[dtest['month']>=2017.09997]
dtest1['month_mean']=300000
dtest2=dtest[dtest['month']>=2017.09997]
dtest2['month_mean']=400000
dtest3=dtest[dtest['month']>=2017.09997]
dtest3['month_mean']=500000
dextra=pd.concat([dtest1,dtest2,dtest3])
xgextra = xgb.DMatrix(dextra[predictors].values, label=dextra[target].values,missing=np.NAN,feature_names=predictors)
dextra['predict']=10**model.predict(xgextra)
def main():
    fig, ax1 = plt.subplots(figsize=(5, 5))
    fig.tight_layout(pad=4.0, w_pad=4, h_pad=4)
    temp =[]
    for i in sorted(dextra['month_mean'].unique()):
        temp += [dextra[dextra['month_mean']==i]['predict']]
    temp += [10**(dextra[dextra['month_mean']==i][target])]
    bp=plt.boxplot(temp,notch=0, sym='+', vert=1, whis=1.5)
    x_axis=0
    for i in sorted(dextra['month_mean'].unique()):
        x = np.random.normal(x_axis+1, 0.02, size=len(temp[x_axis]))
        plt.plot(x,temp[x_axis] , 'r.', alpha=0.2)
        x = plt.hist(temp[x_axis],bins=20,normed=1)#np.random.normal(x_axis+1, 0.02, size=len(temp[x_axis]))
        offset=x[1][1]-x[1][0]
        plt.plot(1+x_axis+0.3*x[0]/max(x[0]),x[1][:-1]+0.5*offset , 'b',alpha=0.5)
        temp1,temp2=[],[]
        for i in 0.3*x[0]/max(x[0]):
            temp1 += [0,i,i,0]
        for i in x[1][1:-1]:
            temp2 += [i,i,i,i]
        temp2 = [x[1][0],]*2 + temp2 + [x[1][-1],]*2
        plt.plot(np.array(temp1+[0,0])+1+x_axis,temp2+[max(temp2),min(temp2)],'b',alpha=0.0)
        x_axis += 1
    x = np.random.normal(x_axis+1, 0.02, size=len(temp[x_axis]))
    plt.plot(x,temp[x_axis] , 'r.', alpha=0.2)
    x = plt.hist(temp[x_axis],bins=20,normed=1)#np.random.normal(x_axis+1, 0.02, size=len(temp[x_axis]))
    offset=x[1][1]-x[1][0]
    plt.plot(1+x_axis+0.3*x[0]/max(x[0]),x[1][:-1]+0.5*offset , 'b',alpha=0.5)
    temp1,temp2=[],[]
    for i in 0.3*x[0]/max(x[0]):
        temp1 += [0,i,i,0]
    for i in x[1][1:-1]:
        temp2 += [i,i,i,i]
    temp2 = [x[1][0],]*2 + temp2 + [x[1][-1],]*2
    plt.plot(np.array(temp1+[0,0])+1+x_axis,temp2+[max(temp2),min(temp2)],'b',alpha=0.0)
    plt.xlabel('Property Market Outlook')
    plt.ylabel('Price in dollars')
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    xtickNames = plt.setp(ax1, xticklabels=['poor\n(predicted)','current\n(predicted)','good\n(predicted)','current\n(actual)'])
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('forcast',dpi=200)
#plt.show()
main()
