
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

dictt_flat_model={}
counter=0
for i in train.flat_model.unique():
    dictt_flat_model[i]=counter
    counter += 1
train['flat_model']=train['flat_model'].map(dictt_flat_model)

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
    dictt_flattype[i]=i


# In[8]:

# predicts on future data from 2013
dtrain=train[0:int(len(train)*.92)]
dtest=train[int(len(train)*.92):]
train=train[train['Time_sinceMRTbuilt'].isnull() == 0]


# In[9]:

from matplotlib import *
from matplotlib.colors import LogNorm
print predictors

def triple_plot(x_var,y_var,plt=plt,train=train,legend=True): #plots the median, and 25,75 percentiles for hist2d plot
    uniq=sorted(train[x_var].unique())
    uniq= uniq
    diff = (uniq[-1]-uniq[0])/len(uniq)
    while len(uniq) > 100:
        uniq= uniq[0::2]
        diff = (uniq[-1]-uniq[0])/len(uniq)
    plt.plot(uniq,map(lambda x : np.percentile(train[(train[x_var] > x-diff) & (train[x_var]< x+diff)][y_var],75),uniq),markersize=2,color='orange',alpha=0.99,linewidth=1,label='interquartile range')
    plt.plot(uniq,map(lambda x : np.percentile(train[(train[x_var] > x-diff) & (train[x_var]< x+diff)][y_var],25),uniq),markersize=2,color='orange',alpha=0.99,linewidth=1)
    plt.plot(uniq,map(lambda x : np.median(train[(train[x_var] > x-diff) & (train[x_var]< x+diff)][y_var]),uniq),markersize=2,color='red',label='median',alpha=0.99,linewidth=1)
    if legend:
        plt.legend()




# In[10]:

''' getting the price per square meter, normalized by mean of houses sold that month. Gives a guage of how expensive the house is relative to other places in singapore
'''
train['price_sqm']=train[target]/(train['month_mean']*train['floor_area_sqm']) #normalized by monthly mean
train['lease_length']=train.month - train.lease_commence_date


# In[11]:

'''
Finding variables most correlated with normalized price per sqm (price_sqm)
'''
train.corr()['price_sqm']


# In[12]:

plt.hist2d(train['lease_length'],train['price_sqm'],200,norm=LogNorm())
plt.colorbar()
plt.xlabel('lease length (years)')
plt.ylabel('normalized price per sqm')
triple_plot('lease_length','price_sqm')
plt.tight_layout()

plt.savefig('leaseLength_and_price')
##plt.show()
plt.clf()

# In[15]:

plt.hist2d(train['floor_area_sqm'],train['norresale_price'],200,norm=LogNorm())
triple_plot('floor_area_sqm','norresale_price')
plt.colorbar()
plt.xlabel('floor area (sq meter)')
plt.ylabel('normalized price per sqm')
plt.savefig('floor_sq_area')
#plt.show()
plt.clf()
train[['floor_area_sqm','storey_range','lease_commence_date','month',target]].corr()


# In[16]:

plt.hist2d(train['month'],train['resale_price'],200,norm=LogNorm())
triple_plot('month','resale_price')
plt.colorbar()
plt.xlabel('year')
plt.ylabel('resale price (SG dollars)')
plt.tight_layout()
plt.savefig('price_over_years')
#plt.show()
plt.clf()
train[['floor_area_sqm','storey_range','lease_commence_date','month',target]].corr()


# In[17]:
a=train[train['lease_length']< 5]
plt.hist(a['month'],bins=100)
plt.tight_layout()
plt.xlabel('year')
plt.ylabel('number of houses sold before MOP')
plt.tight_layout()
plt.savefig('b4 five year')



plt.hist2d(train['dist_nearestMRT'],train['norresale_price'],200,norm=LogNorm())
triple_plot('dist_nearestMRT','norresale_price')
plt.xlabel('distance to nearest MRT (km)')
plt.ylabel('normalized price per sqm')
plt.colorbar()
plt.tight_layout()
plt.savefig('dist_mrt')
#plt.show()
plt.clf()
train[['dist_nearestMRT',target, 'norresale_price']].corr()


# In[18]:

from matplotlib import *
from matplotlib.colors import LogNorm

plt.hist2d(train['Time_sinceMRTbuilt'],train['nor'+target],100,norm=LogNorm())
plt.colorbar()
#plt.hist2d(train[train['Time_sinceMRTbuilt'] >0 ]['Time_sinceMRTbuilt'] ,train[train['Time_sinceMRTbuilt'] >0 ]['price_sqm'],200,norm=LogNorm())
plt.xlabel('year since nearest MRT was built')
plt.ylabel('normalized resale price \n(relative to mean of particular month)')
triple_plot('Time_sinceMRTbuilt','norresale_price')
plt.tight_layout()
plt.savefig('Time_since_mrtBuilt')
#plt.show()
plt.clf()
train[['nor'+target,'Time_sinceMRTbuilt']].corr() #negative correlation, built longer = lower price


# In[19]:

train['price_sqm']=train[target]/(train['month_mean']*train['floor_area_sqm']) #normalized by monthly mean
plt.hist2d(train['Time_sinceMRTbuilt'],train['price_sqm'],200,norm=LogNorm())
plt.xlabel('year since nearest MRT was built')
plt.ylabel('resale price per sqm \n(relative to mean of particular month)')
triple_plot('Time_sinceMRTbuilt','price_sqm')
plt.tight_layout()
plt.savefig('Time_since_mrtBuilt2')
plt.colorbar()
#plt.show()
plt.clf()
train[['Time_sinceMRTbuilt','price_sqm']].corr() # negative one percent correlation


# In[22]:


train.town.unique()
fig, ax1 = plt.subplots(nrows=5, ncols=6,figsize=(25, 20))

for i in range(0,27):
    if i==0:
        a=ax1[i//6,i%6].hist2d(train[train['town']==i]['lease_length'],train[train['town']==i]['price_sqm'],100,range=((0,50),(0,0.04)),\
                               norm=LogNorm(),normed=True)
        aa=ax1[-1,-1].imshow([[0,0],[0,0]],norm=LogNorm(),vmin=1,vmax=50)
        cb=fig.colorbar(aa,ax=ax1[0,0])
        cb.set_ticks(range(1,10)+[10*x for x in range(1,6)])
        cb.set_ticklabels([1, '', 3, '', '','' ,'' , '','' , 10, '', '','' , 50])
        triple_plot('lease_length','price_sqm',plt=ax1[i//6,i%6],train=train[train['town']==i],legend=True)
        ax1[i//6,i%6].set_xticks([0,10,20,30,40,50])
    else:
        a=ax1[i//6,i%6].hist2d(train[train['town']==i]['lease_length'],train[train['town']==i]['price_sqm'],100,range=((0,50),(0,0.04)),\
                               norm=LogNorm(),normed=True)#,norm=LogNorm())
        triple_plot('lease_length','price_sqm',plt=ax1[i//6,i%6],train=train[train['town']==i],legend=False)
    ax1[i//6,i%6].set_xlabel('lease length ')
    ax1[i//6,i%6].set_title(str([x for x in dictt_town if dictt_town[x]==i][0])+', '+str(sum(train['town']==i))+' flats')
    ax1[i//6,i%6].set_ylabel('normalized price per sqm')
    #print i,i//6,i%6
fig.tight_layout()
plt.savefig('town')
#plt.show()
plt.clf()
    


# In[14]:

''' getting the price per square meter, normalized by mean of houses sold that month. Gives a guage of how expensive the house is relative to other places in singapore
'''
fig, ax1 = plt.subplots(figsize=(10, 10))
fig.tight_layout(pad=4.0, w_pad=4, h_pad=4)
temp =[]
for i in range(0,int(train['storey_range'].max()),5):
    temp += [train[(train['storey_range'] > i) & (train['storey_range'] < 5+i)]['price_sqm']]
bp=plt.boxplot(temp,notch=0, sym='+', vert=1, whis=1.5)
x_axis=0
for i in range(0,int(train['storey_range'].max()),5):
    x = np.random.normal(x_axis+1, 0.02, size=len(temp[x_axis]))
    plt.plot(x,temp[x_axis] , 'r.', alpha=0.002)
    x_axis += 1
plt.xlabel('level')
plt.ylabel('normalized price per sqm')
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
xtickNames = plt.setp(ax1, xticklabels=range(0,int(train['storey_range'].max()),5))
plt.tight_layout()
plt.savefig('story')
#plt.show()
plt.clf()



class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin,  self.vmax], [0,  1]
        return np.ma.masked_array(np.interp(value, x, y))
def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    #print cdict
    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap
