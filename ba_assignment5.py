# -*- coding: utf-8 -*-
"""
find best lamda
@author: smsun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', dtype={'StateHoliday':'str'})
store=pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')
train = train.merge(store, how="left") #merge


train['Date_2'] = pd.to_datetime(train['Date'])

##transform promoInterval data by categorical data
train['Year'] = train['Date_2'].dt.year
train['Month'] = train['Date_2'].dt.month


train=train.fillna(0)

train.loc[train["PromoInterval"]=="Jan,Apr,Jul,Oct"] = 1
train.loc[train["PromoInterval"]=="Feb,May,Aug,Nov"] = 2
train.loc[train["PromoInterval"]=="Jan,Apr,Jul,Oct"] = 3

#check data violdate
plt.hist(sales_count.values)


##drop volidate date
sales_count = train.groupby('Store')['Date_2'].count()
train=train[train['Store'].isin(sales_count[sales_count==sales_count.max()].index)]

#a = train[train['PromoInterval']=="Jan,Apr,Jul,Oct"] 
#b = train[train['PromoInterval']=="Feb,May,Aug,Nov"]
#c = train[train['PromoInterval']=="Mar,Jun,Sept,Dec"]
#d = train[train['PromoInterval']==0]
#a['PromoInterval'] = 1
#b['PromoInterval'] = 2
#c['PromoInterval'] = 3

#train_data = a.append(b).append(c).append(d)

##varlist
varlist = ['Promo2', 'PromoInterval','CompetitionDistance',
           'Month','Year','Customers','Date_2','Date','Sales'] #PromoInterval

train_data = train[varlist]

##add categorical value
catvar=['Promo2','PromoInterval','Month','Year']
for c in catvar:
    temp=pd.get_dummies(train[c],prefix=c,drop_first=True)
    train_data=pd.concat((train_data,temp), axis=1)
    

##split data
train_data['Date'] = pd.to_datetime(train['Date'])

train_X = train_data[train_data['Date']<=pd.to_datetime('20141231')] #split data
test_X = train_data[train_data['Date']>pd.to_datetime('20141231')]

train_y = train_X['Sales'] 
test_y = test_X['Sales'] 

train_X = train_X.drop(['Date','Sales','Date_2'], axis=1)
test_X = test_X.drop(['Date','Sales','Date_2'],axis=1)



##linearRegression ridge Rasso
from sklearn.linear_model import LinearRegression, Ridge, Lasso
reg1 = LinearRegression()
reg2 = Ridge(alpha=1)
reg3 = Lasso(alpha=1)

reg1.fit(train_X,train_y)
reg1.score(test_X, test_y)

reg2.fit(train_X,train_y)
reg2.score(test_X,test_y)

reg3.fit(train_X, train_y)
reg3.score(test_X, test_y)


#logspace이용하여 알파 파라미터 찾기
alphas = np.logspace(-3,3,30)
result=pd.DataFrame(index=alphas, columns=['Ridge', 'Lasso'])
for alpha in alphas:
    reg2.alpha=alpha
    reg3.alpha=alpha
    reg2.fit(train_X,train_y)
    result.loc[alpha, 'Ridge'] = reg2.score(test_X,test_y)
    reg3.fit(train_X,train_y)
    result.loc[alpha,'Lasso']=reg3.score(test_X,test_y)

param_Ridge = 0.78804
param_Lasso = 0.001
    
##test 5-fold cross validation
from sklearn.model_selection import KFold

kf=KFold(n_splits=5, shuffle=True, random_state=1)
for train,test in kf.split(train_data):
    print(train,test)
    
    
train_X2 = train_data.iloc[train]
test_X2 = train_data.iloc[test]
train_y2 = train_data.iloc[train]['Sales']
test_y2 = train_data.iloc[test]['Sales']
train_X2=train_X2.drop(['Date','Sales','Date_2'], axis=1)
test_X2=test_X2.drop(['Date','Sales','Date_2'], axis=1)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
reg_1 = LinearRegression()
reg_2 = Ridge(alpha=param_Ridge)
reg_3 = Lasso(alpha=param_Lasso)

reg_1.fit(train_X2,train_y2)
reg_1.score(test_X2, test_y2)

reg_2.fit(train_X2,train_y2)
reg_2.score(test_X2,test_y2)

reg_3.fit(train_X2, train_y2)
reg_3.score(test_X2, test_y2)

