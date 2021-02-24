# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = datasets.load_diabetes()
X=data.data
y=data.target

X /= X.std(axis=0)

trainX, validX, trainy, validy = train_test_split(X,y, test_size=0.2, shuffle=True, random_state=30)

reg1 = LinearRegression()
reg2=Ridge(alpha=1)
reg3=Lasso(alpha=1)

reg1.fit(trainX,trainy)
reg1.coef_
reg2.fit(trainX,trainy)
reg2.coef_
reg3.fit(trainX,trainy)
reg3.coef_

alphas=np.logspace(-3,3,30)  #30개 생성

linear_r2=reg1.score(validX,validy)
result=pd.DataFrame(index=alphas, columns=['Ridge', 'Lasso'])
for alpha in alphas:
    reg2.alpha=alpha
    reg3.alpha=alpha
    reg2.fit(trainX,trainy)
    result.loc[alpha, 'Ridge'] = reg2.score(validX,validy)
    reg3.fit(trainX,trainy)
    result.loc[alpha,'Lasso']=reg3.score(validX,validy)
    
plt.plot(np.log(alphas), result['Ridge'], label="Ridge")
plt.plot(np.log(alphas), result['Lasso'], label="Lasso")
plt.hlines(linear_r2, np.log(alphas[0]), np.log(alphas[-1]), ls=':', 
          color="k", label='Ordinary')
plt.legend()

X,y=datasets.make_regression(n_samples=1000, n_features=10, n_informative=10)

X.mean(axis=0)
X.std(axis=0)

reg1.fit(X,y)
reg1.coef_

reg2.alpha=10
reg2.fit(X,y)
reg2.coef_

beta1=reg2.coef_

X2=X.copy()
X2[:,0]/=10

reg1.fit(X2,y)
reg1.coef_

reg2.fit(X2,y)
beta2=reg2.coef_
beta1
beta2

beta1/beta2 #index 0 ration is so low, by scaling...

from sklearn.linear_model import lasso_path, enet_path

data = datasets.load_diabetes()
X=data.data
y=data.target

X /= X.std(axis=0)

eps=5e-3
alphas_lasso, coefs_lasso, _ = lasso_path(X,y,eps,fit_intercept=False)

alphas_enet, coefs_enet, _ = enet_path(X,y,eps=eps,l1_ratio=0.5, fit_intercept=False)
from itertools import cycle

colors=cycle(plt.cm.tab10(np.arange(10)/10))

neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_enet = -np.log10(alphas_enet)

for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
    l1=plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    l2=plt.plot(neg_log_alphas_enet, coef_e, c=c, ls="--")
    
plt.legend((l1[-1],l2[-1]),('Lasso', 'Elastic Net'))
plt.axis('tight')

##lesson2

import pandas as pd

train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', dtype={'StateHoliday':'str'})
store=pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')

train=train.merge(store, on=['Store'])

sales=train[['Store','Date','Sales','Open']]

sales_count=sales.groupby('Store')['Date'].count()
sales=sales[sales['Store'].isin(sales_count[sales_count==sales_count.max()].index)]
sales['Date']=pd.to_datetime(sales['Date']) #change datatype

daily_sales=sales[sales['Open']==1].groupby(['Date'])['Sales'].mean()

plt.plot(daily_sales)

s=pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
s=s.cumsum()
plt.plot(s)

r=s.rolling(window=60)

s.plot(style='k--')
r.mean().plot(style="r")

fig=plt.figure(figsize=(10,8))
plt.plot(daily_sales, ':k')
plt.plot(daily_sales.rolling(window=30).mean(), 'r')

train['Date']=pd.to_datetime(train['Date'])
train['Year']=train['Date'].dt.year
train['Month']=train['Date'].dt.month

sel_store=sales_count[sales_count==sales_count.max()].index
sel_train=train[train['Store'].isin(sel_store)]
sel_train=sel_train.sort_values(['Store','Date'])

## add categorical values
catvar=['DayOfWeek','Month','StoreType', 'Assortment','StateHoliday']
for c in catvar:
    temp=pd.get_dummies(sel_train[c],prefix=c,drop_first=True)
    sel_train=pd.concat((sel_train,temp), axis=1)
    
sel_train=sel_train.drop(catvar,axis=1)
sel_train=sel_train[sel_train['Open']==1]

trainX=sel_train[sel_train['Date']<=pd.to_datetime('20141231')]
valX=sel_train[sel_train['Date']>pd.to_datetime('20141231')]

remove_cols=['Store','Date','Customers','Open','Sales','Year','CompetitionDistance',
             'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
             'Promo2SinceWeek','Promo2SinceYear','PromoInterval']

trainY=trainX['Sales']
trainX=trainX.drop(remove_cols,axis=1)

valY=valX['Sales']
valX=valX.drop(remove_cols,axis=1)

reg1=LinearRegression()
reg1.fit(trainX,trainY)
reg1.score(trainX,trainY)
reg1.score(valX,valY)

sel_store_history=sel_train[sel_train['Store'].isin(np.random.choice(sel_store,10))]
sel_store_history=pd.pivot_table(sel_store_history, index="Date", columns="Store", values="Sales")
sel_store_history=sel_store_history.fillna(0)

for c in sel_store_history.columns:
    plt.plot(sel_store_history[c].rolling(window=30).mean())

sel_train=sel_train.set_index('Date')

new_variables=sel_train.groupby('Store')['Sales'].rolling(window='7D').mean()
new_variables=new_variables.to_frame().rename(columns={'Sales':'Sales1W'})
new_variables['Sales2W']=sel_train.groupby('Store')['Sales'].rolling(window='14D').mean()
new_variables['Sales1_2_diff']=new_variables['Sales1W']-new_variables['Sales2W']
new_variables['Sales1_2_ratio']=new_variables['Sales1W']/new_variables['Sales2W']

new_variables.head(30)

new_variables=new_variables.reset_index()
new_variables['Date'] = new_variables['Date']+pd.to_timedelta('7D')

new_variables.head()  

new_sel_train=sel_train.merge(new_variables, on=['Store','Date'], how="left")
new_sel_train=new_sel_train[new_sel_train['Date']>=pd.to_datetime('2013-0115')]  
new_sel_train=new_sel_train.fillna(0)

trainX2=new_sel_train[new_sel_train['Date']<=pd.to_datetime('20141231')]
valX2=new_sel_train[new_sel_train['Date']>pd.to_datetime('20141231')]

remove_cols=['Store','Date','Open','Customers','Sales','Year','CompetitionDistance',
             'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
             'Promo2SinceWeek','Promo2SinceYear','PromoInterval']

trainY2=trainX2['Sales']
trainX2=trainX2.drop(remove_cols,axis=1)

valY2=valX2['Sales']
valX2=valX2.drop(remove_cols,axis=1)

reg1=LinearRegression()
reg1.fit(trainX2,trainY2)
reg1.score(valX2,valY2)  #by add 4 input variables, the score is higher than previous model
