# -*- coding: utf-8 -*-
'''
assignment4

Rossmann data

purpose : predicting sales after one week

split data
1.training set : 2013,2014

2.validation set : 2015

to do
1. select explanatory variables to predict sales
 -> list variables used in learning and explain why
 
2. select learning methods to predict sales
 -> at least two    
 -> compare the trained models using validtion set
 
3. summarize procedures and results
 -> describe preprocessing steps
 -> explain the results from different algorithms
'''

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import datetime

train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', dtype={'StateHoliday':'str'})
store=pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')

train = train.fillna(0) #transform Nan as 0
store = store.fillna(0)
#index_2015 = 0

#for i in range(0, len(train)):
#    if train['Date'][i][0:4] == '2015':
#        index_2015 = i

date_transform = []
for i in train['Date']:
    time_data = i.split('-')
    time_index = datetime.datetime(int(time_data[0]),int(time_data[1]),int(time_data[2]))- datetime.datetime(2013,1,1)
    date_transform.append(time_index.days%365)

train['date_transform'] = date_transform

##split data 
df1 = train[0:236380] 
df2 = train[236380:len(train)]

##data shuffle and regression


##regression 
merge_data = pd.merge(train,store, how="left")
train_data = pd.merge(df2, store, how="left")
test_data = pd.merge(df1, store, how="left")



varlist = ['Customers','Open', 'Promo',
           'CompetitionDistance','Promo2','date_transform']


##OLS result
train_X = train_data[varlist]
train_y = train_data['Sales']

test_X = test_data[varlist]
test_y = test_data['Sales']

train_X1 = sm.add_constant(train_X)

reg1 = sm.OLS(train_y, train_X1)
result1 = reg1.fit()
result1.summary()

##check train and test

from sklearn.linear_model import LinearRegression

reg2 = LinearRegression()
reg2.fit(train_X,train_y)
reg2.score(train_X,train_y)
reg2.score(test_X, test_y)
reg2.predict(test_X)

## cross validation
train_X_cross, test_X_cross, train_y_cross, test_y_cross = train_test_split(
    merge_data[varlist],merge_data['Sales'],
    test_size=0.2, random_state=100)

reg3 = LinearRegression()
reg3.fit(train_X, train_y)
reg3.score(test_X, test_y)

##predict

#index_2013 = 0
#index_2014 = 0
#index_2015 = 236380

#for i in range(0, len(merge_data)):
#    if merge_data['Date'][i] == '2014-08-01':
#        index_2014 = i
#        break

#for i in range(0, len(merge_data)):
#    if merge_data['Date'][i] == '2013-08-01':
#        index_2013 = i
#        break

#data_2013 = merge_data[index_2013:len(merge_data)][varlist]
#data_2014 = merge_data[index_2014:index_2013][varlist]
#data_2015 = merge_data[0:index_2014]     

merge_data_group = merge_data.groupby('Date')[varlist].mean()
predict_input = merge_data_group.groupby("date_transform")[varlist].mean()[212:365]
                        
learn_X = merge_data[varlist]
learn_y = merge_data['Sales']
reg4 = LinearRegression()
reg4.fit(learn_X, learn_y)
predict_y = reg4.predict(predict_input)


##autoregressive model
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR #use autoagressive model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


time_sales_data = pd.DataFrame(train_data.groupby("Date").mean()['Sales'],
                               columns=['Sales'])
time_sales_2013 = time_sales_data[0:365]
time_sales_2014 = time_sales_data[365:730]

test_data = pd.DataFrame(test_data.groupby("Date").mean()['Sales'],
                         columns=['Sales'])


plt.plot(time_sales_2013.index, time_sales_2013['Sales'])
plt.plot(time_sales_2014.index, time_sales_2014['Sales'])
plot_acf(time_sales_2013)
plot_acf(time_sales_2014)
plot_pacf(time_sales_2013)


model1 = AR(time_sales_2013)
model_fit1 = model1.fit()
model_fit1.k_ar
model_fit1.params
predict_by_2013 = model_fit1.predict(start=len(test_data),
                                   end=len(time_sales_2013)-1, dynamic=False)

model2 = AR(time_sales_2014)
model_fit2 = model2.fit()
model_fit2.k_ar
model_fit2.params
predict_by_2014 = model_fit1.predict(start=len(test_data),
                                   end=len(time_sales_2014)-1, dynamic=False)


predict_2015 = (predict_by_2013 + predict_by_2014) / 2




