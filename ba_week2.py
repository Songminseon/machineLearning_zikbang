# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import statsmodels.api as sm

salary=pd.read_csv('https://drive.google.com/uc?export=download&id=1kkAZzL8uRSak8gM-0iqMMAFQJTfnyGuh')

dummy=pd.get_dummies(salary['sex'], prefix='sex', drop_first=True)

varname='rank'
gmean=salary.groupby(varname)['salary'].mean()
gstd=salary.groupby(varname)['salary'].std()

plt.bar(range(len(gmean)), gmean)
plt.errorbar(range(len(gmean)), gmean, yerr=gstd, fmt='o', c='r', ecolor='r',
             capthick=2, capsize=3)
plt.xticks(range(len(gmean)), gmean.index)

salary['rank'].value_counts()
groups=[x[1].values for x in salary.groupby(['rank'])['salary']]

f_oneway(*groups)

catvar=['rank','discipline','sex']

for c in catvar:
    dummy=pd.get_dummies(salary[c], prefix=c, drop_first=True)
    salary=pd.concat((salary,dummy),axis=1)
    
X=salary.drop(catvar+['salary'], axis=1)
y=salary['salary']

X=sm.add_constant(X)
model=sm.OLS(y,X)
result=model.fit()

result.summary()

##King country with house prices
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import probplot
from statsmodels.stats import diagnostic

house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')

varlist=['bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront',
         'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
         'sqft_living15', 'sqft_lot15']

X1=house[varlist]
X1=sm.add_constant(X1)
y1=house['price']

model1=sm.OLS(y1,X1)
result1=model1.fit()
result1.summary()

xx=np.linspace(y1.min(),y1.max(),100)

y_pred1=result1.predict(X1)
err1=y1-y_pred1

plt.scatter(y1,y_pred1)
plt.plot(xx,xx,color='k')
plt.ylabel('Predicted')
plt.xlabel('Real')

##Need to find another model, when price is higher, it seems under-estimate

plt.hist(err1, bins=50)
probplot(err1,plot=plt)

##
diagnostic.het_breuschpagan(err1,X1) ##result ftest, chi-square
diagnostic.het_breuschpagan(err1, X1[['bedrooms','bathrooms']]) 
##to delete outliers

cond=(house['price']<1000000)&(house['price']>=20000)
X2=house[cond][varlist]
y2=house[cond]['price']
X2=sm.add_constant(X2)

y2.plot.kde()

model2=sm.OLS(y2,X2)
result2=model2.fit()
result2.summary()

y_pred2=result2.predict(X2)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error

mean_squared_error(y1,y_pred1)
mean_squared_error(y2,y_pred2)
mean_squared_error(y1[cond],y_pred1[cond])



##although we exclude outlier, the score is smaller than previous model.
##SST의 값이 작아지면서 r2의 값은 작아짐 (r2=SSR/SST)
##previous model is greater than new model    


