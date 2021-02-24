# -*- coding: utf-8 -*-
#goal1
#variable set으로 나누고 결과 설명(view,condition,grade)
#describe idea to utilize zipcode, lat, and long

#goal2
# how to use this information with other information
# think some idea how to utilize this varaible with addtion with other information
# don't need to implement my idea, not programming assignment


#view : An index from 0 to 4 of how good the view of the property was

#condition : An index from 1 to 5 on the condition of the aprtment

#grade: An index from 1 to 13, where 1-3 falls short of building construction
#and design, 7 has an average level of construction and design, and 11-13
#have a high-quality level of construction and design


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression

house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')
#by lecture, i set input variables with constant

varlist=['bedrooms','bathrooms','floors','waterfront','sqft_above',
         'sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15']

X=house[varlist]
y=house['price']

#check OLS
X=sm.add_constant(X)
model=sm.OLS(y,X)
result=model.fit()

result.summary()

##check category
house['view'].value_counts()  #0,1,2,3,4
house['condition'].value_counts() #1,2,3,4,5
house['grade'].value_counts() #1,2,3,...,13



plt.bar(house['view'].value_counts().keys(),house['view'].value_counts().values)
plt.bar(house['condition'].value_counts().keys(),house['condition'].value_counts().values)
plt.bar(house['grade'].value_counts().keys(),house['grade'].value_counts().values)


catvar=['view','condition','grade']
X1=X

for c in catvar:
    dummy=pd.get_dummies(house[c], prefix=c, drop_first=True)
    X1=pd.concat((X1,dummy),axis=1)
    
    
#model1    check
X1=sm.add_constant(X1)
model1=sm.OLS(y,X1)
result1=model1.fit()
result1.summary()


#score 0.684
    
##remodeling
##make grade column for new varaible
catvar = ['view', 'condition', 'grade']
X2=X

plt.boxplot(house['view']) #only view0
plt.boxplot(house['condition']) #only 2, 3, 4, 5
plt.boxplot(house['grade']) #only 6,7,8,9

for c in catvar:
    X2=pd.concat((X2, house[c]), axis=1)
    
cond = (house['view']==0) & (house['condition']>=2) & (house['condition']<=5) & (house['grade']>=6) & (house['grade']<=9)

X2=house[cond][varlist+catvar]
y2=house[cond]['price']

for c in catvar:
    dummy=pd.get_dummies(X2[c], prefix=c, drop_first=True)
    X2=pd.concat((X2,dummy),axis=1)
    
X2=X2.drop(['grade','condition','view'],axis=1)

X2=sm.add_constant(X2)
model2=sm.OLS(y2,X2)
result2=model2.fit()
result2.summary()

from sklearn.metrics import mean_squared_error
a=mean_squared_error(y, result1.predict(X1))
b=mean_squared_error(y2, result2.predict(X2))






##check for zipcode, lat, long
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')

X=house[['zipcode','lat','long']]

house['zipcode'].plot.kde()
house['lat'].plot.kde()
house['long'].plot.kde()
house['zipcode'].min()
house['zipcode'].max()

scatter_matrix(X)

##use regression from sklear
x1=X[['lat','long']]
y=X['zipcode']
reg=LinearRegression()
reg.fit(x1,y)
reg.score(x1,y)

## use statsmodels
import statsmodels.api as sm
x1=X[['lat','long']]
y=X['zipcode']

x1=sm.add_constant(x1)
model1=sm.OLS(y,x1)
result1=model1.fit()
result1.summary()


import folium

m=folium.Map(location=[47.560052,-122.213896], zoom_start=5)

for i in range(0, len(house)):
    folium.Circle(
        location = [house['lat'][i], house['long'][i]],
        tooltip = house['zipcode'][i],
        radius = 100).add_to(m)

folium.Marker(
    location =[47.619708,-122.3225248],
    popup="Captio Hill Station"
    ).add_to(m)

folium.Marker(
    location =[47.6119999,-122.3386432],
    popup="Westlake Station"
    ).add_to(m)

folium.Marker(
    location =[47.5983889,-122.3320973],
    popup="King street Station"
    ).add_to(m)


folium.Marker(
    location =[47.6025616,-122.3335043],
    popup="Pioneer square station"
    ).add_to(m)

m.save('C:/Users/user/Desktop/example2.html')

#captio hill station
#47.619708,-122.3225248

#westlake station
#47.6119999,-122.3386432

#King street Station
#47.5983889,-122.3320973

#pioneer square station
#47.6025616,-122.3335043

## 지하철거리
## 지형적 조건 무시
## 거리단위 불정확