##assignment2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')
reg=LinearRegression()

##check correlation

corr = house.corr()
data_corr = np.array(house.corr()['price'])
data_corr = np.sort(np.abs(data_corr),axis = 0 )

##regression model
X=house[['sqft_living','grade','sqft_above','sqft_living15','bathrooms']]
y=house['price']
reg.fit(X,y)
r2 = reg.score(X,y)


##checking by visualization
plt.scatter(house['sqft_living'],y)
plt.scatter(house['grade'],y) 
plt.scatter(house['sqft_above'],y)
plt.scatter(house['sqft_living15'],y)
plt.scatter(house['bathrooms'],y) 

scatter_matrix(X, figsize=(10,10))



##check by OLS
import statsmodels.api as sm
X = sm.add_constant(X)
model=sm.OLS(y,X)
result = model.fit()
result.summary()

##check vif
vif_list = []
for i in X.columns[1:6]:
    reg.fit(house[[i]],y)
    vif_list.append(1/(1-reg.score(house[[i]],y)))
    
