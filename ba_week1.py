import pandas as pd
import matplotlib.pyplot as plt
house = pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')

house['bedrooms'].mean()
house['bedrooms'].var()
house['bedrooms'].std()
house['bedrooms'].min()
house['bedrooms'].max()
house['bedrooms'].median()
house['bedrooms'].quantile(0.25)
house['bedrooms'].describe()

fig=plt.figure(figsize=(10,8))
house['bedrooms'].plot.box()

fig=plt.figure(figsize=(10,8))
house['bathrooms'].plot.box()

fig=plt.figure(figsize=(10,8))
plt.boxplot(house['bathrooms'], whis=(0,100)) #whis(0,100) => 0%부터 100%까지

fig=plt.figure(figsize=(10,8))
house['bedrooms'].plot.hist(bins=20)

fig=plt.figure(figsize=(10,8))
house['bedrooms'].plot.hist(bins=20)

fig=plt.figure(figsize=(10,8))
house['bedrooms'].plot.kde(bw_method=2)

fig=plt.figure(figsize=(10,8))
house.groupby('yr_built')['price'].mean().plot()

fig=plt.figure(figsize=(10,8))
house[house['yr_renovated']==0].groupby('yr_built')['price'].mean().plot()

fig=plt.figure(figsize=(10,8))
house[house['yr_renovated']>0].groupby('yr_renovated')['price'].mean().plot()

from pandas.plotting import scatter_matrix

scatter_matrix(house[['price','bedrooms','bathrooms']], figsize=(10,10))

fig=plt.figure(figsize=(10,8))
plt.scatter(house['bedrooms'], house['price'])


corr=house[['price','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors']].corr()

fig=plt.figure(figsize=(12,8))
cax=plt.imshow(corr, vmin=-1, vmax=1, cmap=plt.cm.RdBu)
plt.colorbar(cax)

freq=house['grade'].value_counts()

fig=plt.figure(figsize=(12,8))
house['waterfront'].value_counts().plot.bar()

fig=plt.figure(figsize=(12,8))
house.boxplot(column=['price'], by='waterfront')

import statsmodels.api as sm
from sklearn import datasets

boston=datasets.load_boston()
X=boston.data
y=boston.target

X=boston.data
y=boston.target

X=sm.add_constant(X)

model=sm.OLS(y,X)
result=model.fit()
result.summary()


    