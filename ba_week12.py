# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

n, p=50,4
na_size=10

np.random.seed(96)
X=np.random.randn(n,p)
na_ind = np.random.randint(n*p, size=na_size)
rows, cols = np.unravel_index(na_ind,(n,p))

X[rows,cols] = np.nan

np.mean(X, axis=0)

df=pd.DataFrame(X, columns=['X%d'%(i) for i in range(1,1+p)])
df.mean() #get exclude missing value

np.mean(X[np.isnan(X[:,0])==False,0])

df.dropna()
df.dropna(axis=1)
df.dropna(axis=1, thresh=48)

df.cov()
df.dropna().cov()

df['X1'].dropna().var()
df['X1'].var()


x1_na = df[df['X1'].isna()].index

df.loc[x1_na,'X1']
df['X1'].fillna(df['X1'].mean()).loc[x1_na] #nan값 평균으로 넣기
df['X1'].fillna(method="ffill").loc[x1_na]
df['X1'].fillna(method="ffill").loc[x1_na-1]
df['X1'].fillna(method="bfill").loc[x1_na]
df['X1'].fillna(method='bfill').loc[x1_na+1]

from sklearn.datasets import load_diabetes

X_diabetes, y_diabetes = load_diabetes(return_X_y = True)

def add_missing_value(X_full, y_full, random_state=None):
    n_samples, n_features = X_full.shape
    missing_rate=0.75
    n_missing_samples = int(n_samples*missing_rate)
    
    rng=np.random.RandomState(random_state)
    missing_samples = np.zeros(n_samples, dtype=np.bool)
    missing_samples[:n_missing_samples]=True
    rng.shuffle(missing_samples)
    
    missing_features=rng.randint(0, n_features, n_missing_samples)
    X_missing = X_full.copy()
    X_missing[missing_samples, missing_features] = np.nan
    y_missing = y_full.copy()
    return X_missing, y_missing

X_miss_diabetes, y_miss_diabetes = add_missing_value(X_diabetes, y_diabetes)

from sklearn.impute import SimpleImputer, KNNImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="median", add_indicator=True)

X_filled = imputer.fit_transform(X_miss_diabetes)

X_miss_diabetes[np.isnan(X_miss_diabetes[:,0])==False, 0].mean()

imputer = KNNImputer(missing_values=np.nan, n_neighbors=3, add_indicator=True)
X_filled = imputer.fit_transform(X_miss_diabetes)

from scipy.stats import norm, binom
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def scatter_dist(x,y,bins=20):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    # start with a rectangular Figure
    fig=plt.figure(figsize=(8, 8))    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    
    # the scatter plot:
    ax_scatter.scatter(x, y, alpha=0.7)
    
    # now determine nice limits by hand:
    #lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    #ax_scatter.set_xlim((-lim, lim))
    #ax_scatter.set_ylim((-lim, lim))
    
    ax_histx.hist(x, bins=bins,alpha=0.5)    
    ax_histx2=ax_histx.twinx()
    kdex=gaussian_kde(x)
    xlim=ax_scatter.get_xlim()
    xx = np.linspace(xlim[0],xlim[-1],100)
    ax_histx2.plot(xx,kdex.pdf(xx))
    ax_histy.hist(y, bins=bins, orientation='horizontal',alpha=0.5)
    ax_histy2=ax_histy.twiny()
    kdey=gaussian_kde(y)
    ylim=ax_scatter.get_ylim()
    yy = np.linspace(ylim[0],ylim[-1],100)
    ax_histy2.plot(kdey.pdf(yy), yy)
    
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    return fig,ax_scatter



np.random.seed(654654)
N = np.arange(1500)

a=0
b=1

eps=np.array([norm(0,n).rvs() for n in N])
y=(a+b*N+eps)/100
x=(N+norm(10,10).rvs(len(N)))/100


y[binom(1,0.3).rvs(len(N))==1]=np.nan

data_het_miss = pd.DataFrame({'y':y, 'x':x})

scatter_dist(X_filled[:,0], X_filled[:,1])

from autoimpute.imputations import SingleImputer

compare_methods=['mean','norm','interpolate']

variance=[]
for m in compare_methods:
    sim=SingleImputer(strategy=m)
    sim.fit(data_het_miss)
    data_fill=sim.transform(data_het_miss)
    variance.append(data_fill['y'].var())
    
    fig, ax=scatter_dist(data_fill['x'],data_fill['y'])
    ax.scatter(data_fill[data_het_miss['y'].isnull()]
               ['x'], data_fill[data_het_miss['y'].isnull()]['y'], c='r', alpha=0.7)
    plt.title(m, fontsize=16)
    plt.show()    
    
    
compare_methods=['least squares', 'stochastic']

variance=[]
for m in compare_methods:
    sim=SingleImputer(strategy=m, predictors={'y':['x']})
    sim.fit(data_het_miss)
    data_fill=sim.transform(data_het_miss)
    variance.append(data_fill['y'].var())
    
    fig, ax=scatter_dist(data_fill['x'],data_fill['y'])
    ax.scatter(data_fill[data_het_miss['y'].isnull()]
               ['x'], data_fill[data_het_miss['y'].isnull()]['y'], c='r', alpha=0.7)
    plt.title(m, fontsize=16)
    plt.show()    
    
n_samples = 10000
toy_df = pd.DataFrame({'gender':np.random.choice(['Male','Feamle'], n_samples), 'employment':
                       np.random.choice(['Unemployed', 'Employed','Part Time', 'Self-Employed'],
                       n_samples,p=[0.05,0.6,0.15,0.2])})
    
toy_df['gender'].value_counts()
toy_df['employment'].value_counts()

for c in toy_df.columns:
    toy_df.loc[np.random.choice(range(n_samples), replace=False, size=100), c] = np.nan

sim=SingleImputer(strategy='categorical')
sim.fit(toy_df)
toy_df_fill=sim.transform(toy_df)

toy_df_fill.loc[toy_df['employment'].isnull(), 'employment'].value_counts()

toy_df.loc[toy_df['employment'].isnull()==False, 'employment'].value_counts()/sum(toy_df['employment'].isnull()==False)


from autoimpute.imputations import MultipleImputer

mi=MultipleImputer(strategy='stochastic', return_list=True, predictors={'y':['x']})
mi_data_fill=mi.fit_transform(data_het_miss)



