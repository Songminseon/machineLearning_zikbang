# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, weights=[0.01,0.05,0.94],
                           class_sep=0.8, random_state=0)

plt.scatter(X[:,0],X[:,1],c=y)

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy="auto", random_state=0)

X_resampled, y_resampled = ros.fit_resample(X,y)

np.bincount(y_resampled)

num_samples=np.bincount(ind)

plt.scatter(X[:,0],X[:,1])

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0, sampling_strategy='auto')
X_resampled, y_resampled = rus.fit_resample(X,y)

np.bincount(y_resampled)

plt.scatter(X_resampled[:,0],X_resampled[:,1],c=y_resampled)


rus = RandomUnderSampler(random_state=0, sampling_strategy={1:64*2,2:64*10}) 
X_resampled, y_resampled = rus.fit_resample(X,y)

np.bincount(y_resampled)

from imblearn.over_sampling import SMOTE

sm=SMOTE(k_neighbors=5, random_state=0)
X_resampled, y_resampled = sm.fit_resample(X,y)

np.bincount(y_resampled)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)

from imblearn.over_sampling import ADASYN

ada=ADASYN(random_state=0, n_neighbors=5)

X_resampled, y_resampled = ada.fit_resample(X,y) 

np.bincount(y_resampled)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)

from imblearn.under_sampling import NearMiss
nm=NearMiss(version=1)
nm.sample_indices=True
X_resampled, y_resample = nm.fit_resample(X,y)

np.bincount(y_resampled)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)

deleted_ind = np.setdiff1d(np.arange(len(X)), ind)

plt.scatter(X[deleted_ind,0],X[deleted_ind,1],c=y[deleted_ind], marker='x', alpha=0.2
plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)

from imblearn.under_sampling import OneSidedSelection

oss=OneSidedSelection(random_state=0, n_neighbors=1, n_seeds_S=1)

X_resampled, y_resampled = oss.fit_resample(X,y)

np.bincount(y_resampled)


deleted_ind = np.setdiff1d(np.arange(len(X)), ind)

plt.scatter(X[deleted_ind,0],X[deleted_ind,1],c=y[deleted_ind], marker='x', alpha=0.2
plt.scatter(X_resampled[:,0], X_resampled[:,1], c=y_resampled)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c="gray", alpha=0.2)
plt.scatter(X[deleted_ind,0], X[deleted_ind,1], c=y[deleted_ind], marker='x')

colors = plt.cm.virdis(y[deleted_ind]/2)
plt.scatter(X_resampled[:,0], X_resampled[:,1], c="gray", alpha=0.2)
plt.scatter(X[deleted_ind,0], X[deleted_ind,1], c=colors, marker='x')

from imblearn.under_sampling import TomekLinks

tl=TomekLinks(sampling_strategy="all")

X_resampled, y_resampled = tl.fit_resample(X,y)

deleted_ind=np.setdiff1d(np.arange(len(X)), ind)
colors=plt.cm.viridis(y[deleted_ind]/2)

plt.scatter(X_resampled[:,0], X_resampled[:,1], c='gray', alpha=0.2)
plt.scatter(X[deleted_ind,0], X[deleted_ind,1], c=colors, marker='x')

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()
clf.fit(X,y)

xmin,xmax,ymin,ymax=X[:,0].min(), X[:,0].max(),X[:,1].min(),X[:,1].max()
xx,yy = np.meshgrid(np.linspace(xmin-0.5,xmax+0.5,100), np.linspace(ymin-0.5,ymax+0.5,100))
zz=np.c_[xx.ravel(),yy.ravel()]
zz_pred=clf.predict(zz)

plt.contourf(xx,yy,zz_pred.reshape(xx.shape), alpha=0.7)
plt.scatter(X[:,0],X[:,1],c=y)

clf.fit(X_resampled, y_resampled)
zz_pred=clf.predict(zz)
