# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor

##partition data
diabetes = datasets.load_diabetes()
X=diabetes.data
y=diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=100)

knn1=KNeighborsRegressor(n_neighbors=5)
knn2=KNeighborsRegressor(n_neighbors=7)

knn1.fit(X_train, y_train)
knn1.score(X_test, y_test)

knn2.fit(X_train, y_train)
knn2.score(X_test, y_test)

##score model1<model2

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=200)

knn1.fit(X_train, y_train)
knn1.score(X_test, y_test)

knn2.fit(X_train, y_train)
knn2.score(X_test, y_test)

##score model1>model2

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

X = ['a','b','c','d','e','f']
kf=KFold(n_splits=3, shuffle=True, random_state=1)

for train,test in kf.split(X):
    print("train:%s, Test:%s"%(train,test))
    
X=np.ones(10)
y=np.array([0,0,0,0,1,1,1,1,1,1])

skf=StratifiedKFold(n_splits=3, shuffle=True)

for train,test in skf.split(X,y):
    print("Train:%s(0:%d, 1:%d), Test:%s(0:%d, 1:%d)"%
          (train, sum(y[train]==0), sum(y[train]==1), test, sum(y[test]==0), sum(y[test]==1)))   

kf=KFold(n_splits=3, shuffle=True)
for train,test in skf.split(X,y):
    print("Train:%s(0:%d, 1:%d), Test:%s(0:%d, 1:%d)"%
          (train, sum(y[train]==0), sum(y[train]==1), test, sum(y[test]==0), sum(y[test]==1)))   
    
X=np.array([0.1,0.2,2.2,2.4,2.3,4.55,5.8,8.8,9,10])
y=np.array([0,1,1,1,2,2,2,3,3,3])
groups=np.array([1,1,1,2,2,2,3,3,3,3])

gkf=GroupKFold(n_splits=3)    
for train,test in gkf.split(X,groups=groups, y=y):
    print("Train:%s(%s), Test:%s(%s)"
          %(train,np.unique(groups[train]),test,np.unique(groups[test])))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
digits=datasets.load_digits()
X=digits.data
y=digits.target

C_s=np.logspace(-10,0,10)

logistic=LogisticRegression()

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

accs=[]
for c in C_s:
    logistic.C=c
    temp=[]
    for train,test in skf.split(X,y):
        logistic.fit(X[train],y[train])
        temp.append(logistic.score(X[test],y[test]))
    accs.append(temp)
    
accs=np.array(accs)
avg=np.mean(accs,axis=1)
avg
np.argmax(avg)
C_s[np.argmax(avg)]

ks=np.linspace(1,10,10)
knn3=KNeighborsClassifier()

accs2=[]

for k in ks:
    knn3.n_neighbors=int(k)
    temp=[]
    for train,test in skf.split(X,y):
        knn3.fit(X[train],y[train])
        temp.append(knn3.score(X[test],y[test]))
    accs2.append(temp)

np.mean(accs2,axis=1)
np.argmax(np.mean(accs2,axis=1))
ks[np.argmax(np.mean(accs2,axis=1))]
