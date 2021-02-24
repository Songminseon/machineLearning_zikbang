# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
iris = datasets.load_iris()
X=iris['data']
y=iris['target']

bagging = BaggingClassifier(oob_score=True)
bagging.fit(X,y)

bagging.base_estimator_
bagging.estimators_

est = bagging.estimators_

est[0].predict(X)

bs_sets=bagging.estimators_samples_ #server all case prdict value
bagging.estimators_features_

bagging.oob_score_

knn=KNeighborsClassifier(n_neighbors=3)

bagging2 = BaggingClassifier(base_estimator=knn)
bagging2.fit(X,y)

bagging2.base_estimator_

y_pred=bagging2.predict(X)
bagging2.score(X,y)
y_prob=bagging2.predict_proba(X)

import matplotlib.pyplot as plt

knn=KNeighborsClassifier(n_neighbors=1)

bagging2 = BaggingClassifier(base_estimator=knn)
bagging2.fit(X[:,:2],y)

x_min, x_max=X[:,0].min()-1, X[:,0].max()+1
y_min, y_max=X[:,1].min()-1, X[:,1].max()+1

XX,YY=np.meshgrid(np.linspace(x_min,x_max,100), np.linspace(y_min,y_max,100))
ZZ=np.c_[XX.ravel(),YY.ravel()]
ZZ_pred=bagging2.predict(ZZ)

plt.contourf(XX,YY,ZZ_pred.reshape(XX.shape),cmap=plt.cm.RdYlBu,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y,s=30)

knn1=bagging2.estimators_[0]
Z_pred_knn=knn1.predict(ZZ)

plt.contourf(XX,YY,Z_pred_knn.reshape(XX.shape),cmap=plt.cm.RdYlBu,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y,s=30)

from sklearn.tree import DecisionTreeClassifier

t=DecisionTreeClassifier(max_depth=3)

bagging3=BaggingClassifier(base_estimator=t)
bagging3.fit(X[:,:2],y)

ZZ_pred=bagging3.predict(ZZ)

plt.contourf(XX,YY,ZZ_pred.reshape(XX.shape),cmap=plt.cm.RdYlBu,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y,s=30)


t1=bagging3.estimators_[0]
ZZ_pred_tree=t1.predict(ZZ)


plt.contourf(XX,YY,ZZ_pred_tree.reshape(XX.shape),cmap=plt.cm.RdYlBu,alpha=0.8)
plt.scatter(X[:,0],X[:,1],c=y,s=30)

from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_neighbors=3)

x=np.random.uniform(-4,4,100)
y=np.sin(x)+np.random.normal(size=100, scale=0.4)

plt.scatter(x,y)

bagging=BaggingRegressor(base_estimator=knn)
bagging.fit(x.reshape((-1,1)),y)

y_pred=bagging.predict(x.reshape((-1,1)))

xx=np.linspace(-4,4,100)
yy=bagging.predict(xx.reshape((-1,1)))

plt.scatter(x,y)
for est in bagging.estimators_:
    yy=est.predict(xx.reshape((-1,1)))
    plt.plot(xx,yy,'gray',ls=":")
plt.plot(xx,yy,'r')

x,y=datasets.make_hastie_10_2(random_state=0)

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(max_depth=3,min_samples_split=200,max_features='sqrt')

rf.fit(x,y)

rf.estimators_


x,y=datasets.make_friedman1(n_samples=1200, random_state=-0, noise=1.0)


from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(max_depth=5, min_samples_split=20,max_features='sqrt')
rf_reg.fit(x,y)

rf_reg.estimators_


from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

clf1=LogisticRegression(max_iter=300)
clf2=KNeighborsClassifier(n_neighbors=3)
clf3=DecisionTreeClassifier(max_depth=5)

ens_clf=VotingClassifier(estimators=[('lr',clf1),('knn',clf2),('tree',clf3)], voting='hard')
x,y=datasets.make_hastie_10_2(random_state=0)
ens_clf.fit(x,y)

