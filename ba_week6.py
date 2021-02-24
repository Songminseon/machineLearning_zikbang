# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

y_true = np.array([2,0,2,2,0,1])
y_pred = np.array([0,0,2,2,0,1])

accuracy_score(y_true, y_pred)

y_true2 = np.array([0,2,2,2,0,2])
y_pred2 = np.array([0,0,2,2,2,2])

recall_score(y_true2, y_pred2, pos_label=2)

recall_score(y_true, y_pred, average='micro')
recall_score(y_true, y_pred, average="macro")
recall_score(y_true, y_pred, average="weighted")

recall_score(y_true, y_pred, labels=[0,1,2], average=None)

from sklearn.metrics import classification_report

print(classification_report(y_true,y_pred))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_true, y_pred)

data = pd.read_csv('https://drive.google.com/uc?export=download&id=1Bs6z1GSoPo2ZPr5jL2qDjRghYcMUOHbS')

data['target']=(data['quality']>=7)*1 #good quality
data['target'].sum()/len(data) #checck amount good quality data


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

clf1=LogisticRegression(C=1, max_iter=1000)
clf2=DecisionTreeClassifier(max_depth=5)

X=data.drop(['quality','target'], axis=1)
y=data['target']

trnX,valX,trnY,valY=train_test_split(X,y,stratify=y, test_size=0.2, random_state=50)

clf1.fit(trnX,trnY)
clf2.fit(trnX,trnY)

y_pred1=clf1.predict(valX)
y_pred2=clf2.predict(valX)
print('Accuracy: Logistic=%.4f    Tree=%.4f'%(accuracy_score(valY,y_pred1),
                                              accuracy_score(valY,y_pred2)))

metrics=[recall_score, precision_score, f1_score]
for nm,m in zip(('Recall','Precision','F1'), metrics):
    print('%s: Logistic=%.4f       Tree=%.4f'%(nm,m(valY,y_pred1, pos_label=1),
                                               m(valY,y_pred2,pos_label=1)))
y_prob=clf1.predict_proba(valX)

from sklearn.metrics import roc_curve, roc_auc_score
    
fpr, tpr, thresholds = roc_curve(valY, y_prob[:,1], pos_label=1)    

import matplotlib.pyplot as plt

xx=np.linspace(0,1,10)

plt.plot(fpr,tpr)
plt.plot(xx,xx,'k--')

roc_auc_score(valY, y_prob[:,1])

