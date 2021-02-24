# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import seaborn as sn

data=pd.read_csv('C:/Users/smsun/desktop/final-1.csv', encoding='utf-8')

dt=data.drop(['Unnamed: 0'],axis=1)

head=dt.head(100)
dt.columns
dt.info()

for i in ['dong', 'year_of_completion',
       'floor', 'gu']:
    print(i,'------------------------------------')
    print(dt[i].value_counts(),"\n")

dong_counts=dt['dong'].value_counts()
year_of_completion_counts=dt['year_of_completion'].value_counts()
floor_counts=dt['floor'].value_counts()
gu_counts=dt['gu'].value_counts()



def dummi(dt):
    dum=pd.get_dummies(dt[['dong', 'year_of_completion',
       'floor', 'gu']])
    dum=pd.concat([dum, dt[['exclusive_use_area',
       'transaction_real_price','popularity', 'mail#',
       'femail#', 'korean#', 'korean_mail#', 'korean_femail#',
       'total_foreigner', 'foreigner_mail#', 'foreigner_femail#', 'over65#',
       'square_meters', 'popularity_density', 'households', 'near_subway#',
       'crime5_occur#', 'police#', 'fire_station#', 'total_student#',
       'SNU_stduent#', 'park#', 'park_area', 'care_center#', 'library#',
       'theater#', 'gym#', 'total_store#', 'franchise_stroe#',
       'normal_stroe#']]],axis=1)
    return dum

dt_dum=dummi(dt)
corr=dt_dum.corr()

corr_list = corr['transaction_real_price'].sort_values(ascending=False)
corr_list_abs = abs(corr['transaction_real_price']).sort_values(ascending=False)
#at least 0.19
varlist = list(corr_list_abs[1:19].index)
data=dt_dum[['transaction_real_price','exclusive_use_area',
 'SNU_stduent#',
 'gu_강남구',
 'crime5_occur#',
 'gu_서초구',
 'dong_반포동',
 'park#',
 'fire_station#',
 'police#',
 'dong_대치동',
 'dong_압구정동',
 'over65#',
 'theater#',
 'gu_노원구',
 'dong_한남동',
 'gym#',
 'dong_잠실동',
 'total_store#']]

data.columns=['transaction_real_price','exclusive_use_area', 'SNU_stduent#', 'gu_Gangnam', 'crime5_occur#',
       'gu_Seocho', 'dong_Banpo', 'park#', 'fire_station#', 'police#', 'dong_Daechi',
       'dong_Apgujeong', 'over65#', 'theater#', 'gu_Nowon', 'dong_Hanam', 'gym#',
       'dong_Jamsil', 'total_store#']



data_corr=data.corr()

plt.rcParams['figure.figsize']=[7,7]
sn.heatmap(data_corr,annot=True)
a=data.loc[(data['over65#']>6000)]

b=data.loc[(data['crime5_occur#']==data['crime5_occur#'].max())]
data=data.loc[(data['exclusive_use_area']<250)]

## draw graph
dummyInCorr=['gu_강남구','gu_서초구','dong_반포동','dong_대치동','dong_압구정동','gu_노원구','dong_한남동','dong_잠실동']

graph_list = list(set(varlist)-set(dummyInCorr))

for i in graph_list:
    plt.boxplot(data[i])
    plt.title("Boxplot Of "+i)
    root_path = "C:/Users/smsun/Desktop/"
    file_path = "boxplot_"+i
    plt.savefig(root_path+file_path)
    plt.show()

for i in graph_list:
   plt.scatter(data[i], y)
   plt.title("Scatter plot of "+i)
   root_path = "C:/Users/smsun/Desktop/"
   file_path = "scatterplot_"+i
   plt.savefig(root_path+file_path)
   plt.show()
   
   
data3=pd.read_csv('C:/Users/smsun/desktop/final.csv', encoding='utf-8')
check_list=['transaction_id', 'apartment_id', 'city', 'dong', 'jibun', 'apt',
       'addr_kr', 'exclusive_use_area', 'year_of_completion',
       'transaction_year_month', 'transaction_date', 'floor',
       'transaction_real_price']

data4 = data3[check_list]
dummy=pd.get_dummies(data4['dong'])
dummy=pd.concat([dummy,data4],axis=1)
a=dummy.corr()
b=a['transaction_real_price'].sort_values()
b['floor']
b['year_of_completion']

plt.scatter(data3['year_of'])

##draw scatter plot
train_data=pd.read_csv('C:/Users/smsun/Desktop/final.csv')
y=train_data['transaction_real_price']
plt.scatter(train_data['floor'],y)
plt.title("scatter floor")
plt.scatter(train_data['year_of_completion'],y)
plt.title("scatter completion year")

##draw graph by dong
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import seaborn as sn
import time


data2=pd.read_csv('C:/Users/smsun/Desktop/2학기/business analytics/team_project/final-1.csv')
data3=pd.read_csv('C:/Users/smsun/Desktop/2학기/business analytics/team_project/train.csv')
data['transaction_real_price'].max()
plt.hist(data['transaction_real_price'],bins=1000)


dt=data.drop(['Unnamed: 0'],axis=1)
dt.info()
y=dt['transaction_real_price']
a=dt['dong'].value_counts().reset_index().drop(['dong'],axis=1).values.flatten()
dtt=dt[['dong', 'exclusive_use_area','transaction_real_price', 'year_of_completion', 'floor']]
graph_list = ['floor','year_of_completion']
for dong in a:
    data=dtt.loc[dt['dong']==dong]
    y=data['transaction_real_price']
    for i in graph_list:
        print(dong,"----",i)
        plt.scatter(data[i], y)
        plt.title(i)
        plt.show()
        time.sleep(1)

b=set(dt['dong'])

plt.boxplot(data['transaction_real_price'])

##draw all scatter
y = data['transaction_real_price']
x_list = ['exclusive_use_area', 'year_of_completion', 'floor', 'popularity', 'mail#', 'femail#',
            'korean#', 'korean_mail#', 'korean_femail#','total_foreigner', 'foreigner_mail#', 'foreigner_femail#', 'over65#',
       'square_meters', 'popularity_density', 'households', 'near_subway#',
       'crime5_occur#', 'police#', 'fire_station#', 'total_student#',
       'SNU_stduent#', 'park#', 'park_area', 'care_center#', 'library#',
       'theater#', 'gym#', 'total_store#', 'franchise_stroe#',
       'normal_stroe#','add']

for i in x_list:
   plt.scatter(data[i], y)
   plt.title("Scatter plot of "+i)
   root_path = "C:/Users/smsun/Desktop/plot"
   file_path = "scatterplot_"+i
   plt.savefig(root_path+file_path)
   plt.show()
   
a = data['transaction_real_price']

data = data[data['transaction_real_price']>200000]

data = data[data['transaction_real_price']<126000]



##project3 decision tree with dong and gu
from sklearn.tree import DecisionTreeClassifier                            
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#data=pd.read_csv('C:/Users/smsun/Desktop/2학기/business analytics/team_project/week3.csv')
data10=pd.read_csv('C:/Users/smsun/Desktop/data_outlier.csv')


data = pd.read_csv('C:/Users/smsun/Desktop/data_outlier.csv')
data = data[data['transaction_real_price']<126000]


input_list=['exclusive_use_area', 'year_of_completion','police#',
            'near_subway#', 'crime5_occur#','SNU_student#','park_area','park#',
            'total_store#','franchise_store#','normal_store#','gu','dong']

train_X = data[input_list]
train_y = data['transaction_real_price'] - data['transaction_real_price']%100

label_column1 = train_X[['gu']]
label_column2 = train_X[['dong']]

encoder1=LabelEncoder()
encoder1.fit(label_column1)
labels=encoder1.transform(label_column1)

encoder2=LabelEncoder()
encoder2.fit(label_column2)
labels2=encoder2.transform(label_column2)

labels = pd.DataFrame(labels, columns=['gu'])
labels2 = pd.DataFrame(labels2, columns=['dong'])


train_X = train_X.drop(['gu','dong'], axis=1)
train_X = pd.concat([labels,train_X], axis=1)
train_X = pd.concat([labels2,train_X], axis=1)


kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train, test in kf.split(train_X):
    print(train, test)


train_X1 = train_X.iloc[train]
test_X1 = train_X.iloc[test]
train_y1 = train_y.iloc[train]
test_y1 = train_y.iloc[test]


clf = DecisionTreeClassifier()
clf.fit(train_X1, train_y1)


y_predict = clf.predict(test_X1)
rms = sqrt(mean_squared_error(test_y1,y_predict))  ##5536


##decision tree with dong, drop gu
train_X_2 = data[input_list]
train_y_2 = data['transaction_real_price'] - data['transaction_real_price']%100

label_column1 = train_X_2[['gu']]
label_column2 = train_X_2[['dong']]

encoder1=LabelEncoder()
encoder1.fit(label_column1)
labels=encoder1.transform(label_column1)

encoder2=LabelEncoder()
encoder2.fit(label_column2)
labels2=encoder2.transform(label_column2)

labels = pd.DataFrame(labels, columns=['gu'])
labels2 = pd.DataFrame(labels2, columns=['dong'])


train_X_2 = train_X_2.drop(['gu','dong'], axis=1)
train_X_2 = pd.concat([labels,train_X_2], axis=1)
train_X_2 = pd.concat([labels2,train_X_2], axis=1)
train_X_2 = train_X_2.drop(['gu'], axis=1)


train_X2 = train_X_2.iloc[train]
test_X2 = train_X_2.iloc[test]
train_y2 = train_y_2.iloc[train]
test_y2 = train_y_2.iloc[test]


clf2 = DecisionTreeClassifier(random_state=100)
clf2.fit(train_X2, train_y2)

y_predict2 = clf2.predict(test_X2)
rms2 = sqrt(mean_squared_error(test_y2,y_predict2))  

##deicision tree with gu, drop dong
train_X_3 = train_X_2.drop(['dong'], axis=1)
train_X_3 = pd.concat([labels, train_X_3], axis=1)

train_X3 = train_X_3.iloc[train]
test_X3 = train_X_3.iloc[test]
train_y3 = train_y.iloc[train]
test_y3 = train_y.iloc[test]

clf3 = DecisionTreeClassifier(criterion='entropy',random_state=250)
clf3.fit(train_X3, train_y3)
clf3.score(test_X3, test_y3)

y_predict3 = clf3.predict(test_X3)
rms3 = sqrt(mean_squared_error(test_y3, y_predict3)) 

from sklearn.tree import export_graphviz
from IPython.display import Image

export_graphviz(clf3, out_file='tree.dot')


##apply best parameter
clf4 = DecisionTreeClassifier(criterion='gini', random_state=80)
clf4.fit(train_X3, train_y3)
clf4.score(test_X3, test_y3)
y_predict4 = clf4.predict(test_X3)
rms4 = sqrt(mean_squared_error(test_y3, y_predict4))

feature_list = clf4.feature_importances_
feature_list = pd.Series(feature_list).sort_values()
feature_sr = pd.concat((feature_sr,pd.Series(feature_list)),axis=1)

clf4.feature_importances_
def plot_feature_importances(model,train):
    n_feature=train.shape[1]
    plt.barh(range(n_feature),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_feature),train.columns)
    plt.xlabel("feature_importances")
    plt.ylim(-1, n_feature)
    plt.show()
    
plot_feature_importances(clf4, train_X3)
plot_feature_importances(clf3, train_X3)

##find best parameter

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier      

from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=1)
param_grid = {'criterion':['gini','entropy'],
              'random_state':[20,30,50,100,200,250,300],
              'min_samples_split':[2,3,4,5],
              'max_leaf_nodes':[None,1,2,3,4],
              'max_depth':[None,1,2,3,4]
              
              }
from sklearn.model_selection import GridSearchCV, KFold
model = DecisionTreeClassifier() 
gcv = GridSearchCV(model, param_grid=param_grid, cv=kf,n_jobs=4,scoring='neg_root_mean_squared_error')
gcv.fit(train_X3, train_y3)
best_param = gcv.best_params_


##find random state detail
state_list = []
for i in range(30, 100):
    state_list.append(i)
param_grid2 = {'random_state':state_list}

model2 = DecisionTreeClassifier(criterion="gini")
gcv2 = GridSearchCV(model2, param_grid=param_grid2, cv=kf, n_jobs=4)
gcv2.fit(train_X3, train_y3)
best_param2 = gcv2.best_params_

##draw local outlier plot
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
data=pd.read_csv('C:/Users/smsun/Desktop/2학기/business analytics/team_project/week3.csv')
dataX = data[['total_store#']]
datay = data['transaction_real_price']

data_X = train_X[['year_of_completion']]
data_y = data['transaction_real_price']


lof2=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
plt.scatter(data_X, data_y)
y_pred2 = lof2.fit_predict(data_X)

plt.scatter(data_X, data_y, s=3)
plt.scatter(data_X[y_pred2==-1],data_y[y_pred2==-1],marker='x',c='r', s=3)
plt.figure(figsize=(30,15))




##크게 확대한 그래프
lof_x = data_X[y_pred2==-1]
lof_y = data_y[y_pred2==-1]

data=data[data['year_of_completion']<1980]
data_x2 = data[['year_of_completion']]
data_y2 = data['transaction_real_price']

plt.scatter(data_x2,data_y2)
plt.scatter(data_X[y_pred2==-1],data_y[y_pred2==-1],marker='x',c='r', s=5)


graph_list=[ 'exclusive_use_area', 'year_of_completion', 'police#',
       'near_subway#', 'crime5_occur#', 'SNU_student#', 'park_area', 'park#',
       'total_store#', 'franchise_store#', 'normal_store#']
##


## delete outlier

import numpy as np

delete_list=[]
for i in graph_list:
    data_X = data[[i]]
    lof=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = lof.fit_predict(data_X)
    delete_list.append(np.where(y_pred==-1))
    plt.scatter(data_X, data_y, s=0.5)
    plt.scatter(data_X[y_pred==-1],data_y[y_pred==-1],marker='x',c='r', s=0.5)


delete_list2 = []
for i in delete_list:
    for j in range(0,len(i)):
        delete_list2.extend((list(i)[j]))        
        
delete_list2 = list(set(delete_list2))



new_data = data.drop(delete_list2)

new_data.to_csv("C:/Users/smsun/Desktop/data.csv")

#check_data = data[data['year_of_completion']>1995]
#check_x = check_data[['year_of_completion']]
#check_y = check_data['transaction_real_price']
#lof3 = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
#y_pred3 = lof3.fit_predict(check_x)
#plt.scatter(check_x, check_y)
#plt.scatter(check_x[y_pred3==-1], check_y[y_pred3==-1], marker='x',c='r')


##project presentation3인데 다음주에 쓸거임
import pandas as pd

data=pd.read_csv('C:/Users/smsun/Desktop//data_outlier.csv', encoding="utf-8")
data = data[data['transaction_real_price']<126000]

input_list=['exclusive_use_area', 'year_of_completion','police#', 'fire_station#',
            'near_subway#', 'crime5_occur#','SNU_student#','park_area','park#',
            'gym#', 'theater#', 'total_store#','franchise_store#','normal_store#','gu','dong']

train_X = data[input_list]
train_y = data['transaction_real_price'] - data['transaction_real_price']%100
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

label_column1 = train_X[['gu']]
label_column2 = train_X[['dong']]

encoder1=LabelEncoder()
encoder1.fit(label_column1)
labels=encoder1.transform(label_column1)

encoder2=LabelEncoder()
encoder2.fit(label_column2)
labels2=encoder2.transform(label_column2)

labels = pd.DataFrame(labels, columns=['gu'])
labels2 = pd.DataFrame(labels2, columns=['dong'])


train_X = train_X.drop(['gu','dong'], axis=1)
train_X = pd.concat([labels,train_X], axis=1)
train_X = pd.concat([labels2,train_X], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train, test in kf.split(train_X):
    print(train, test)

train_X1 = train_X.iloc[train]
test_X1 = train_X.iloc[test]
train_y1 = train_y.iloc[train]
test_y1 = train_y.iloc[test]

from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='gini', random_state=50, min_samples_split=4)
rf.fit(train_X1,train_y1)
y_predict = rf.predict(test_X1)
rms = sqrt(mean_squared_error(test_y1,y_predict))
    
 

feature_list=rf.feature_importances_
plot_feature_importances(rf, train_X1)


kf = KFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {'criterion':['gini','entropy'],
              'random_state':[20,50,75,100,200],
              'min_samples_split':[2,4,5,6],
              'max_leaf_nodes':[None,1,2,3,4],
              'max_depth':[None,1,2,3,4,5]   
              }
from sklearn.model_selection import GridSearchCV, KFold
model = RandomForestClassifier() 
gcv = GridSearchCV(model, param_grid=param_grid, cv=kf,n_jobs=4,scoring='neg_root_mean_squared_error')
gcv.fit(train_X1, train_y1)
best_param = gcv.best_params_





##merge data
data4=pd.read_csv('C:/Users/smsun/Desktop/data_outlier.csv')

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
data=pd.read_csv('C:/Users/smsun/Desktop/2학기/business analytics/team_project/final.csv')


x_input = ['dong','exclusive_use_area', 'year_of_completion','police#',
            'near_subway#', 'crime5_occur#','SNU_student#','park_area','park#',
            'total_store#','franchise_store#','normal_store#', 'addr_kr','apt','jibun','apartment_id',
            'transaction_real_price']

graph_list=[ 'exclusive_use_area', 'year_of_completion', 'police#',
       'near_subway#', 'crime5_occur#', 'SNU_student#', 'park_area', 'park#',
       'total_store#', 'franchise_store#', 'normal_store#']
data = data[x_input]

data = data[data['transaction_real_price']<126000]

data.to_csv("C:/Users/smsun/Desktop/a.csv")

data = pd.read_csv("C:/Users/smsun/Desktop/a.csv")


import numpy as np

delete_list=[]
for i in graph_list:
    data_X = data[[i]]
    lof=LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = lof.fit_predict(data_X)
    delete_list.append(np.where(y_pred==-1))


delete_list2 = []
for i in delete_list:
    for j in range(0,len(i)):
        delete_list2.extend((list(i)[j]))        
        
delete_list2 = list(set(delete_list2))
new_data = data.drop(delete_list2, axis=0)

new_data.to_csv("C:/Users/smsun/Desktop/data.csv")


from sklearn.preprocessing import LabelEncoder
label_column1 = train_X[['gu']]
label_column2 = train_X[['dong']]

encoder1=LabelEncoder()
encoder1.fit(label_column1)
labels=encoder1.transform(label_column1)

encoder2=LabelEncoder()
encoder2.fit(label_column2)
labels2=encoder2.transform(label_column2)

labels = pd.DataFrame(labels, columns=['gu'])
labels2 = pd.DataFrame(labels2, columns=['dong'])


train_X = train_X.drop(['gu','dong'], axis=1)
train_X = pd.concat([labels,train_X], axis=1)
train_X = pd.concat([labels2,train_X], axis=1)

check = new_data['apt'].value_counts()


##
import pandas as pd
compare = pd.read_csv("C:/Users/smsun/Desktop/compare2.csv")
a=compare.groupby('dong').mean()
a['percent'] = a['difference'] / a['transaction_real_price'] 

compare['percent'] = compare['difference'] / compare['transaction_real_price']


a['percent'].mean()


##아 시발발바ㅣㅏ시바지닺비ㅏ딪바딪받ㅈ dt 다시그리기

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:/Users/smsun/Desktop/data_outlier.csv")
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', 'dong', 'gu'], axis=1)

from sklearn.tree import DecisionTreeClassifier      
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for train, test in kf.split(data):
    print(train, test)


#6130

data_X = data.drop(['transaction_real_price'], axis=1)
data_y = data['transaction_real_price']

train_X = data_X.iloc[train]
test_X = data_X.iloc[test]
train_y = data_y.iloc[train]
test_y = data_y.iloc[test]

clf=DecisionTreeClassifier()
clf.fit(train_X, train_y)
y_predict = clf.predict(test_X)
rms = math.sqrt(mean_squared_error(test_y,y_predict))

##select  best input by loop...
a = clf.feature_importances_
feature_df = pd.DataFrame({"column" : train_X.columns, "value" : a}) 
feature_df = feature_df.sort_values('value')


train_XX = data_X.iloc[train]
test_XX = data_X.iloc[test]
break_column = ""


min_rms = rms

for i in feature_df['column']:
    train_XX=train_XX.drop([i], axis=1)
    test_XX=test_XX.drop([i], axis=1)
    clf=DecisionTreeClassifier()
    clf.fit(train_XX,train_y)
    yy_predict = clf.predict(test_XX)
    rms_loop = math.sqrt(mean_squared_error(test_y,yy_predict))
    if rms_loop < min_rms:
        min_rms=rms_loop
        break_column=i
    if len(train_XX.columns)==1:
        break
    

def plot_feature_importances(model,train):
    n_feature=train.shape[1]
    plt.barh(range(n_feature),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_feature),train.columns)
    plt.xlabel("feature_importances")
    plt.ylim(-1, n_feature)
    plt.show()

plot_feature_importances(clf, train_X)



input_list = ['floor', 'exclusive_use_area', 'year_of_completion', 'near_subway#',
              'over65#', 'square_meters', 'popularity_density', 'franchise_store#',
              'park_area', 'normal_store#', 'crime5_occur#', 'park#', 'total_store#']

#train_X3 = train_X[input_list]
#test_X3 = test_X[input_list]
#clf2 = DecisionTreeClassifier()
#clf2.fit(train_X3, train_y)
#y_predict2 = clf2.predict(test_X3)
#rms = math.sqrt(mean_squared_error(test_y, y_predict2))


train_X2=train_X[input_list]
test_X2=test_X[input_list]

##select best parameter
kf = KFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {
              'criterion':['entropy','gini'],
              'random_state':[10,20,30,50,100,200,300],
              'max_depth':[None,5,10,20,30,40],
              'min_samples_split':[2,3,4,5],
              'max_leaf_nodes':[None,2,5,10]
              }
model = DecisionTreeClassifier() 
gcv = GridSearchCV(model, param_grid=param_grid, cv=kf,n_jobs=4,scoring='neg_root_mean_squared_error')
gcv.fit(train_X2, train_y)
best_param = gcv.best_params_



clf = DecisionTreeClassifier()
clf.fit(train_X2, train_y)
y_predict = clf.predict(test_X2)
rms = math.sqrt(mean_squared_error(test_y,y_predict))




##random forest  train_X 는 그냥 원본 트레인 셋

clf3=RandomForestClassifier()
clf3.fit(train_X,train_y)
y_predict3 = clf3.predict(test_X)
rms3 = math.sqrt(mean_squared_error(test_y, y_predict3))



##select  best input in randomforest by loop... in RandomForest
a = clf3.feature_importances_
feature_df = pd.DataFrame({"column" : train_X.columns, "value" : a}) 
feature_df = feature_df.sort_values('value')
plot_feature_importances(clf3, train_X)


train_XX = data_X.iloc[train]
test_XX = data_X.iloc[test]
break_column = ""
min_rms = rms3

for i in feature_df['column']:
    train_XX=train_XX.drop([i], axis=1)
    test_XX=test_XX.drop([i], axis=1)
    clf3=RandomForestClassifier()
    clf3.fit(train_XX,train_y)
    yy_predict = clf3.predict(test_XX)
    rms_loop = math.sqrt(mean_squared_error(test_y,yy_predict))
    if rms_loop < min_rms:
        min_rms=rms_loop
        break_column=i
    if len(train_XX.columns)==1:
        break


input_list = ['floor', 'exclusive_use_area', 'year_of_completion', 'near_subway#',
              'over65#', 'square_meters', 'popularity_density', 'franchise_store#',
              'park_area', 'normal_store#', 'crime5_occur#', 'park#', 'total_store#']


train_X3 = train_X[input_list]
test_X3 = test_X[input_list]

clf=RandomForestClassifier(criterion='entropy', max_depth=20, random_state=50)
clf.fit(train_X3, train_y)
y_predict=clf.predict(test_X3)
rms = math.sqrt(mean_squared_error(test_y,y_predict))


#find best parameter
kf = KFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {'criterion':['entropy','gini'],
              'random_state':[10,50,100,200],
              'max_depth':[None,10,20,30], 
              }
model = RandomForestClassifier() 
gcv = GridSearchCV(model, param_grid=param_grid, cv=kf,n_jobs=4,scoring='neg_root_mean_squared_error')
gcv.fit(train_X3, train_y)
best_param = gcv.best_params_



import pandas as pd
import math
new_data = pd.read_csv("C:/Users/smsun/Desktop/compare.csv")
origin_data = pd.read_csv("C:/Users/smsun/Desktop/data.csv")


data['error_rate'] = data['difference'] / data['transaction_real_price']
error_data_over = data[data['error_rate']>=0.15]
error_data_under = data[data['error_rate']<=-0.15]
a=error_data_under.groupby('dong').count()
data2 = data.groupby(['dong']).mean()
data2['error_rate'] = data2['difference']/data2['transaction_real_price']
data2 = data2.sort_values('error_rate')


##편차큰 지도 찍어봤는데 서로 다른 위치이고 이걸 보면서 ㅋㅋㄹㅃㅃ
new_data_sort = new_data.sort_values('ratio_Abs', ascending=False)
high_error_list = new_data_sort['addr_kr'][0:30]
high_error_lag_long = [(37.493837,126.9238265),(37.493837,126.9238265),(37.55338,126.8649062),
                      (37.5509599,127.0030196),(37.4936699,126.9336628),(37.4936647,126.900832),
                      (37.5393977,127.1267621),(37.5823099,126.9316458),(37.5822892,126.8637944),
                      (37.6585835,127.0496103),(37.5496602,126.933037),(37.4900725,127.0092242),
                      (37.5526194,126.9535334),(37.5038061,127.0712005),(37.6051677,126.9020925),
                      (37.5049678,127.0942053),(37.5390674,127.0549961),(37.5390674,127.0549961),
                      (37.5451182,126.834073),(37.4929026,127.0369033),(37.5140112,126.8641898),
                      (37.5408727,127.0559682),(37.5006547,127.1224341),(37.608365,127.0940463),
                      (37.6211547,126.9097727),(37.5186904,126.8339763),(37.5948092,126.9064139),
                      (37.6394858,126.9167997),(37.4873422,126.8871091),(37.5184662,127.0358582)]

##draw map
import folium

map_df = pd.DataFrame(data=high_error_list)
map_df['lag_long'] = high_error_lag_long

center = [37.541, 126.986]
m=folium.Map(location=center, zoom_start=10)

for i in map_df.index:
    lag = map_df['lag_long'][i][0]
    long = map_df['lag_long'][i][1]
    apt = map_df['addr_kr'][i]
    folium.Marker([lag,long], tooltip=apt).add_to(m)

m.save('map.html')


new_data_drop = new_data.drop(['Unnamed: 0', 'difference', 'ratio_Abs'],axis=1)


check_data = new_data[(new_data['dong']=="신대방동") & (new_data['exclusive_use_area']>75)]
check_data_2 = check_data[['dong','addr_kr','exclusive_use_area','year_of_completion','transaction_real_price','predict','ratio']]

check_data_yeom = new_data[(new_data['dong']=="염창동")]
check_data_sindang = new_data[(new_data['dong']=="신당동")]
check_data_sangdo = new_data[(new_data['dong']=="상도동")]
check_data_sinjeong = new_data[(new_data['dong']=="신정동")]
check_data_floor = new_data[new_data['floor']==-1]
check_data_cheon = new_data[new_data['dong']=='천호동']
check_data_hong = new_data[new_data['dong']=='홍은동']
origin_data_floor = origin_data[origin_data['floor']==-1]


check_data_old = new_data[new_data['year_of_completion']<=1990]
check_data_sam = new_data[new_data['dong']=='삼성동']


