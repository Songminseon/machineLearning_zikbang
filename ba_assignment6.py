# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

elect=pd.read_csv('https://drive.google.com/uc?export=download&id=1fq9qDqXLiUm0un_saxAUpPsSJa05F_bV', index_col=0)
county = pd.read_csv('https://drive.google.com/uc?export=download&id=1LciKFXkb3MmpXFEHDk1Db8YFsK0liF3a')
county_dict= pd.read_csv('C:/Users/smsun/Desktop/county_facts_dictionary.csv')
##merge data
data = elect.merge(county, left_on="FIPS", right_on="fips", how="left") #left merge

##check data trends
data['target']=(data['votes_dem_2016']>data['votes_gop_2016'])*1 #if democracy win set 1
plt.bar(data['target'].value_counts().index, data['target'].value_counts().values, width=0.1) #republic wins


##solve Alaska issue
data = elect[elect['county_name']!='Alaska'].merge(
    county, left_on="FIPS", right_on="fips", how="left")

data_ak = elect[elect['county_name']=='Alaska'].drop_duplicates(
    ['votes_dem_2016','votes_gop_2016'])  ##intergrate same value rows, as a result only 1 row...
data_ak['FIPS']=2000
data_ak = data_ak.merge(county,left_on='FIPS', right_on="fips", how="left")

data=pd.concat((data,data_ak), axis=0).sort_values('fips')

data['target']=(data['votes_dem_2016']>data['votes_gop_2016'])*1 #if democracy win set 1

##delete no relevant data => FIPS are duplicated, 2012 vote values are used for
## train, string data like state, county name is not used
## delete target after checking correlation
drop_list = ['total_votes_2012', 'votes_dem_2012', 'votes_gop_2012',
             'per_dem_2012', 'per_gop_2012', 'diff_2012', 'per_point_diff_2012',
             'state_abbr', 'county_name','FIPS','fips', 'area_name', 'state_abbreviation']   

#vote result should be used only in target, not for regression.
drop_list2 = ['per_dem_2016','per_gop_2016','votes_dem_2016','votes_gop_2016'
              ,'diff_2016','per_point_diff_2016']
 
#drop the unnecessary input varialbes
data=data.drop(drop_list, axis=1)
data=data.drop(drop_list2, axis=1)

##select input variables
corr = data.corr()
data_corr=corr['target']
data_corr = corr['target'].sort_values()
data_corr_index = corr['target'].sort_index() #sort by alphabet
data_corr_sort = data_corr.sort_values(ascending=False) #check by descending for abs minus value
data_corr_sort2 = data_corr.sort_values(ascending=True) #check by ascending


data_corr2 = np.array(data.corr()['target']) 
data_corr2 = np.sort(np.abs(data_corr2),axis = 0 ) #check absoulte value


##bring variable index by high correlation
index1 = list(data_corr_sort.index[1:4]) 
index2 = list(data_corr_sort2.index[0:3])
varlist = index1+index2



#Age : I transform data who can participate election
youth_column = data['AGE295214'] #data under 18
old_column = data['AGE775214'] 
data['age_young'] = 100 - youth_column - old_column #new input variables
data['age_not'] = youth_column + old_column


##bring variables by different types of data
#Edu : check distribution 
plt.hist(data['EDU635213'].values) #highschool dgree
plt.hist(data['EDU685213'].values) #bachelor degree
data['EDU635213'].std()
data['EDU685213'].std() #more high standard deviation
corr['EDU635213']['EDU685213'] #correlation between high school degree and bachelor degree

#Retail information
corr['RTN130207']['RTN131207']
plt.hist(data['RTN130207'].values, bins=10) #Retail sales => biased data
plt.hist(data['RTN131207'].values, bins=50) #Retail sales per capita


plt.plot(data['RTN130207'].index, data['RTN130207'].values) 
plt.plot(data['RTN130207'].index, data['RTN131207'].values)

#Income : select one
corr['INC110213']['INC910213'] #high corrleation between two variables, so i cut one of them
plt.hist(data['INC110213'], bins=50)
plt.hist(data['INC910213'], bins=50)
data['INC110213'].std() #more distributed
data['INC910213'].std()


#Racial information
white_column = data['RHI125214']
data['not_white'] = 100 - white_column  #set the ratio of 'white people'

varlist3 = ['EDU685213','INC110213','age_young', 'RTN131207','not_white']
varlist4 = ['EDU685213','INC110213','AGE775214', 'RTN131207','not_white']



## use Logistic Regression
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression()

## use 5-fold cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1)
for train,test in kf.split(data):
    print(train,test)
    
train_X = data.iloc[train][varlist]
train_y = data.iloc[train]['target']
test_X = data.iloc[test][varlist]
test_y = data.iloc[test]['target']

clf1.fit(train_X, train_y)
clf1.score(test_X, test_y)

y_true = np.array(test_y)
y_predict = clf1.predict(test_X)

##calculate accuracy, recall,precision, F1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy_score(y_true, y_predict)
precision_score(y_true, y_predict)
recall_score(y_true, y_predict)
f1_score(y_true, y_predict)

from sklearn.metrics import roc_curve, roc_auc_score
y_prob=clf1.predict_proba(test_X)
fpr, tpr, thresholds = roc_curve(test_y, y_prob[:,1], pos_label=1)    
xx=np.linspace(0,1,10)
plt.plot(fpr,tpr)
plt.plot(xx,xx,'k--')
roc_auc_score(test_y, y_prob[:,1])



#divied by income over average and under average
data_over = data[data['INC110213']>=data['INC110213'].mean()]
data_under = data[data['INC110213']<data['INC110213'].mean()]



#Rgression  by income
clf2 = LogisticRegression()
clf3 = LogisticRegression()

kf = KFold(n_splits=5, shuffle=True, random_state=1)
for train2,test2 in kf.split(data_over):
    print(train,test)

for train3,test3 in kf.split(data_under):
    print(train,test)

        
#over data
train_X2 = data_over.iloc[train2][varlist3]
test_X2 = data_over.iloc[test2][varlist3]
train_y2 = data_over.iloc[train2]['target']
test_y2 = data_over.iloc[test2]['target']
clf2.fit(train_X2, train_y2)
clf2.score(test_X2, test_y2)
y_true2 = np.array(test_y2)
y_predict2 = clf2.predict(test_X2)
accuracy_score(y_true2, y_predict2)
precision_score(y_true2, y_predict2)
recall_score(y_true2, y_predict2)
f1_score(y_true2, y_predict2)

from sklearn.metrics import roc_curve, roc_auc_score
y_prob2=clf2.predict_proba(test_X2)
fpr, tpr, thresholds = roc_curve(test_y2, y_prob2[:,1], pos_label=1)    
xx=np.linspace(0,1,10)
plt.plot(fpr,tpr)
plt.plot(xx,xx,'k--')
roc_auc_score(test_y2, y_prob2[:,1])




train_X3 = data_under.iloc[train3][varlist3]
test_X3 = data_under.iloc[test3][varlist3]
train_y3 = data_under.iloc[train3]['target']
test_y3 = data_under.iloc[test3]['target']
clf3.fit(train_X3, train_y3)
clf3.score(test_X3, test_y3)
y_true3 = np.array(test_y3)
y_predict3 = clf3.predict(test_X3)
accuracy_score(y_true3, y_predict3)
precision_score(y_true3, y_predict3)
recall_score(y_true3, y_predict3)
f1_score(y_true3, y_predict3)

from sklearn.metrics import roc_curve, roc_auc_score
y_prob3=clf3.predict_proba(test_X3)
fpr, tpr, thresholds = roc_curve(test_y3, y_prob3[:,1], pos_label=1)    
xx=np.linspace(0,1,10)
plt.plot(fpr,tpr)
plt.plot(xx,xx,'k--')
roc_auc_score(test_y3, y_prob3[:,1])
    