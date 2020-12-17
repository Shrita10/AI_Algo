# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:36:44 2020

@author: Shrita
"""
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle

df = pd.read_csv('C:\\Users\\Shrita\\Desktop\\AI_Algo\\startup.csv')

a = 'California'
b = 'New York'
c= 'Texas'
d = 'Mississippi'
e = 'Others'
f = 0
df['Place_Company'] = np.where(df['is_CA']== 1, a, (np.where(df['is_NY']== 1, b, (np.where(df['is_TX']== 1, c, (np.where(df['is_MA']== 1, d, (np.where(df['is_otherstate']== 1, e,f)))))))))

df = df[df['Place_Company'] != '0']

colsw = ['Unnamed: 0', 'state_code','latitude', 'longitude', 'zip_code', 'id','city', 'Unnamed: 6', 'name','state_code.1', 'is_CA', 'is_NY','is_MA', 'is_TX', 'is_otherstate','is_software','is_web', 'is_mobile', 'is_enterprise', 'is_advertising','is_gamesvideo', 'is_ecommerce', 'is_biotech', 'is_consulting','is_othercategory', 'object_id','closed_at']
df.drop(colsw,axis='columns', inplace=True)

df = df.rename(columns = {'has_VC': 'Venture Capital?', 'has_angel': 'Angel Investor?','has_roundA':'Round A?','has_roundB':'Round B?','has_roundC':'Round C?','has_roundD':'Round D?'}, inplace = False)

df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(df['age_first_milestone_year'].mean())
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(df['age_last_milestone_year'].mean())

df['founded_at'] = pd.to_datetime(df['founded_at'])
df['first_funding_at'] = pd.to_datetime(df['first_funding_at'])
df['last_funding_at'] = pd.to_datetime(df['last_funding_at'])
df['founded_at_year'] = df['founded_at'].dt.year
df['first_funding_at_year'] = df['first_funding_at'].dt.year
df['last_funding_at_year'] = df['last_funding_at'].dt.year

colsw = ['founded_at','first_funding_at','last_funding_at']
df.drop(colsw,axis='columns', inplace=True)

df = df[df['relationships'] < 50]
df = df[df['avg_participants'] < 11]

categorical = ['category_code','status','Place_Company']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        
pickle.dump(le, open('le.pkl','wb'))
        
X = df.drop(['status'], axis=1)
y = df['status']    
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(23).plot(kind='barh')
plt.show()
    
colsw = ['Angel Investor?','Venture Capital?','Round C?', 'Round D?'
         ,'last_funding_at_year','first_funding_at_year','age_last_milestone_year'
         ,'Place_Company','category_code','funding_total_usd','Round A?'
         ,'age_last_funding_year','founded_at_year','funding_rounds']
df.drop(colsw,axis='columns', inplace=True)

X = df.drop(['status'], axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
pickle.dump(sc, open('sc.pkl','wb'))

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))

X=StandardScaler().fit_transform(X)
pca=PCA(n_components=4)
X_pca=pca.fit_transform(X)
exp_var=pca.explained_variance_ratio_
cumsum_var=np.cumsum(exp_var)
cumsum_var
plt.plot(cumsum_var)
plt.grid()

pickle.dump(pca, open('pca.pkl','wb'))

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.3, random_state = 0)
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

param_grid = {'C':[0.01,0.1,1,10,100,1000],'gamma':[10,1,0.1,0.001,0.0001,0.00001], 'kernel':['linear','rbf']}
grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)
grid.fit(X_train,y_train)
print(grid.best_params_)

classifier = SVC(C= 10,kernel = 'rbf', gamma = 0.1,random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv =10)
print(accuracies.mean()*100)
print(accuracies.std()*100)

pickle.dump(classifier, open('model.pkl','wb'))


    
    
    
    
    
    
    
    
    
    
    