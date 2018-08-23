# -*- coding: utf-8 -*-
"""
About: Tumor Prediction project

Created on Fri Aug 10 16:01:00 2018

Last update Date  Aug 23 16:01:00 2018

@author: Mounir ZAGHBA
"""
# 1- import lbraries
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 2- load data set
dataset = pd.read_csv('data.csv')
X=dataset.iloc[:,2:-1].values
y=dataset.iloc[:,1].values


# 3- encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y)
# 4- Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# 4- fit data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#5- Split dataset to training and testing dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# 6- Train models
print("************************************************************")
print("***********  LOGISTIC REGRESSION  **************************")
print("************************************************************")
#-6-1- Logistic regression
from sklearn.linear_model import LogisticRegression
logisticRegressor = LogisticRegression()
logisticRegressor.fit(X_train,y_train)
y_pred_LR=logisticRegressor.predict(X_test)


cm_LR=confusion_matrix(y_test,y_pred_LR)
accuracyRate_LR=(cm_LR[0,0]+cm_LR[1,1])/cm_LR.sum()
print('accuracy rate for logistic regression is : ',accuracyRate_LR)#0.9649122807017544
print("************************************************************")
print("***********     RANDOM FOREST     **************************")
print("************************************************************")
#-6-2- Random Forest
from sklearn.ensemble import RandomForestClassifier
randomForestClassifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
randomForestClassifier.fit(X_train,y_train)
y_pred_RF=randomForestClassifier.predict(X_test)
cm_RF=confusion_matrix(y_test,y_pred_RF)
accuracyRate_RF=(cm_RF[0,0]+cm_RF[1,1])/cm_RF.sum()#0.9736842105263158
print('accuracy rate for Random Forest  is : ',accuracyRate_RF)
