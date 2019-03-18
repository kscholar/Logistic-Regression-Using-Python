################### Source code for Practicing logistic regression in Python ################### 

##### Importing the Libraries #####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#### Importing the Dataset ####
dataset =pd.read_csv('Social_Network_Ads.csv')
x= dataset.iloc[:, [2,3]].values
y= dataset.iloc[:, [4]].values

#### Splitting the Dataset into Training set and test set ####
from sklearn.cross_validation import train_test_split
x_train, x_test , y_train , y_test = train_test_split(x,y , test_size = 0.25, random_state=0)
 
#### Feature Scaling ####
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#### Fitting the logistic regression on training set ####
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

####Predicting the classifier on test set####
y_pred = classifier.predict(x_test)

####confusion matrix####
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm 
