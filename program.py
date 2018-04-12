# -*- coding: utf-8 -*-
"""
@author: Rohan Anish Shah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz


from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV

#read from the csv file and return a Pandas DataFrame.
wine= pd.read_csv('wine.csv')

# "quality" is the class attribute we are predicting. 
class_column = 'quality'

 
#After evaluating the data, the results demonstrate they all data is useful for classification and thus
#include them as features.
 
feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', \
                   'chlorides', 'free sulfur dioxide', 'residual sugar', \
                   'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

'''
Scaling data
'''
scaler = MinMaxScaler
for row in feature_columns:
    b = wine[row].max()
    wine[row] /= b
    
#Pandas DataFrame allows you to select columns. 
#We use column selection to split the data into features and class. 
wine_feature = wine[feature_columns]
wine_class = wine[class_column]

#Using the 75% of the data for training and the rest for testing
train_feature, test_feature, train_class, test_class = \
                         train_test_split(wine_feature, wine_class, stratify=wine_class, \
                         train_size=0.75, test_size=0.25)

training_accuracy = []
test_accuracy = []

#Decision Tree Classification using GridSearchCV object - dtc
tree = DecisionTreeClassifier()

param_dist = {"max_depth": [3, None],
              "criterion": ["gini", "entropy"]}

dtc = GridSearchCV(tree, param_dist, cv=5)
dtc.fit(train_feature, train_class)
prediction = dtc.predict(test_feature)
print("Training set score: {:.3f}".format(dtc.score(train_feature, train_class)))
test_score = dtc.score(test_feature, test_class)
print("Test set score: {:.3f}".format(dtc.score(test_feature, test_class)))
    
#Using scikit's function for producing confusion matrix
#prediction = dtc.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))


# Performing K-fold stratified cross-validation for classification
skf = StratifiedKFold(n_splits=10)
X = np.array(wine_feature)
y = np.array(wine_class)

accuracy = []

for train_index, test_index in skf.split(X, y):
    y_train = y[train_index]
    y_test = y[test_index]
    X_train = X[train_index]
    X_test = X[test_index]
    dtc.fit(X_train, y_train)
    
    temp = dtc.score(X_test, y_test)
    accuracy.append(temp)

answer = np.array(accuracy)
print("Cross-validation scores: {}".format(answer))
print("Average cross-validation score: {:.2f}".format(answer.mean()))
print("improved accuracy is: {:.2f}".format(test_score-answer.mean()))
    
    
