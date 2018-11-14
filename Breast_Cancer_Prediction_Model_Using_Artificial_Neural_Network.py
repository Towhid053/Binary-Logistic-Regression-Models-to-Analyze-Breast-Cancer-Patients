# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:19:28 2018

@author: Safayet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('BreastCancer.csv')
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 10].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer):
    
    classifier = Sequential()
    
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9 ))
    classifier.add(Dropout(0.125))
    
    classifier.add(Dense(units = 16, kernel_initializer= 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.125))
    
    classifier.add(Dense(units = 16, kernel_initializer= 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.125))

    classifier.add(Dense(units = 16, kernel_initializer= 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.125))
    
    classifier.add(Dense(units = 16, kernel_initializer= 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.125))
    
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [8, 12],
              'epochs': [15, 20],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)


grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


y_pred = grid_search.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

