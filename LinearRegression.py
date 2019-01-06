#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# import Data

fillonData = pd.read_csv('fillonData.csv', encoding='latin-1')
hamonData = pd.read_csv('hamonData.csv', encoding='latin-1')
lepenData = pd.read_csv('lepenData.csv', encoding='latin-1')
macronData = pd.read_csv('macronData.csv', encoding='latin-1')
melenchonData = pd.read_csv('melenchonData.csv', encoding='latin-1')
cantonsData = pd.read_csv('cantonsData.csv', encoding='latin-1')


# linear regression algorithm

def linearRegression(X_train,y_train,X_test):
    reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    return(reg.predict(X_test))


# cross validation algorithm

def crossValidation(data,algo,nb_folds) :
    # X : add intercept term, delete id
    list_var=data.columns.drop(['% Voix/Exp','Département','Code Canton'])
    X = np.array(data[list_var])
    X_new = np.ones((X.shape[0],X.shape[1]+1))
    for i in range (X.shape[0]) :
        for j in range (X.shape[1]) :
                X_new[i,j+1] = X[i,j]
    X = X_new
    # y
    y = np.array(data['% Voix/Exp'])
    
    kf = KFold(n_splits=nb_folds, shuffle=True)
    
    totalInstances = 0
    totalGap = 0
    
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        y_predicted = algo(X_train, y_train, X_test)
        
        gap = 0
        for i in range(y_test.size):
            gap += np.abs(y_predicted[i]-y_test[i])/y_test[i]
        print ('GAP :',str(gap/y_test.size))
        totalInstances += y_test.size
        totalGap += gap
    print ('TOTAL GAP : ',str(totalGap/totalInstances))

###

print('\n Macron')
crossValidation(macronData,linearRegression,5)

print('\n Fillon')
crossValidation(fillonData,linearRegression,5)

print('\n Le Pen')
crossValidation(lepenData,linearRegression,5)

print('\n Hamon')
crossValidation(hamonData,linearRegression,5)

print('\n Mélenchon')
crossValidation(melenchonData,linearRegression,5)