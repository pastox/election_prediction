#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# import Data

fillonData = pd.read_csv('fillonData.csv', encoding='latin-1')
hamonData = pd.read_csv('hamonData.csv', encoding='latin-1')
lepenData = pd.read_csv('lepenData.csv', encoding='latin-1')
macronData = pd.read_csv('macronData.csv', encoding='latin-1')
melenchonData = pd.read_csv('melenchonData.csv', encoding='latin-1')
cantonsData = pd.read_csv('cantonsData.csv', encoding='latin-1')

# =============================================================================
# 
# # create a dictionary -> key : department, attribute : number of cantons)
# 
# listDepartements = list(cantonsData['Code du département'])
# listCantons = list(cantonsData['Code du canton'])
# 
# dicoCanton = {}
# i = 0
# departement = listDepartements[0]
# canton = 0
# while i < len(listDepartements):
#     if listDepartements[i]==departement:
#         canton += 1
#     else :
#         dicoCanton[departement] = canton
#         departement = listDepartements[i]
#         canton = listCantons[i]
#     i += 1
# dicoCanton[departement]=canton
# 
# 
# print(dicoCanton)
# # {'1': 23, '2': 21, '3': 19, '4': 15, '5': 15, '6': 27, '7': 17, '8': 19, '9': 13, '10': 17, '11': 19, '12': 23, '13': 29, '14': 25, '15': 15, '16': 19, '17': 27, '18': 19, '19': 19, '2A': 11, '2B': 15, '21': 23, '22': 27, '23': 15, '24': 25, '25': 19, '26': 19, '27': 23, '28': 15, '29': 27, '30': 23, '31': 27, '32': 17, '33': 33, '34': 25, '35': 27, '36': 13, '37': 19, '38': 29, '39': 17, '40': 15, '41': 15, '42': 21, '43': 19, '44': 31, '45': 21, '46': 17, '47': 21, '48': 13, '49': 21, '50': 27, '51': 23, '52': 17, '53': 17, '54': 23, '55': 17, '56': 21, '57': 27, '58': 17, '59': 41, '60': 21, '61': 21, '62': 39, '63': 31, '64': 27, '65': 17, '66': 17, '67': 23, '68': 17, '69': 14, '70': 17, '71': 29, '72': 21, '73': 19, '74': 17, '75': 34, '76': 35, '77': 23, '78': 21, '79': 17, '80': 23, '81': 23, '82': 15, '83': 23, '84': 17, '85': 17, '86': 19, '87': 21, '88': 17, '89': 21, '90': 9, '91': 21, '92': 23, '93': 21, '94': 25, '95': 21, 'ZA': 21, 'ZB': 1, 'ZC': 1, 'ZD': 25, 'ZM': 13, 'ZN': 3, 'ZP': 6, 'ZS': 1, 'ZW': 1, 'ZX': 1, 'ZZ': 1}
#      
# =============================================================================



# linear regression algorithm

def linearRegression(X_train,y_train,X_test):
    reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
    return(reg.predict(X_test))

# =============================================================================
# 1 DIMENSION
# 
# plt.plot(fillonData['% Voix/Exp'], fillonData['Indice Evasion Client'], 'ro', markersize=1)
# plt.show()
# 
# X = np.matrix([np.ones(fillonData.shape[0]), fillonData['Indice Evasion Client'].as_matrix()]).T
# y = np.matrix(fillonData['% Voix/Exp']).T
# 
# theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# 
# plt.plot([0,1], [theta.item(0),theta.item(0) + theta.item(1)])
# =============================================================================



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
    totalCorrect = 0
    
    
    for train_index, test_index in kf.split(fillonData):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        y_predicted = algo(X_train, y_train, X_test)
        
        # If error percentage < 5%, we assume the prediction is satisfying
        correct = 0	
        for i in range(y_test.size):
            if np.abs(y_predicted[i]-y_test[i])/y_test[i] < 0.05 :
                correct += 1
        print ('CORRECT :',str(correct/y_test.size))
        totalCorrect += correct
        totalInstances += y_test.size
    print ('TOTAL CORRECT : ',str(totalCorrect/totalInstances))

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
