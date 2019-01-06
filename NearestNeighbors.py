#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier 

# import Data

fillonData = pd.read_csv('fillonData.csv', encoding='latin-1')
hamonData = pd.read_csv('hamonData.csv', encoding='latin-1')
lepenData = pd.read_csv('lepenData.csv', encoding='latin-1')
macronData = pd.read_csv('macronData.csv', encoding='latin-1')
melenchonData = pd.read_csv('melenchonData.csv', encoding='latin-1')
cantonsData = pd.read_csv('cantonsData.csv', encoding='latin-1')

# create intervals for classification

inter = {1: [0.0, 0.015], 2: [0.015, 0.03], 3: [0.03, 0.045], 4: [0.045, 0.06], 5: [0.06, 0.075], 6: [0.075, 0.09], 7: [0.09, 0.105], 8: [0.105, 0.12],
        9: [0.12, 0.135], 10: [0.135, 0.15], 11: [0.15, 0.165], 12: [0.165, 0.18], 13: [0.18, 0.195], 14: [0.195, 0.21], 15: [0.21, 0.225], 16: [0.225, 0.24],
        17: [0.24, 0.255], 18: [0.255, 0.27], 19: [0.27, 0.285], 20: [0.285, 0.3], 21: [0.3, 0.34], 22: [0.34, 0.38], 23: [0.38, 0.42], 24: [0.42, 0.46],
        25: [0.46, 0.5], 26: [0.5, 0.55], 27: [0.55, 0.6], 28: [0.6, 0.65], 29: [0.65, 0.7], 30: [0.7, 0.8], 31: [0.8, 0.9], 32: [0.9, 1.0]}

inter2 = {1: [0,0.03], 2: [0.03,0.06], 3: [0.06,0.09], 4: [0.09, 0.12], 5: [0.12,0.15], 6: [0.15,0.18], 7: [0.18,0.21], 8: [0.21,0.24],
        9: [0.24,0.27], 10: [0.27,0.30], 11: [0.30, 0.34], 12: [0.34, 0.38], 13: [0.38, 0.42], 14: [0.42, 0.46], 15: [0.46, 0.50],
        16: [0.50, 0.55], 17: [0.55, 0.60], 18: [0.60, 0.65], 19: [0.65, 0.7], 20: [0.7, 0.8], 21: [0.8, 0.9], 22: [0.9, 1.0]}

def match(x):
    for i in inter2 :
        if inter2.get(i)[0] < x and x <= inter2.get(i)[1] :
            return i
    return (0)

# nearest neighbors

def nearestNeighbors(X_train, y_train, X_test,nb_neighbors):
    classifier = KNeighborsClassifier(n_neighbors=nb_neighbors)  
    classifier.fit(X_train, y_train)
    result = classifier.predict(X_test)
    return(result)

def crossValidation(data,algo,nb_folds,nb_neighbors):

    list_var=data.columns.drop(['% Voix/Exp','Département','Code Canton'])
    X = np.array(data[list_var])
    y = np.array(data['% Voix/Exp'])
    for k in range(y.size):
        y[k] = match(y[k])
    
    kf = KFold(n_splits=nb_folds, shuffle=True)
    
    totalInstances = 0
    totalCorrect = 0
    totalGap = 0
    
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        y_predicted = algo(X_train, y_train, X_test,nb_neighbors)
        
        gap = 0
        correct = 0	
        for i in range(y_test.size):
            if y_predicted[i] == y_test[i] :
                correct += 1
            gap += np.abs(y_predicted[i] - y_test[i])
            
        # print ('CORRECT :',str(correct/y_test.size))
        # print ('GAP :',str(gap/y_test.size))
        totalCorrect += correct
        totalGap += gap
        totalInstances += y_test.size
    # print('TOTAL CORRECT : ',str(totalCorrect/totalInstances))
    # print ('TOTAL GAP : ',str(totalGap/totalInstances))
    return(totalGap/totalInstances)

points=[]
for k in range (1,2):
    rate = 0 
    for j in range (5):
        rate += crossValidation(fillonData,nearestNeighbors,5,k)
    points.append(rate/5)
    print(k)
plt.plot(points)
plt.show()

## OPTIMAL NUMBER OF NEIGHBORS : 3 ?