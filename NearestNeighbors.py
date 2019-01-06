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

inter = {1: [0,0.03], 2: [0.03,0.06], 3: [0.06,0.09], 4: [0.09, 0.12], 5: [0.12,0.15], 6: [0.15,0.18], 7: [0.18,0.21], 8: [0.21,0.24],
        9: [0.24,0.27], 10: [0.27,0.30], 11: [0.30, 0.34], 12: [0.34, 0.38], 13: [0.38, 0.42], 14: [0.42, 0.46], 15: [0.46, 0.50],
        16: [0.50, 0.55], 17: [0.55, 0.60], 18: [0.60, 0.65], 19: [0.65, 0.7], 20: [0.7, 0.8], 21: [0.8, 0.9], 22: [0.9, 1.0]}

def match(x):
    for i in inter :
        if inter.get(i)[0] < x and x <= inter.get(i)[1] :
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
            if y_predicted[i] == y_test[i] or y_predicted[i] == y_test[i]+1 or y_predicted[i] == y_test[i]-1:
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

#x=[]
#points=[]
#for k in range (5,15):
#    rate = 0 
#    for j in range (10):
#        rate += crossValidation(macronData,nearestNeighbors,5,k)
#    points.append(rate/5)
#    x.append(k)
#    print(k)
#plt.plot(x,points)
#plt.show()

## OPTIMAL NUMBER OF NEIGHBORS : 3

print(crossValidation(lepenData,nearestNeighbors,5,11))
print(crossValidation(macronData,nearestNeighbors,5,11))
print(crossValidation(melenchonData,nearestNeighbors,5,11))
print(crossValidation(fillonData,nearestNeighbors,5,11))
print(crossValidation(hamonData,nearestNeighbors,5,11))