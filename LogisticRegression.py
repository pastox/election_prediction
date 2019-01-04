#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op
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

# create intervals for classification

inter= {1: [0.0, 0.015], 2: [0.015, 0.03], 3: [0.03, 0.045], 4: [0.045, 0.06], 5: [0.06, 0.075], 6: [0.075, 0.09], 7: [0.09, 0.105], 8: [0.105, 0.12],
        9: [0.12, 0.135], 10: [0.135, 0.15], 11: [0.15, 0.165], 12: [0.165, 0.18], 13: [0.18, 0.195], 14: [0.195, 0.21], 15: [0.21, 0.225], 16: [0.225, 0.24],
        17: [0.24, 0.255], 18: [0.255, 0.27], 19: [0.27, 0.285], 20: [0.285, 0.3], 21: [0.3, 0.34], 22: [0.34, 0.38], 23: [0.38, 0.42], 24: [0.42, 0.46],
        25: [0.46, 0.5], 26: [0.5, 0.55], 27: [0.55, 0.6], 28: [0.6, 0.65], 29: [0.65, 0.7], 30: [0.7, 0.8], 31: [0.8, 0.9], 32: [0.9, 1.0]}

def match(x):
    for i in inter :
        if inter.get(i)[0] < x and x <= inter.get(i)[1] :
            return i
    return (0)


# logistic regression
    
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def computeCost(theta, X_train, y_train): 
	# Computes the cost of using theta as the parameter for logistic regression. 
	m = X_train.shape[0]
	J = 0
	for i in range(m):
		J += (-y_train[i] * np.log(sigmoid(np.dot(X_train[i,:],theta))) - (1 - y_train[i]) * np.log(1 - sigmoid(np.dot(X_train[i,:],theta))))	
	J /= m
	return J

def computeGrad(theta, X_train, y_train):
	# Computes the gradient of the cost with respect to the parameters.
	m = X_train.shape[0]
	grad = np.zeros(theta.size)
	for i in range(theta.shape[0]):
	    for j in range(m):
	        grad[i] += (sigmoid(np.dot(X_train[j,:],theta)) - y_train[j]) * X_train[j,i]		
	grad /= m
	return grad

def predict(theta, X):
	p = np.zeros(X.shape[0])
	for i in range(X.shape[0]):
		p[i] = sigmoid(np.dot(X[i,:],theta))
	return p

def addIntercept(X):
    X_new = np.ones((X.shape[0],X.shape[1]+1))
    for i in range (X.shape[0]) :
        for j in range (X.shape[1]) :
                X_new[i,j+1] = X[i,j]
    return(X_new)
    
def logisticRegression_v1(X_train,y_train,X_test):
    
    X_train = addIntercept(X_train)
    X_test = addIntercept(X_test)
    
    initial_theta = np.zeros((X_train.shape[1],1))
    
    Result = op.minimize(fun = computeCost, x0 = initial_theta, args = (X_train, y_train), method = 'TNC',jac = computeGrad);
    theta = Result.x;
    
    p = predict(np.array(theta), X_test)
    return (p)


def crossValidation(data,algo,nb_folds):

    list_var=data.columns.drop(['% Voix/Exp','Département','Code Canton'])
    X = np.array(data[list_var])
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
        
        correct = 0	
        for i in range(y_test.size):
            if match(y_predicted[i]) == match(y_test[i]) :
                correct += 1
           
        print ('CORRECT :',str(correct/y_test.size))
        totalCorrect += correct
        totalInstances += y_test.size
    print ('TOTAL CORRECT : ',str(totalCorrect/totalInstances))





#print('\n Macron')
#crossValidation(macronData,logisticRegression,5)
#
#print('\n Fillon')
#crossValidation(fillonData,logisticRegression,5)
#
#print('\n Le Pen')
#crossValidation(lepenData,logisticRegression,5)
#
#print('\n Hamon')
#crossValidation(hamonData,logisticRegression,5)
#
#print('\n Mélenchon')
#crossValidation(melenchonData,logisticRegression,5)
