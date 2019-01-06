# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 23:42:17 2019

@author: Vince
"""

import numpy as np
import pandas as pd
import pylab as plt
from time import sleep
from IPython import display
import matplotlib.pyplot as plt


### Fetch the data and load it in pandas =====================================
data = pd.read_csv('fillonData.csv', encoding='latin-1')
print ("Size of the data: ", data.shape)

# See data (five rows) using pandas tools ====================================
#print data.head(2)

### Prepare input to scikit and train and test cut




n=0.2
for i in range(data["% Voix/Exp"].shape[0]):
    if (data["% Voix/Exp"][i]>=n):
        data["% Voix/Exp"][i]=1
    else:
        data["% Voix/Exp"][i]=-1
X = data[data.columns.drop(['% Voix/Exp','Code Canton','Département'])]
y = data['% Voix/Exp'].values
print (np.unique(y))


# Import cross validation tools from scikit ==================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

### Train a single decision tree =============================================
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=8)

# Train the classifier and print training time
clf.fit(X_train, y_train)

# Do classification on the test dataset and print classification results =====
#from sklearn.metrics import classification_report
#target_names = data['% Voix/Exp'].unique().astype(str).sort()
y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=target_names))

# Compute accuracy of the classifier (correctly classified instances) ========
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy = ", accuracy)

D = 2 # tree depth
T = 10 # number of trees
w = np.ones(X_train.shape[0]) / X_train.shape[0]
training_scores = np.zeros(X_train.shape[0])
test_scores     = np.zeros(X_test.shape[0])

ts = np.arange(len(training_scores))
training_errors = []
test_errors = []

for t in range(T):
    clf = DecisionTreeClassifier(max_depth=D)
    clf.fit(X_train, y_train, sample_weight = w)
    y_pred = clf.predict(X_train)
    indicator = np.not_equal(y_pred, y_train)
    gamma = w[indicator].sum() / w.sum()
    alpha = np.log((1-gamma) / gamma)
    w *= np.exp(alpha * indicator) 
    
    training_scores += alpha * y_pred
    training_error = 1. * len(training_scores[training_scores * y_train < 0]) / len(X_train)
    y_test_pred = clf.predict(X_test)
    test_scores += alpha * y_test_pred
    test_error = 1. * len(test_scores[test_scores * y_test < 0]) / len(X_test)
    #print t, ": ", alpha, gamma, training_error, test_error
    
    training_errors.append(training_error)
    test_errors.append(test_error)
    
plt.plot(training_errors, label="training error")
plt.plot(test_errors, label="test error")
plt.legend()
plt.show()