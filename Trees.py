# -*- coding: utf-8 -*-
# env python2.7
# https://www.python.org/dev/peps/pep-0008#introduction<BR>
# http://scikit-learn.org/<BR>
# http://pandas.pydata.org/<BR>
"""
Created on Sat Jan  5 01:28:23 2019

@author: Vince
"""
import numpy as np
import pandas as pd
import pylab as plt
from time import sleep
from IPython import display
import matplotlib.pyplot as plt


### Fetch the data and load it in pandas =====================================


# See data (five rows) using pandas tools ====================================
#print data.head(2)

### Prepare input to scikit and train and test cut




def f(n):
    data = pd.read_csv('fillonData.csv', encoding='latin-1')
    print ("Size of the data: ", data.shape)
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
    return(accuracy)

N=[0.1,0.12,0.15,0.18,0.2,0.3,0.4]
F=[]
for n in N:
    x=f(n)
    F=F+[x]
plt.plot(N,F)
plt.show()




