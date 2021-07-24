#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 23:29:10 2021

@author: fubao
"""




# logistic regression

#reference: https://towardsdatascience.com/logistic-regression-from-scratch-69db4f587e17

#loss
# y = 1/(1+e^z) z = wx + b
#L = -ylog(y^) - (1-y)log(1-y^)         # y^ is the predicted value

#dL/dw = x * (y-y^)


from numpy import log, dot, e
from numpy.random import rand
from sklearn import datasets
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

class LogisticRegression:
    
    def __init__(self):
        self.weights = None
        self.loss = None
    
    def sigmoid(self, z):
        return 1/(1+e**(-z))
    
    def fit(self, X, y, epochs=20, lr=0.05):
        # input: X -- NXD ; y -- NX1
        loss = []
        
        weights = rand(X.shape[1])   # weight: D
        N = len(X)          # number of instances
        
        for _ in range(epochs):
            
            #Gradient descent
            y_hat = self.sigmoid(dot(X, weights))
            
            # update weight with GD  # with stochastic gradient descent
            weights -= lr * dot(X.T, y_hat -y)/ N
            
            # save the loss
            loss.append(self.cost_function(X, y, weights))
            
        self.weights = weights
        
        self.loss = loss
        
        
    def predict(self, X):
        # predict 
        # y = 1/(1+e^z) z = wx + b
        
        z = dot(X, self.weights)
        
        y = self.sigmoid(z)
        
        return [1 if v > 0.5 else 0 for v in y]
        
    
    def cost_function(self, X, y, weights): 
        # y is the ground truth
        
        z = dot(X, weights)
        l1 = y * log(self.sigmoid(z))
        l0 = (1-y) * log(1-self.sigmoid(z))
        
        loss = -sum(l1 + l0)/X.shape[0]
        
        return loss
    
    
    def test_example(self):
        # random data half circle
        X, y = datasets.make_moons(100, noise =0.3, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        LogisticRegressionObj = LogisticRegression()
        LogisticRegressionObj.fit(X_train, y_train)
        
        y_test_hat = LogisticRegressionObj.predict(X_test)
            
        plt.plot(LogisticRegressionObj.loss)
        plt.show()
        print("y: ", y)
        print("y_test_hat: ", y_test_hat)
        print("loss: ", LogisticRegressionObj.loss)
        
        prfs =  precision_recall_fscore_support(y_test, y_test_hat, average='macro')
        
        print("prfs: ", prfs)
    
    
if __name__ == '__main__':
    
    LogisticRegressionObj = LogisticRegression()    
    LogisticRegressionObj.test_example()
    
            