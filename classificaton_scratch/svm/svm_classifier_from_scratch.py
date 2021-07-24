#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:49:29 2020

@author: fubao
"""

# svm classiifer from scratch
# binary
#http://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/

# https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2

# https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

# https://dzone.com/articles/classification-from-scratch-svm-78

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

from data_process import read_file_features



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
 

# http://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/
class LinearSVMUsingSoftMargin:
    def __init__(self, C=1.0):
        self._support_vectors = None
        self.C = C
        self.beta = None
        self.b = None
        self.X = None
        self.y = None
 
        # n is the number of data points
        self.n = 0
 
        # d is the number of dimensions
        self.d = 0
 
    def __decision_function(self, X):
        # βT*x + b  #  wT * x + b
        return X.dot(self.beta) + self.b
 
    def __cost(self, margin):
        #L=||β2||2+C∑i=1 to n max(0,1−yi(βTxi+b))
        return (1 / 2) * self.beta.dot(self.beta) + self.C * np.sum(np.maximum(0, 1 - margin))
 
    def __margin(self, X, y):
        # yf(x) = y((βTx+b))
        return y * self.__decision_function(X)
 
    def fit(self, X, y, lr=1e-3, epochs=500):
        # Initialize Beta and b
        self.n, self.d = X.shape
        self.beta = np.random.randn(self.d)
        self.b = 0
 
        # Required only for plotting
        self.X = X
        self.y = y
 
        loss_array = []
        for _ in range(epochs):
            margin = self.__margin(X, y)
            loss = self.__cost(margin)
            loss_array.append(loss)
 
            misclassified_pts_idx = np.where(margin < 1)[0] # support vector
            
            d_beta = self.beta - self.C * y[misclassified_pts_idx].dot(X[misclassified_pts_idx])
            self.beta = self.beta - lr * d_beta
 
            d_b = - self.C * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b
 
        self._support_vectors = np.where(self.__margin(X, y) <= 1)[0]
 
    def predict(self, X):
        return np.sign(self.__decision_function(X))
 
    def score(self, X, y):
        P = self.predict(X)
        return np.mean(y == P)
 
    def plot_decision_boundary(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50, cmap=plt.cm.Paired, alpha=.7)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
 
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.__decision_function(xy).reshape(XX.shape)
 
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])
 
        # highlight the support vectors
        ax.scatter(self.X[:, 0][self._support_vectors], self.X[:, 1][self._support_vectors], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')
 
        plt.show()
 
 
def load_data(cols):
    iris = sns.load_dataset("iris")
    print("iris column ", iris.columns)
    iris = iris.tail(100)       # get last 100 rows for two classes, binary classification
 
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
 
    X = iris.drop(["species"], axis=1)
 
    if len(cols) > 0:
        X = X[cols]
 
    return X.values, y
 
 
if __name__ == '__main__':
    # make sure the targets are (-1, +1)
    # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width

    #cols = ["petal_length", "petal_width"]
    cols = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
    X, y = load_data(cols)
 
    y[y == 0] = -1
 
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # now we'll use our custom implementation
    model = LinearSVMUsingSoftMargin(C=15.0)
 
    model.fit(X, y)
    print("train score:", model.score(X, y))
 
    model.plot_decision_boundary()


"""
# https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
# consider two features only for classify to train to get w1 and w2 for svm        
class SVM_Iris_Two_Features_Only(object):
    
    def __init__(self):
        
        self.df = read_file_features()
        #pass
    
    def train_test(self):
    
        ## Drop rest of the features and extract the target values
        # only use two features for testing here
        self.df = self.df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
        
        # construct y label  -1, +1 class
        Y = []
        target = self.df['Species']
        for val in target:
            if(val == 'Iris-setosa'):
                Y.append(-1)
            else:
                Y.append(1)
            
        df_x = self.df.drop(['Species'],axis=1)
        X = df_x.values.tolist()
        
        ## Shuffle and split the data into training and test set
        X, Y = shuffle(X,Y)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)
        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        y_train = y_train.reshape(90,1)         # total number 90
        y_test = y_test.reshape(10,1)          # toal 10

        return x_train, x_test, y_train, y_test 
    
    def svm_scratch(self, x_train, x_test, y_train, y_test):
        # Support Vector Machine from scratch
    
        train_f1 = x_train[:,0]
        train_f2 = x_train[:,1]
        
        train_f1 = train_f1.reshape(90,1)
        train_f2 = train_f2.reshape(90,1)
        
        w1 = np.zeros((90,1))
        w2 = np.zeros((90,1))
        
        epochs = 1
        alpha = 0.0001
        
        while(epochs < 10000):
            y = w1 * train_f1 + w2 * train_f2
            prod = y * y_train
            if epochs % 1000 == 0:
                print("epochs: ", epochs)
            count = 0
            for val in prod:
                if(val >= 1):
                    cost = 0
                    w1 = w1 - alpha * (2 * 1/epochs * w1)
                    w2 = w2 - alpha * (2 * 1/epochs * w2)
                    
                else:
                    cost = 1 - val 
                    w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
                    w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1/epochs * w2)
                count += 1
            epochs += 1
        
        return w1, w2
        
    def predict_svm(self, w1, w2):

        ## Clip the weights 
        index = list(range(10,90))
        w1 = np.delete(w1,index)
        w2 = np.delete(w2,index)
        
        w1 = w1.reshape(10,1)
        w2 = w2.reshape(10,1)
        ## Extract the test data features 
        test_f1 = x_test[:,0]
        test_f2 = x_test[:,1]
        
        test_f1 = test_f1.reshape(10,1)
        test_f2 = test_f2.reshape(10,1)
        ## Predict
        y_pred = w1 * test_f1 + w2 * test_f2
        predictions = []
        for val in y_pred:
            if(val > 1):
                predictions.append(1)
            else:
                predictions.append(-1)
        
        print(accuracy_score(y_test,predictions))
            
    
if __name__ == "__main__":

    
    svm_object = SVM_Iris_Two_Features_Only()
    x_train, x_test, y_train, y_test = svm_object.train_test()
    w1, w2 = svm_object.svm_scratch(x_train, x_test, y_train, y_test)
    svm_object.predict_svm(w1, w2)
    
"""
    
