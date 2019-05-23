# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:38:39 2018

@author: Karthik Kumarasubramanian
"""
import numpy as np

## Generating data
def produce_Train_test():

    n = 1000
    m = 500
    mean1 = [1, 0]
    mean2 = [0, 1.5]
    covariance = [[1, 0.75], [0.75, 1]]

    x1_train = np.random.multivariate_normal(mean1, covariance, n)
    ## print(x1)
    x2_train = np.random.multivariate_normal(mean2, covariance, n)
    ## print(x2) 
    X_train = np.concatenate((x1_train,x2_train), axis = 0)
    ## print(X_train.shape)
    
    y1_train = np.zeros(n)
    y2_train = np.ones(n)
    Y_train = np.concatenate((y1_train, y2_train), axis = 0)
    ## print(y_train)
    ## print(y_train.shape)
    
    x1_test = np.random.multivariate_normal(mean1, covariance, m)
    ## print(x1)    
    x2_test = np.random.multivariate_normal(mean2, covariance, m)
    ## print(x2)     
    X_test= np.concatenate((x1_test,x2_test), axis = 0)
    ## print(X_train.shape)
    
    y1_test = np.zeros(m)
    y2_test = np.ones(m)    
    Y_test = np.concatenate((y1_test, y2_test), axis = 0)
    
    return X_train, Y_train, X_test, Y_test

## Activation Function
def sigmoid(x):
    
    return (1/ (1 + np.exp(-x)))

## Cross Entropy Function
def cross_entropy(h, y):
    
    return (-y * np.log(h) - (1 -y) * np.log(1 -h))

## Online gradient descent funtion    
def perceptron_Stochastic(X, y, alpha, iterations):
    
    w = np.ones(X.shape[1])
    J_hist = np.zeros(iterations)    
    n = X.shape[0]
    num = 0
    break_point = False
    
    for i in range(iterations):
        
        for j in range(n):
            
            h = sigmoid(X[j].dot(w))
            
            J_hist[i] = cross_entropy(h, y[j])

            for k in range(len(w)):
                
                grad = (1/n) * (y[j] - h) * X[j][k]
                w[k] = w[k] + (alpha * grad)
            
            if(J_hist[i] == 0):
                num = i
                break_point = True
                break
            
            elif(J_hist[i] == J_hist[i-1]):
                num = i
                break_point = True
                break
        
        if(break_point == True):
            break
    
    ## E = sum(J_hist)/n
    
    return w, num

def test_grad(X, w):
    
    m = X.shape[0]
    h = sigmoid(X.dot(w))
    y_pred = np.zeros(m)
    
    for i in range(len(y_pred)):
        
        if (h[i] > 0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
            
    return y_pred

def accuracy(y_pred, y):
    
    count = 0
    for i in range(y_pred.shape[0]):
        
        if(y_pred[i] == y[i]):

            count = count + 1
    
    accuracy = (count/y_pred.shape[0]) * 100
    
    return accuracy


X_train, Y_train, X_test, Y_test = produce_Train_test()
## print(X_test.shape)

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
## print(X_train)

alpha = float(input("\n Enter the learning rate from {1, 0.1, 0.01}: "))
iterations = 10000

w, stepno = perceptron_Stochastic(X_train, Y_train, alpha, iterations)
print("\n The weights are: ",w)
print("\n The number of iterations is: ",stepno)

Y_pred = test_grad(X_test, w)
## print(Y_pred)

acc = accuracy(Y_pred, Y_test)
print("\n The accuracy of the dataset is: ",acc)
