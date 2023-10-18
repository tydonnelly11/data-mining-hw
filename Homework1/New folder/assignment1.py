# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:03:52 2023

@author: legok
"""
''' It is a classification problem because we try to predict if the product is purcahsed based on
the persons country, salary, and age. In addtion, the value that we are trying to predict is a yes or no value
and it is a categorical output''' 
#Importing library's
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#Putting the data set into different sets based on the what the independent and dependet vars should be
dataset = pd.read_csv('Customer_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
Y = dataset.iloc[:, -1].to_numpy()

imputerMean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputerMedian = SimpleImputer(missing_values=np.nan, strategy='median')



X[:,1:2] = imputerMedian.fit_transform(X[:, 1:2])
X[:,2:] = imputerMean.fit_transform(X[:, 2:])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')

X = ct.fit_transform(X)
le = LabelEncoder()
Y = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

ms = MinMaxScaler()

X_train[:, 3:] = ms.fit_transform(X_train[:, 3:])
X_test[:, 3:] = ms.transform(X_test[:, 3:])

