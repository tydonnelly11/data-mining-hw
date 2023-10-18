# -*- coding: utf-8 -*-
# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Housing_Data.csv')
X = dataset.iloc[:, 0:-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X_train)
y_scaled = sc_y.fit_transform(y_train.reshape(len(y_train), 1)).flatten()
X_test_scaled = sc_X.fit_transform(X_test)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf')  
regressor_svr.fit(X_scaled, y_scaled)
y_pred_scaled = regressor_svr.predict(X_test_scaled) #predict y values using x-test-scaled values
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(len(y_pred_scaled), 1)) #Reverse the scaling


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)
r2_adjusted = 1 - (((1 - r2) * (len(y_test) - 1))/ (len(y_test) - 6 - 1))
print(r2_adjusted)

#.69 1
#.68 2
#.53 3
#.80 4
#.75 5
#.66 6
#.71 7
#.66 8
#.56 9
#.74 10
#avg = .678


# Random Forest Regression

# Importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Housing_Data.csv')
X = dataset.iloc[:, 0:-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) #Split data into training and test

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor_forest = RandomForestRegressor(n_estimators = 500)
regressor_forest.fit(X_train, y_train)
y_pred_forest = regressor_forest.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred_forest)
print(r2)
r2_adjusted = (1 - (((1 - r2) * (len(y_test) - 1))/ (len(y_test) - 6 - 1)))
print(r2_adjusted)
#.78 1
#.76 2
#.55 3
#.81 4
#.73 5
#.61 6
#.76 7
#.73 8
#.57 9
#.65 10
# avg = .695

#The forest tree regression is slightly better with an average of .695 adjusted r-squared