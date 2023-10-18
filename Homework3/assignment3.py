# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Crime_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, #Splits the training and testing set with 10% going to the test
                                                    random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) #Trains the model using the training sets

# Predicting the Test set results


from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
scy = StandardScaler()
X_train = scx.fit_transform(X_train) #Scale X
y_train = scy.fit_transform(y_train.reshape(len(y_train), 1)).flatten() #Scale Y



# Backward Elimination
import statsmodels.api as sm
X_train = sm.add_constant(X_train)

#X_opt = X_train[:, [0,1,2,3,4,5,6]]  Highest p-value is X3 with a p-vale of .945

#X_opt = X_train[:, [0,1,2,4,5,6]] Higest p-value is X5 with a p-value of .232

#X_opt = X_train[:, [0,1,2,4,6]] highest p-valye is X6 with a p-value of .100

X_opt = X_train[:, [0,1,2,4]] #All vars under .05 p-value
regressor_opt = sm.OLS(endog = y_train, exog = X_opt).fit()

regressor_opt.summary() 

print(regressor_opt.params) #First value is intercept, rest of values are coeffiecent values
print(regressor_opt.predict([1,500,50,30]))
