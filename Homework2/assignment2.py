import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Dealership_Data.csv') # Importing the dataset
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, #Splitting up training set and test set with 25% going to test set
                                                    random_state = 0)



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) #Putting the training sets through the model

y_pred = regressor.predict(X_test) #prediction of results using regressor

plt.scatter(X_train, y_train, color = 'red') #Plotting of the training set graph
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('List price vs Sell price (training set)')
plt.xlabel('List price')
plt.ylabel('Sell price')
plt.show()

plt.scatter(X_test, y_test, color = 'red') #plotting of the test set graph
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('List price vs Sell price')
plt.xlabel('List price')
plt.ylabel('Sell price')
plt.show()


print("Sell price in thousands", regressor.predict([[20]])) #Printing the coefficents and predicted value
print("slope of model" , regressor.coef_)
print("Intercept of model" , regressor.intercept_)