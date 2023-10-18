# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Disease_Data.csv')
X = dataset.iloc[:, 0:1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_feature = PolynomialFeatures(degree = 2) #Poly feature of degree 2
X_poly = poly_feature.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y) #fit the regressor with X_poly and y

# Finding the optimal degree using p-value


from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
sc_y = StandardScaler() 
sc_x = StandardScaler()

y = sc_y.fit_transform(y.reshape(len(y), 1)).flatten()  #scale y

'''
poly_feature = PolynomialFeatures(degree = 2) 
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc_x.fit_transform(X_poly[:, 1:])      #Highest P-value is .003
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()


poly_feature = PolynomialFeatures(degree = 3)  
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc_x.fit_transform(X_poly[:, 1:])      #Highest P-value is .009
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()
   

poly_feature = PolynomialFeatures(degree = 4)       
X_poly = poly_feature.fit_transform(X)              #Highest P-value is .000
X_poly[:,1:] = sc_x.fit_transform(X_poly[:, 1:])
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()
'''

poly_feature = PolynomialFeatures(degree = 5)       
X_poly = poly_feature.fit_transform(X)              #All p-values under .05
X_poly[:,1:] = sc_x.fit_transform(X_poly[:, 1:])
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()


'''
poly_feature = PolynomialFeatures(degree = 6) 
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc.fit_transform(X_poly[:, 1:])      #P-value of degree 1 is higher than .05
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()
'''


# Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Days')
plt.ylabel('Number of cases(scaled)')
plt.show()

x_val = poly_feature.transform([[365]])
x_val[:,1:] = sc_x.transform(x_val[:,1:])

scaled_pred = regressor.predict(x_val)

y_pred = sc_y.inverse_transform(scaled_pred.reshape(len(scaled_pred), 1))
print(y_pred)
