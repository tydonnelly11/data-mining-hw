# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()


from sklearn.preprocessing import PolynomialFeatures

#Adding degree2 to datas
poly_feature = PolynomialFeatures(degree = 2, include_bias=False) 
X = poly_feature.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Showing the Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))

#Scaling and adding degree2 variables
input_array = np.array([X1.flatten(), X2.flatten()]).T
labels = sc.inverse_transform(poly_feature.transform(input_array))
labels = labels[:, 0:2]
labels = poly_feature.fit_transform(labels)
labels = sc.transform(labels)

plt.contourf(X1, X2, classifier.predict(labels).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

for i in np.unique(y_set):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                color = ListedColormap(('red', 'green'))(i),
                edgecolors = 'black')
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.show()

# Visualizing the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))

#Scaling and adding degree2 variables
input_array = np.array([X1.flatten(), X2.flatten()]).T
labels = sc.inverse_transform(poly_feature.transform(input_array))
labels = labels[:, 0:2]
labels = poly_feature.fit_transform(labels)
labels = sc.transform(labels)

plt.contourf(X1, X2, classifier.predict(labels).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))

for i in np.unique(y_set):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                color = ListedColormap(('red', 'green'))(i),
                edgecolors = 'black')
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.show()

##Background for graphs above stayes the same

