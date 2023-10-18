# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()



# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
sc = StandardScaler()
X = sc.fit_transform(X)

#Encoding categorical variables
le = LabelEncoder()
le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
y = le.transform(y)


# Training the Kernel SVM model on dataset
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X, y)



# Predicting the results
y_pred = classifier.predict(X)

# Showing the Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y, y_pred))
print(accuracy_score(y, y_pred))


# Visualizing the set results
from matplotlib.colors import ListedColormap
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.
             predict(np.array([X1.flatten(), X2.flatten()]).T).
             reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'yellow')))
for i in np.unique(y_set):
    plt.scatter(X_set[y_set == i, 0], X_set[y_set == i, 1],
                color = ListedColormap(('red', 'green','yellow'))(i),
                edgecolors = 'black')
plt.title('Kernel SVM')
plt.xlabel('Sepal length (Scaled)')
plt.ylabel('Sepal width (Scaled)')
plt.show()

