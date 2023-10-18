dataset = read.csv('Housing_data.csv')


# Splitting the dataset into the Training set and Test set
library(caTools)
split = sample.split(dataset$Y, SplitRatio = 3/4)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting SVR to the whole dataset
library(e1071)
regressor = svm(formula = Y ~ .,
                data = training_set,
                type = 'eps-regression',
                kernel = 'radial')
y_pred = predict(regressor, test_set)

#Find R^2 and adjusted R^2
ssr = sum((test_set$Y - y_pred) ^ 2)
sst = sum((test_set$Y - mean(test_set$Y)) ^ 2)
r2 = 1 - (ssr/sst)
print(r2)
r2_adjusted = 1 - (1 - r2) * (length(test_set$Y) - 1) / (length(test_set$Y) - 6 - 1)
print(r2_adjusted) 

#.59
#.54
#.61
#.70
#.76
#.68
#.73
#.72
#.61
#.67
#avg = 0.656


# Random Forest Regression


#Splitting data in training and test set
library(caTools)
split = sample.split(dataset$Y, SplitRatio = 3/4)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Regression to the whole dataset
library(randomForest)
regressor = randomForest(formula = Y ~ .,
                         data = training_set,
                         ntree = 500)

y_pred = predict(regressor, newdata = test_set)

#Calculate R^2 and adjusted R^2
ssr = sum((test_set$Y - y_pred) ^ 2)
sst = sum((test_set$Y - mean(test_set$Y)) ^ 2)
r2 = 1 - (ssr/sst)
print(r2)
r2_adjusted = 1 - (1 - r2) * (length(test_set$Y) - 1) / (length(test_set$Y) - 6 - 1)
print(r2_adjusted) 

#.57
#.71
#.69
#.75
#.78
#.64
#.76
#.83
#.76
#.74
#avg = 0.723

#The random forest tree is the better model with an avg of .723