# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('Crime_Data.csv')


# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Y, SplitRatio = 0.9)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Y ~ ., 
               data = training_set)

newdata = data.frame('X1' = c(500),'X2' = c(50),'X4'= c(30))

#Building the optimal model using Backward Elimination

#regressor_opt = lm(formula = Y ~ X1 + X2 + X3 + X4 + X5 + X6, data=training_set) X3 is largest P-value of .799

#regressor_opt = lm(formula = Y ~ X1 + X2 + X4 + X5 + X6, data=training_set) X6 is largest P-value of .245

#regressor_opt = lm(formula = Y ~ X1 + X2 + X4 + X5, data=training_set) X5 is largest P-value of .408

regressor_opt = lm(formula = Y ~ X1 + X2 + X4, data=training_set) #All of the remaing independent vars are under a p-value of .05

y_pred = predict(regressor_opt, newdata = test_set)


y_pred1 = predict(regressor_opt, newdata = data.frame('X1' = c(500),'X2' = c(50),'X4'= c(30)))
print(y_pred1)

print(regressor_opt$coefficients)

