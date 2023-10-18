# Polynomial Regression

# Importing the dataset
dataset = read.csv('Disease_Data.csv')
# Fitting Polynomial Regression to the whole dataset


dataset$Day2 = dataset$Day^2 
regressor = lm(formula = Cumulative.Cases ~ ., #Higest P-value of .003
               data = dataset)


 #Finding the optimal degree using p-value
dataset$Day3 = dataset$Day^3
regressor = lm(formula = Cumulative.Cases ~ ., #Highest p-value of .008
               data = dataset)

dataset$Day4 = dataset$Day^4
regressor = lm(formula = Cumulative.Cases ~ ., #Highest p-value of 2e-16
               data = dataset)

dataset$Day5 = dataset$Day^5
regressor = lm(formula = Cumulative.Cases ~ ., #Highest p-value of 2e-16
               data = dataset)

#dataset$Day6 = dataset$Day^6
#regressor = lm(formula = Cumulative.Cases ~ ., #Day has p-value of .99
#               data = dataset)


summary(regressor)
# Visualizing the Polynomial Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Day, y = dataset$Cumulative.Cases),
             colour = 'red') +
  geom_line(aes(x = dataset$Day, y = predict(regressor, newdata = dataset)), #plot line using X values and y pred values 
            colour = 'blue') +
  ggtitle('Polynomial Regression') +
  xlab('Days') +
  ylab('Cumlative cases')

y_pred1 = predict(regressor, newdata = data.frame('Day' = 365, 'Day2' = 365^2, 'Day3' = 365^3, 'Day4' = 365^4, 'Day5' = 365^5))
