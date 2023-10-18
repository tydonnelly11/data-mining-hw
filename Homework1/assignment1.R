
dataset = read.csv('Customer_Data.csv')

dataset$Age = ifelse(is.na(dataset$Age),
                     median(dataset$Age, na.rm = TRUE),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        mean(dataset$Salary, na.rm = TRUE),
                        dataset$Salary)

dataset$Country = factor(dataset$Country,
                         levels = c('China', 'India', 'Sri lanka'),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.67)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

normalize_min_max = function(data, cols, train) {
  
  n = nrow(data)
  
  if(missing(train)){
    for (i in cols) {
      
      col_min = min(data[, i])
      col_max = max(data[, i])
      
      
      data[, i] = (data[, i] - col_min) / (col_max - col_min)
    }
  }
  else{
    for (i in cols) {
      
      col_min = min(train[, i])
      col_max = max(train[, i])
      
      
      data[, i] = (data[, i] - col_min) / (col_max - col_min)
    }
    
  }
  return(data)
}


train_set_scaled = normalize_min_max(training_set, c(2,3))

test_set_scaled = normalize_min_max(test_set, c(2,3), training_set)
