# Load required libraries
if (!require("class")) install.packages("class", dependencies = TRUE) # For KNN
if (!require("mice")) install.packages("mice", dependencies = TRUE) # For multiple imputation
if (!require("caret")) install.packages("caret", dependencies = TRUE) # For evaluation metrics and cross-validation

library(class)
library(mice)
library(caret)
library(ROSE)

# Assuming 'imputed_data' is your MICE output with 10 imputations

#Accessing first imputation to get best k as same k was found to work for each imputation 
first_imputation <- complete(imputed_data, action = 1) #first imputation

# Convert 'weekend' factor to numeric: "yes" -> 1, "no" -> 0
first_imputation$weekend <- ifelse(first_imputation$weekend == "yes", 1, 0)

# Defining Summary function "Balanced_Accuracy" for training imbalanced data
balancedAccSummary <- function(data, lev = NULL, model = NULL) {
  if (is.null(lev)) {
    stop("Levels should be specified for binary classification")
  }
  
  # Confusion matrix calculation
  cm <- confusionMatrix(data$pred, data$obs, positive = lev[1] )
  
  # Calculating sensitivity and specificity
  sensitivity <- cm$byClass['Sensitivity']
  specificity <- cm$byClass['Specificity']
  
  
  # Balanced accuracy calculation
  balancedAcc <- (sensitivity + specificity) / 2
  
  # Return named vector
  out <- c(Balanced_Accuracy=balancedAcc,sensitivity=sensitivity,specificity=specificity)
  names(out) <- c("Balanced_Accuracy","sensitivity","specificity")
  return(out)
}

# Split the combined data into 80% training and 20% testing
set.seed(123) # Set seed for reproducibility
training_indices <- sample(1:nrow(first_imputation), 0.8 * nrow(first_imputation))
train_data <- first_imputation[training_indices, ]
test_data <- first_imputation[-training_indices, ]

# Ensure 'Target' is a factor with syntactically valid names for R variable names
train_data$Target <- factor(train_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))
test_data$Target <- factor(test_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))

# Define a range of k values to test (odd numbers between 1 and 20)
k_values <- seq(1, 30, by = 2) # Odd values to avoid ties

# Setup cross-validation parameters with shuffling and repeated CV
set.seed(123)
cv_control <- trainControl(
  method = "cv",
  number = 5,
  search = "grid",
  savePredictions = "final",
  classProbs = TRUE, # If class probabilities are needed
  #summaryFunction = twoClassSummary, # If dealing with binary classification
  summaryFunction = balancedAccSummary,
  returnResamp = "all",
  sampling = "smote", # Address class imbalance by up-sampling
  allowParallel = TRUE
)

# Perform cross-validation for each k using combined training data
cv_results <- train(
  Target ~ .,
  data = train_data,
  method = "knn",
  metric = "Balanced_Accuracy",
  preProcess = "scale", # Scale features
  tuneGrid = expand.grid(k = k_values),
  trControl = cv_control
)

# Find the best k based on cross-validation accuracy
best_k <- cv_results$bestTune$k
cat("\nBest k based on cross-validation:", best_k, "\n")

# Apply KNN with the best k for each imputed dataset
final_performance_list <- list() # To store final performance metrics
for (i in 1:10) {
  completed_data <- complete(imputed_data, action = i)
  completed_data$weekend <- ifelse(completed_data$weekend == "yes", 1, 0)
  
  # Split the data into 80% training and 20% testing
  set.seed(123)
  training_indices <- sample(1:nrow(completed_data), 0.8 * nrow(completed_data))
  train_data <- completed_data[training_indices, ]
  test_data<-completed_data[-training_indices, ]
  
  # Ensure 'Target' is a factor with valid names
  train_data$Target <- factor(train_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))
  test_data$Target <- factor(test_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))
  
  # Separate predictors and target for KNN
  train_x <- train_data[, -which(names(train_data) == "Target")]
  train_y <- train_data$Target
  test_x <- test_data[, -which(names(test_data) == "Target")]
  test_y <- test_data$Target
  
  # Apply KNN with the best k
  predictions <- knn(train = train_x, test = test_x, cl = train_y, k = best_k)
  
  # Evaluate model performance using confusion matrix
  confusion_matrix <- confusionMatrix(predictions, test_y)
  
  # Store final performance metrics for this imputation
  final_performance_list[[i]] <- list(
    Model = paste("KNN with best k =", best_k),
    Predictions = predictions,
    Accuracy = confusion_matrix$overall['Accuracy'],
    Precision = confusion_matrix$byClass['Precision'],
    Recall = confusion_matrix$byClass['Recall'],
    F1 = confusion_matrix$byClass['F1'],
    ConfusionMatrix = confusion_matrix$table
  )
}

# Print final performance metrics for each imputation
for (i in 1:10) {
  cat("\nFinal Performance for Imputation", i, ":\n")
  print(final_performance_list[[i]])
}


