###LONG RUN TIME#####

if (!require("caret")) install.packages("caret", dependencies = TRUE)
library(caret)
if (!require("mice")) install.packages("mice", dependencies = TRUE)
library(mice)
if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
library(e1071)
if (!require("kernlab")) install.packages("kernlab", dependencies = TRUE)
library(kernlab)
if (!require("dplyr")) install.packages("dplyr", dependencies = TRUE)
library(dplyr)
if (!require("smotefamily")) install.packages("smotefamily", dependencies = TRUE)
library(smotefamily)

library(doParallel)

# Register parallel backend to speed up processing
registerDoParallel(cores = detectCores())

# Assuming 'imputed_data' is your MICE output with 10 imputations

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


# List to store models and performance metrics
model_list <- list()
performance_list <- list()

# Function to cap outliers
winsorize <- function(x) {
  qnt <- quantile(x, probs = c(0.05, 0.95), na.rm = TRUE)
  x[x < qnt[1]] <- qnt[1]
  x[x > qnt[2]] <- qnt[2]
  return(x)
}


for (i in 1:10) {
  # Complete the data for each imputation
  completed_data <- complete(imputed_data, action = i)
  
  # Convert 'weekend' factor to numeric: "yes" -> 1, "no" -> 0
  completed_data$weekend <- as.numeric(ifelse(completed_data$weekend == "yes", 1, 0))
  
  normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
  }
  
  # Apply normalization to each column except the target variable
  data_normalized <- as.data.frame(lapply(completed_data[, -which(colnames(completed_data) == "Target")], normalize))
  
  # Combine with the target variable
  data_normalized$Target <- completed_data$Target
  
  
  # Identify numeric columns for outlier capping
  #numeric_columns <- sapply(data_normalized, is.numeric)
  
  # Apply winsorizing to each numeric column, excluding the target column
 # data_normalized[numeric_columns] <- lapply(data_normalized[numeric_columns], winsorize)
  
  # Re-attach target column and ensure it's a factor
  data_normalized$Target <- factor(completed_data$Target, levels = c(0, 1))
  completed_data <- data_normalized
  
  
  
  
  
  
  # Split the data into 80% training and 20% testing
  set.seed(123)  # Set seed for reproducibility for each loop to ensure consistent splits
  training_indices <- createDataPartition(completed_data$Target, p = 0.8, list = FALSE)
  train_data <- completed_data[training_indices, ]
  test_data <- completed_data[-training_indices, ]
  
  
  # Identify numeric columns for outlier capping
  numeric_columns <- sapply(train_data , is.numeric)
  
  # Apply winsorizing to each numeric column, excluding the target column
  train_data [numeric_columns] <- lapply(train_data [numeric_columns], winsorize)
  
  
  # Ensure 'Target' is a factor 
  train_data$Target <- factor(train_data$Target, levels = c(0, 1))
  test_data$Target <- factor(test_data$Target, levels = c(0, 1))
  
  # Ensure 'Target' is the last column to align with train data
  test_data <- test_data[c(setdiff(names(test_data), "Target"), "Target")]
  
  
  # Apply SMOTE to the training data
  is.factor(train_data$Target)
  train_data$Target <- as.numeric(as.character(train_data$Target))
  #train_data <- SMOTE(X=train_data[,-47], target=train_data[,47], K = 5,dup_size=0) 
  train_data <- SMOTE(X = train_data[, !names(train_data) %in% "Target"], target = train_data[,"Target"], K = 5, dup_size = 0)
  colnames(train_data$data)[ncol(train_data$data)] <- "Target" # renaming the column
  
  # Ensure 'Target' still is a factor 
  train_data$data$Target <- factor(train_data$data$Target, levels = c(0, 1), labels = c("Class0", "Class1"))
  test_data$Target <- factor(test_data$Target, levels = c(0, 1), labels = c("Class0", "Class1"))
  
  train_control <- trainControl(
    method = "repeatedcv", 
    number = 10,
    summaryFunction = balancedAccSummary,
    classProbs = TRUE,  # If you need probability output
    savePredictions = "final",
    #preProcOptions = "scale",
    selectionFunction = "best")
  
  
  # Define a broader random tuning grid for C
  set.seed(123)  # For reproducibility of random sampling
  tune_grid <- expand.grid(
    #sigma = runif(5, 0.001, 1),C = runif(5, 0.001, 10)
    C = runif(20, 0.001, 10)
  )
  
  
  
  
  # Use train() to find the optimal SVM parameters
  set.seed(123)
  svm_tune <- train(
    Target ~ .,
    data = train_data$data,
    method = "svmLinear", 
    trControl = train_control,
    #tuneLength = 10,
    tuneGrid = tune_grid,
    search = "random",
    metric = "Balanced_Accuracy"
    
    
  )
  
  best_C=svm_tune$bestTune$C
  
  
  
  # Predict on the test data
  predictions <- predict(svm_tune, newdata = test_data)
  
  # Evaluate model performance using confusion matrix
  actual <- test_data$Target
  predicted <- predictions
  cm <- confusionMatrix(predicted, actual)
  performance_list[[i]] <- cm  
  print(cm)
  print(best_C)
  
  
  # Extract the best model (final model used)
  best_model <- svm_tune$finalModel
  
  ####Feature Importance#####
  final_model <- svm_tune$finalModel
  
  # Access Coef
  if (class(final_model) == "ksvm") {
    # Extract the support vectors
    support_vectors <- final_model@xmatrix[[1]]  # matrix of support vectors
    # Extract the coefficients (alphas), adjusted for the target labels
    alphas <- final_model@coef[[1]]  # vector of coefficients
    
    # Compute the actual SVM coefficients for each feature
    svm_coefs <- data.frame(
      "Importance" = colSums(support_vectors * alphas),
      "AbsImportance" = abs(colSums(support_vectors * alphas))
    )
    
    # Now, create a new table and sort it by 'Abs Importance' in descending order
    sorted_svm_coefs <- svm_coefs[order(svm_coefs$AbsImportance,decreasing = TRUE), ] 
    
    # Display the coefficients
    print(svm_coefs)
    print(sorted_svm_coefs)
  } else {
    print("The final model is not an SVM model from kernlab.")
  }
  
  
}



###########################

# Define a tuning grid for the cost parameter using random values
set.seed(123)  # For reproducibility of random sampling
cost_values <- runif(20, 0.001, 10)

# Use tune.svm from e1071 to find the optimal SVM parameters with 10-fold cross-validation
set.seed(123)
svm_tune <- tune.svm(
  Target ~ .,
  data = train_data$data,
  kernel = "linear",  # Use a linear kernel so that feature weights can be extracted
  cost = cost_values,
  tunecontrol = tune.control(cross = 10)
)

# Retrieve the best cost value and best model
best_cost <- svm_tune$best.parameters$cost
print(best_cost)
best_model <- svm_tune$best.model

# Predict on the test data using the tuned model
predictions <- predict(best_model, newdata = test_data)

# Evaluate model performance using a confusion matrix from caret
cm <- confusionMatrix(predictions, test_data$Target)
performance_list[[i]] <- cm  
print(cm)

#### Feature Importance Extraction using e1071::svm ####
# For a linear SVM, the weight vector can be computed as:
#    w = t(coefs) %*% SV
# where 'coefs' are the Lagrange multipliers and 'SV' are the support vectors.
if (1==1){#(best_model$kernel == "linear") {
  # Compute the weight vector for each feature
  w <- as.vector(t(best_model$coefs) %*% best_model$SV)
  
  # Get the feature names (assuming they are the same as the columns of train_data$data, excluding "Target")
  feature_names <- setdiff(colnames(train_data$data), "Target")
  
  # Create a data frame containing the computed weights and their absolute values
  svm_coefs <- data.frame(
    Feature = feature_names,
    Importance = w,
    AbsImportance = abs(w)
  )
  
  # Sort the features by their absolute importance in descending order
  sorted_svm_coefs <- svm_coefs[order(svm_coefs$AbsImportance, decreasing = TRUE), ]
  
  # Display the results
  print(svm_coefs)
  print(sorted_svm_coefs)
} else {
  print("The final model does not have a linear kernel.")
}


