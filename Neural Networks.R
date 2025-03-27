
library(nnet)
library(caret)
library(mice)
library(dplyr)
library(neuralnet)
library(smotefamily)
library(doParallel)

# Register parallel backend to speed up processing
registerDoParallel(cores = detectCores())

#13/19 0.36
# Initialize a list to store the results of the model evaluations
model_results <- list()


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



# Loop through each imputed dataset
for (i in 1:10) {
  # Complete the dataset for the i-th imputation
  completed_data <- complete(imputed_data, action = i)
  #completed_data <-completed_data %>% dplyr::select(-api_water,-api_total_impurities, -api_l_impurity, -api_ps05)
  #completed_data <-completed_data %>% dplyr::select( -api_l_impurity)
  
  # Convert 'weekend' factor to numeric: "yes" -> 1, "no" -> 0
  completed_data$weekend <- ifelse(completed_data$weekend == "yes", 1, 0)
  normalize <- function(x) {
    return((x - min(x)) / (max(x) - min(x)))
  }
  
  # Apply normalization to each column except the target variable
  data_normalized <- as.data.frame(lapply(completed_data[, -which(colnames(completed_data) == "Target")], normalize))
  
  # Combine with the target variable
  data_normalized$Target <- completed_data$Target
  
  
  
  # Identify numeric columns for outlier capping
  numeric_columns <- sapply(data_normalized, is.numeric)
  

  # Re-attach target column and ensure it's a factor
  data_normalized$Target <- factor(completed_data$Target, levels = c(0, 1))
  completed_data <- data_normalized
  
  # Split data into 80% training and 20% testing
  set.seed(123) # Set seed for reproducibility
  training_indices <- sample(1:nrow(completed_data), 0.8 * nrow(completed_data))
  train_data <- completed_data[training_indices, ]
  test_data <- completed_data[-training_indices, ]
  
  # Convert 'Target' to a factor with appropriate level names
  train_data$Target <- factor(train_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))
  test_data$Target <- factor(test_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))
  
  print("before smote")
  
  # Apply SMOTE to the training data
  is.factor(train_data$Target)
  set.seed(123)
  train_data <- SMOTE(X = train_data[, !names(train_data) %in% "Target"], target = train_data[,"Target"], K = 5, dup_size = 0)
  colnames(train_data$data)[ncol(train_data$data)] <- "Target" # renaming the column
  
  
  print("after smote")
  
  
  # Define a grid for tuning the neural network parameters
  #grids <- expand.grid(.size = seq(from = 1, to = 20, by = 3), .decay = seq(from = 0.08, to = 0.4, by = 0.04))
  #grids <- expand.grid(.size = c(1,2,3,4,5,6), .decay = c(1,0.1,0.01,0.001))
  #grids <- expand.grid(.size = c(1,2,3,4,5,6), .decay = c(1,0.1,0.01,0.001))
  grids <- expand.grid(.decay = runif(10, 0.0001, 0.1), .size = round(runif(5, 0, 5)))
  
  
  # Set up cross-validation parameters
  set.seed(1)
  cv_control <- trainControl(
    method = "repeatedcv",
    number = 10,
    #repeats = 3,
    savePredictions = "final",
    classProbs = TRUE,
    #search = "grid",
    #sampling = "smote",
    summaryFunction = balancedAccSummary 
  )
  
 
  
  # Train the neural network model
  set.seed(1)
  nn_model <- train(
    Target ~ .,
    data = train_data$data,
    method = "nnet",
    trControl = cv_control,
    metric = "Balanced_Accuracy",
    #preProcess = "scale", # Scale features
    #tuneGrid = grids,
    trace = FALSE,
    #MaxNWts = 1000,
    linout = FALSE # binary outcome
  )
  

  
  # After the model has been trained
  best_parameters <- nn_model$bestTune
  
  best_size = best_parameters$size
  best_decay = best_parameters$decay
  
  # Print the best parameters
  cat("Best .size: ", best_parameters$size, "\n")
  cat("Best .decay: ", best_parameters$decay, "\n")
  
  
  # Evaluate the model performance
  predictions <- predict(nn_model, newdata = test_data)#, type = "raw")
  
  # Ensure predictions are factored with levels in the specific order you want
  predictions <- factor(predictions, levels = levels(test_data$Target))
  
  
  confusion_matrix <- confusionMatrix(predictions, test_data$Target)
  
  # Store the results in the list
  model_results[[i]] <- confusion_matrix
  
  print(model_results[[i]])
  print(nn_model)
  
  
  
  
  
  
  
  
  
  
  ###Feature Importance#####
  
  # Separate predictors and target
  train_x <- train_data$data[, -which(names(train_data$data) == "Target")]
  train_y <- train_data$data$Target
  test_x <- test_data[, -which(names(test_data) == "Target")]
  test_y <- test_data$Target
  
  # Function to calculate balanced accuracy using a neural network model
  calculate_balanced_accuracy_nn <- function(nn_model, test_x, test_y) {
    prob_predictions <- predict(nn_model, newdata = test_x, type = "prob")
    predictions <- apply(prob_predictions, 1, which.max)
    predictions <- factor(predictions, levels = 1:length(levels(test_y)), labels = levels(test_y))
    test_y <- factor(test_y, levels = levels(predictions))
    
    confusion_matrix <- confusionMatrix(predictions, test_y)
    return(confusion_matrix$byClass["Balanced Accuracy"])
  }
  
  # Assuming nn_model is already trained and available
  baseline_accuracy <- calculate_balanced_accuracy_nn(nn_model, test_x, test_y)
  
  # Initialize a vector to store balanced accuracy drops for each covariate
  accuracy_drops <- numeric(ncol(test_x))
  names(accuracy_drops) <- colnames(test_x)
  
  # Loop through each covariate and reshuffle it
  for (col in colnames(test_x)) {
    perturbed_test_x <- test_x
    
    # Shuffle the column values
    set.seed(123)  # Ensure reproducibility
    perturbed_test_x[[col]] <- sample(perturbed_test_x[[col]])
    
    # Calculate balanced accuracy after perturbation
    perturbed_accuracy <- calculate_balanced_accuracy_nn(nn_model, perturbed_test_x, test_y)
    
    # Calculate the drop in balanced accuracy
    accuracy_drops[col] <- baseline_accuracy - perturbed_accuracy
  }
  
  # Sort the covariates by balanced accuracy drop and select the top 5
  top_5_covariates <- sort(accuracy_drops, decreasing = TRUE)[1:5]
  names(top_5_covariates) <- names(sort(accuracy_drops, decreasing = TRUE)[1:5])
  
  # Print the top 5 most important covariates and their corresponding balanced accuracy drops
  cat("Top 5 Covariates based on Balanced Accuracy Drop:\n")
  print(top_5_covariates)
  
  
  
  
  
  
  
}

# Print or analyze the results
#print(model_results)



