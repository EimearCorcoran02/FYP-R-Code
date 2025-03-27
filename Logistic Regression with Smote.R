##############################Model selection using BIC###################

library(smotefamily)  # Ensure smotefamily is loaded
library(caret)        # For confusionMatrix
library(dplyr)
library(mice)
library(e1071) 


# Step 6: Perform Stepwise Model Selection Using BIC and Check for Interaction Terms
selected_models <- list()
model_metrics <- list()

# Loop through each imputed dataset to fit the logistic regression model and perform stepwise selection
for (i in 1:10) {  # Loop for 10 imputations
  # Get the ith imputed dataset (from data4_list, already filtered)
  imputed_dataset <- complete(imputed_data, action = i)
  
  # Convert 'weekend' to numeric and ensure 'Target' is a factor
  imputed_dataset$weekend <- ifelse(imputed_dataset$weekend == "yes", 1, 0)  
  imputed_dataset$Target <- factor(imputed_dataset$Target, levels = c(0, 1), labels = c(0, 1))
  
  # Create training (80%) and testing (20%) datasets
  set.seed(1)  # For reproducibility
  training_rows <- sample(1:nrow(imputed_dataset), 0.8 * nrow(imputed_dataset))
  training_data <- imputed_dataset[training_rows, ]
  testing_data <- imputed_dataset[-training_rows, ]
  
  # Convert 'weekend' to numeric and ensure 'Target' is a factor
  training_data$weekend <- ifelse(training_data$weekend == "yes", 1, 0)  
  testing_data$weekend <- ifelse(testing_data$weekend == "yes", 1, 0)   
  training_data$Target <- factor(training_data$Target, levels = c(0, 1), labels = c(0, 1)) 
  testing_data$Target <- factor(testing_data$Target, levels = c(0, 1), labels = c(0, 1))    
  
  
  
  
  # Apply SMOTE to the training data
  is.factor(training_data$Target)
  set.seed(123)
  training_data <- SMOTE(X = training_data[, !names(training_data) %in% "Target"], target = training_data[,"Target"], K = 5, dup_size = 0)
  colnames(training_data$data)[ncol(training_data$data)] <- "Target" # renaming the column
  training_data$data$Target <- as.numeric(training_data$data$Target == 1) # back to numeric
  testing_data$Target <- as.numeric(testing_data$Target == 1)
  
 
  
  # Fit the logistic regression model using glm
  full_model <- glm(Target ~ ., family = binomial, data = training_data$data)  
  
  
  # Perform stepwise model selection based on BIC (without interaction terms)
  stepwise_model <- step(full_model, direction = "both", k = log(nrow(training_data$data)), trace = 0)  
  
  # Generate combinations of interaction terms for the selected predictors ####
  selected_predictors <- names(coef(stepwise_model))[-1]  # Remove intercept
  interaction_terms <- combn(selected_predictors, 2, simplify = FALSE)  # Get pairwise interactions
  
  # Initialize the best model as the current stepwise model ####
  best_model <- stepwise_model
  best_bic <- BIC(best_model)
  
  # Check each combination of interaction terms ####
  for (interaction in interaction_terms) {
    interaction_formula <- as.formula(paste("Target ~", paste(c(interaction, selected_predictors), collapse = " + "), "+", paste(interaction, collapse = ":")))
    model_with_interaction <- glm(interaction_formula, family = binomial, data = training_data$data)
    model_bic <- BIC(model_with_interaction)
    
    # If the BIC is better (lower), update the best model ####
    if (model_bic < best_bic) {
      best_model <- model_with_interaction
      best_bic <- model_bic
    }
  }
  
  # Store the best model after checking interactions ####
  selected_models[[i]] <- list(model = best_model, bic = best_bic)  
  
  
  # Store the final best model after checking interactions
  #selected_models[[i]] <- list(model = stepwise_model, bic = BIC(stepwise_model))  
  
  # Make predictions on the testing data using the best model
  predictions <- predict(stepwise_model, newdata = testing_data, type = "response")  
  predicted_classes <- ifelse(predictions > 0.5, 1, 0)  
  
  
  # Confusion matrix and performance metrics
  confusion <- confusionMatrix(factor(predicted_classes), factor(testing_data$Target), positive = "1")  
  model_metrics[[i]] <- list(
    ConfusionMatrix = confusion$table,
    Accuracy = confusion$overall['Accuracy'],
    Precision = confusion$byClass['Precision'],
    Recall = confusion$byClass['Recall'],
    F1 = confusion$byClass['F1']
  )  
  
  # Output the performance metrics for each model
  cat(sprintf("Model %d: \n", i))
  cat("Confusion Matrix:\n")
  print(confusion$table)
  cat("\nAccuracy: ", confusion$overall['Accuracy'], "\n")
  cat("Precision: ", confusion$byClass['Precision'], "\n")
  cat("Recall: ", confusion$byClass['Recall'], "\n")
  cat("F1 Score: ", confusion$byClass['F1'], "\n\n")
  print(best_model)
  print(summary(best_model))
}



# Step 7: View the best models based on the lowest BIC for each imputation
best_models <- lapply(selected_models, function(x) x$model)

# Step 8: Pool the best models from each imputation using Rubin’s Rules
pooled_results <- pool(best_models)

# Step 9: View the pooled results (combined estimates from the best models)
summary(pooled_results)
summary(stepwise_model)
model_metrics

