##############################Model selection using BIC###################


library(caret)
library(MASS)
library(mice)

# Step 6: Perform Stepwise Model Selection Using BIC and Check for Interaction Terms
selected_models <- list()
model_metrics <- list()  #CHANGED: Added to store metrics

# Loop through each imputed dataset to fit the logistic regression model and perform stepwise selection
for (i in 1:10) {  # Loop for 10 imputations  #CHANGED: Corrected comment to match code
  # Get the ith imputed dataset (from data4_list, already filtered)
  imputed_dataset <- complete(imputed_data, action = i)
  
  # Splitting into training and testing datasets  #CHANGED: Adding training/testing split
  set.seed(1)  # Ensure reproducibility
  training_indices <- sample(1:nrow(imputed_dataset), 0.8 * nrow(imputed_dataset))
  training_data <- imputed_dataset[training_indices, ]
  testing_data <- imputed_dataset[-training_indices, ]
  
  
  
  # Fit the full logistic regression model using all predictors
  full_model <- glm(Target ~ ., family = binomial, data = training_data)  #CHANGED: Use training_data
  
  # Perform stepwise model selection based on BIC (without interaction terms)
  stepwise_model <- step(full_model, direction = "both", k = log(nrow(training_data)), trace = 0)  #CHANGED: Log of training_data
  
  # Store the stepwise model and its BIC value
  selected_models[[i]] <- list(model = stepwise_model, bic = BIC(stepwise_model))
  
  # Get the predictors used in the selected model
  selected_predictors <- names(coef(stepwise_model))[-1]  # Remove intercept
  
  # Generate combinations of interaction terms for the selected predictors
  interaction_terms <- combn(selected_predictors, 2, simplify = FALSE)  # Get pairwise interactions
  
  # Initialize the best model as the current stepwise model
  best_model <- stepwise_model
  best_bic <- BIC(best_model)
  
  # Check each combination of interaction terms
  for (interaction in interaction_terms) {
    # Create a formula with the interaction term
    interaction_formula <- as.formula(paste("Target ~", paste(c(interaction, selected_predictors), collapse = " + "), "+", paste(interaction, collapse = ":")))
    
    # Fit the model with the interaction term
    model_with_interaction <- glm(interaction_formula, family = binomial, data = training_data)  #CHANGED: Use training_data
    
    # Check the BIC for this new model
    model_bic <- BIC(model_with_interaction)
    
    # If the BIC is better (lower), update the best model
    if (model_bic < best_bic) {
      best_model <- model_with_interaction
      best_bic <- model_bic
    }
  }
  
  # Store the final best model after checking interactions
  selected_models[[i]]$model <- best_model
  selected_models[[i]]$bic <- best_bic
  
  # Evaluate model on the testing data  #CHANGED: Added model evaluation
  predictions <- predict(best_model, newdata = testing_data, type = "response")
  predicted_classes <- ifelse(predictions > 0.8, 1, 0)  #inequality can be changed to change weighting 
  confusion <- confusionMatrix(factor(predicted_classes), factor(testing_data$Target), positive = "1")
  model_metrics[[i]] <- list(
    ConfusionMatrix = confusion$table,
    Accuracy = confusion$overall['Accuracy'],
    Precision = confusion$byClass['Precision'],
    Recall = confusion$byClass['Recall'],
    F1 = confusion$byClass['F1']
    
  )
  print(summary(best_model))
  print(confusion)
}

# Display model performance metrics for each imputation  #CHANGED: Adding display of metrics
for (i in 1:10) {
  cat(sprintf("Metrics for Model %d: \n", i))
  print(model_metrics[[i]])
}

# Step 7: View the best models based on the lowest BIC for each imputation
best_models <- lapply(selected_models, function(x) x$model)

# Step 8: Pool the best models from each imputation using Rubin’s Rules
pooled_results <- pool(best_models)

# Step 9: View the pooled results (combined estimates from the best models)
summary(pooled_results)



############ploting probabilities################
# Ensure ggplot2 is installed and loaded
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
library(ggplot2)

# Ensure the best logistic regression model from the last iteration is available
#best_model <- best_models[[10]]  # Use the final imputed model

# Predict probabilities for both training and testing data
train_probs <- predict(best_model, newdata = training_data, type = "response")
test_probs <- predict(best_model, newdata = testing_data, type = "response")

# Convert probabilities into data frames for visualization
train_plot_df <- data.frame(
  Index = 1:nrow(training_data),
  PredictedProb = train_probs,
  ActualClass = factor(training_data$Target),  # Ensure it's a factor for coloring
  DataSet = "Training Set"
)

test_plot_df <- data.frame(
  Index = 1:nrow(testing_data),
  PredictedProb = test_probs,
  ActualClass = factor(testing_data$Target),  # Ensure it's a factor for coloring
  DataSet = "Testing Set"
)

# Combine both datasets for a single plot
combined_plot_df <- rbind(train_plot_df, test_plot_df)

# Create the visualization
plot_obj <- ggplot(combined_plot_df, aes(x = Index, y = PredictedProb, color = ActualClass)) +
  geom_point(alpha = 0.5) +  # Adjust alpha for better visibility
  facet_wrap(~ DataSet, scales = "free_x", 
             labeller = labeller(DataSet = c("Training Set" = "Training Set", 
                                             "Testing Set" = "Testing Set"))) +
  labs(
    title = "Predicted Probability of Success vs. Actual Class (Logistic Regression)",
    x = "Observation Index",
    y = "Predicted Probability for Success"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", colour = "white"),  
    plot.background = element_rect(fill = "white", colour = "white"),  
    panel.border = element_rect(colour = "black", fill = NA, size = 1),  
    strip.background = element_rect(fill = "white", colour = "black"),  
    strip.text = element_text(face = "bold", size = 14),  
    plot.title = element_text(size = 20, face = "bold"),  
    axis.title = element_text(size = 18),  
    axis.text = element_text(size = 16),  
    legend.text = element_text(size = 12)  
  ) +
  scale_color_manual(values = c("0" = "blue", "1" = "red"))  # Define colors for classes

# Save the updated graph
ggsave("LogisticRegression_ProbabilityPlot.png", plot = plot_obj, width = 16, height = 6, dpi = 300)

# Print the location where the graph is saved
cat("Plot saved to:", file.path(getwd(), "LogisticRegression_ProbabilityPlot.png"), "\n")
