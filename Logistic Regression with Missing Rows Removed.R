##########################Setting up##########################

# Load necessary libraries
library(dplyr)

# Remove all rows with any missing values in data2
data_remove_rows <- na.omit(data2)

# View the dimensions to confirm rows were removed
dim(data_remove_rows)


################Univariate Analysis##############################


# Step 2: Univariate analysis on each imputed dataset to get significant predictors
p_value_results <- data.frame(Variable = character(), P_Value = numeric(), stringsAsFactors = FALSE)

# Loop through each imputed dataset
for (i in 1:1) {  # Looping for 5 imputations
  # Get the ith imputed dataset
  
  
  # Loop through each column in the imputed dataset (excluding 'Target')
  for (col in colnames(data_remove_rows)) {
    if (col != "Target") {
      
      # Perform univariate binomial regression for each predictor
      model <- glm(Target ~ data_remove_rows[[col]], data = data_remove_rows, family = binomial)
      
      # Extract the p-value for the predictor variable
      p_value <- summary(model)$coefficients[2, 4]  # 2nd row is the predictor, 4th column is the p-value
      
      # Append the result to the data frame
      p_value_results <- rbind(p_value_results, data.frame(Variable = col, P_Value = p_value))
    }
  }
}


###############Filtering cols with p-value<=0.1###########################
# Step 3: Filter for columns with p-value <= 0.1 (significant predictors)
columns_to_keep <- p_value_results$Variable[p_value_results$P_Value <= 0.1]

# Create data4 with "Target" and the selected significant predictors
data_remove_rows<- data_remove_rows %>% dplyr::select(Target, all_of(columns_to_keep))


######################Stepwise Model Selection##############################



# Fit the full model with all selected predictors
full_model <- glm(Target ~ ., family = binomial, data = data_remove_rows)

# Perform stepwise selection based on BIC
stepwise_model <- step(full_model, direction = "both", k = log(nrow(data_remove_rows)))

# Get predictors from the stepwise model
selected_predictors <- names(coef(stepwise_model))[-1]

# Replace "weekendno" with "weekend" if it appears in the predictors
selected_predictors <- ifelse(selected_predictors == "weekendno", "weekend", selected_predictors)



##############################Check for Interaction Terms##############################

# Generate pairwise interaction terms for the selected predictors
interaction_terms <- combn(selected_predictors, 2, simplify = FALSE)

# Initialize the best model as the current stepwise model
best_model <- stepwise_model
best_bic <- BIC(stepwise_model)

# Loop through each interaction term
for (interaction in interaction_terms) {
  # Create a formula with the interaction term
  interaction_formula <- as.formula(
    paste("Target ~", 
          paste(selected_predictors, collapse = " + "), 
          "+", paste(interaction, collapse = ":"))
  )
  
  # Fit the model with the interaction term
  model_with_interaction <- glm(interaction_formula, family = binomial, data = data_remove_rows)
  model_bic <- BIC(model_with_interaction)
  
  # Update the best model if the BIC improves
  if (model_bic < best_bic) {
    best_model <- model_with_interaction
    best_bic <- model_bic
  }
}

# Display the summary of the best model
summary(best_model)


