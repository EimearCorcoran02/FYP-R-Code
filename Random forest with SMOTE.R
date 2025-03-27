
# Load the necessary libraries
if (!require("caret")) install.packages("caret", dependencies = TRUE)
library(caret)
if (!require("mice")) install.packages("mice", dependencies = TRUE)
library(mice)
if (!require("randomForest")) install.packages("randomForest", dependencies = TRUE)
library(randomForest)
library(smotefamily)

# Assuming 'imputed_data' is your MICE output with 10 imputations
no_col=ncol(complete(imputed_data, action = 1))

# List to store models and performance metrics
model_list <- list()
performance_list <- list()

for (i in 1:10) {
  # Complete the data for each imputation
  completed_data <- complete(imputed_data, action = i)
  
  # Convert 'weekend' factor to numeric: "yes" -> 1, "no" -> 0
  completed_data$weekend <- ifelse(completed_data$weekend == "yes", 1, 0)
  
  # Split the data into 80% training and 20% testing
  set.seed(123)  # Set seed for reproducibility for each loop to ensure consistent splits
  training_indices <- createDataPartition(completed_data$Target, p = 0.8, list = FALSE)
  train_data <- completed_data[training_indices, ]
  test_data <- completed_data[-training_indices, ]
  
  # Ensure 'Target' is a factor
  train_data$Target <- factor(train_data$Target, levels = c(1, 0), labels = c(1, 0))
  test_data$Target <- factor(test_data$Target, levels = c(1, 0), labels = c(1, 0))
  
  # Apply SMOTE to the training data
  is.factor(train_data$Target)
  train_data <- SMOTE(X=train_data[,-43], target=train_data[,43], K = 5,dup_size=0)  
  colnames(train_data$data)[ncol(train_data$data)] <- "Target" # renaming the column
  
  # Ensure 'Target' still is a factor
  train_data$data$Target <- factor(train_data$data$Target, levels = c(1, 0), labels = c(1, 0))
  #test_data$Target <- factor(test_data$Target, levels = c(1, 0), labels = c(1, 0))
  
  # Use tuneRF to find the optimal mtry
  set.seed(123)
  tune_results <- tuneRF(
    x = train_data$data[, -ncol(train_data$data)],
    y = train_data$data$Target,
    mtryStart = round(sqrt(no_col)),
    ntreeTry = 100,
    stepFactor = 1.2,
    improve = 0.05, # If adding more variables does not improve the OOB error by at least 5%, the tuning process will stop
    trace = TRUE,
    plot = TRUE,
    doBest = FALSE
  )
  
  optimal_mtry <- tune_results[which.min(tune_results[,2]), 1]  # Find mtry with min OOB error
  
  # Define a grid of possible weights for class "1"
  weight_grid <- seq(1, 10, by = 1)
  best_oob <- Inf
  optimal_weight <- NA
  
  # Loop over the grid to find the optimal weight based on OOB error
  for (w in weight_grid) {
    rf_temp <- randomForest(
      Target ~ .,
      data = train_data$data,
      ntree = 100,        # Number of trees
      importance = TRUE,  # Calculate variable importance
      mtry = optimal_mtry,
      classwt = c("0" = 1, "1" = w)
    )
    
    # Get the OOB error rate from the last tree
    current_oob <- tail(rf_temp$err.rate[, "OOB"], n = 1)
    
    if (current_oob < best_oob) {
      best_oob <- current_oob
      optimal_weight <- w
    }
  }
  
  cat("Optimal class weight for class '1':", optimal_weight, "\n")
  
  # Train the final Random Forest model using the optimal weight
  rf_model <- randomForest(
    Target ~ .,
    data = train_data$data,
    ntree = 50,        # Reduced number of trees from 100 to 50
    mtry = round(sqrt(no_col)),  # Optimal number of variables tried at each split
    nodesize = 15,     # Minimum size of terminal nodes, increasing this value can lead to simpler models
    maxnodes = NULL,   # Allows growth of the tree till nodesize is reached
    importance = TRUE  # Calculate variable importance
  )
  
  # Store the model in the list
  model_list[[i]] <- rf_model
  
  # Use the model to predict on test data
  predictions <- predict(rf_model, test_data)
  
  # Evaluate model performance using confusion matrix from caret
  actual <- test_data$Target
  predicted <- predictions
  cm <- confusionMatrix(predicted, actual)
  performance_list[[i]] <- cm
  print(cm)
}

# Rows for each model, columns for each feature (excluding 'Target')
importance_matrix <- matrix(nrow = length(model_list), ncol = ncol(train_data$data) - 1)  # Assuming the last column is 'Target'
colnames(importance_matrix) <- names(train_data$data)[-ncol(train_data$data)]

# Loop through each model and store their importance
for (j in 1:length(model_list)) {
  # Fill each row with the importance data from each model
  importance_matrix[j, ] <- importance(model_list[[j]], type = 1)[, "MeanDecreaseAccuracy"]
}

# Calculate the average importance across all models (mean of each column)
average_importance <- matrix(nrow=1,ncol = ncol(train_data$data) - 1)
average_importance[1,] <- colMeans(importance_matrix, na.rm = TRUE)
colnames(average_importance) <-colnames(importance_matrix)

average_importance_t=t(average_importance)

average_importance_t[order(average_importance_t[,1],decreasing=TRUE),]

#####################Overfiting Checking#########################

# Make sure ggplot2 is installed and loaded
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
library(ggplot2)

# Predict probabilities for both training and testing data
train_probs <- predict(rf_model, train_data$data, type = "prob")[, "1"]
test_probs <- predict(rf_model, test_data, type = "prob")[, "1"]

# Create data frames for plotting
train_plot_df <- data.frame(
  Index = 1:nrow(train_data$data),
  PredictedProb = train_probs,
  ActualClass = train_data$data$Target,
  DataSet = "Training Set"
)

test_plot_df <- data.frame(
  Index = 1:nrow(test_data),
  PredictedProb = test_probs,
  ActualClass = test_data$Target,
  DataSet = "Testing Set"
)

# Combine the data frames
combined_plot_df <- rbind(train_plot_df, test_plot_df)

# Create the plot object
plot_obj <- ggplot(combined_plot_df, aes(x = Index, y = PredictedProb, color = ActualClass)) +
  geom_point(alpha = 0.5) +  # Adjusted alpha for better visibility if points overlap
  facet_wrap(~ DataSet, scales = "free_x", labeller = labeller(DataSet = c("Training Set" = "Training Set", "Testing Set" = "Testing Set"))) +
  labs(
    title = "Predicted Probability of Success vs. Actual Class",
    x = "Observation Index",
    y = "Predicted Probability for Success"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", colour = "white"),  # Set panel background to white
    plot.background = element_rect(fill = "white", colour = "white"),  # Set plot background to white
    panel.border = element_rect(colour = "black", fill = NA, size = 1),  # Add borders
    strip.background = element_rect(fill = "white", colour = "black"),  # Background for facet labels
    strip.text = element_text(face = "bold", size = 14),  # Increase and bold facet labels
    plot.title = element_text(size = 20, face = "bold"),  # Increase title font size
    axis.title = element_text(size = 18),  # Increase axis titles font size
    axis.text = element_text(size = 16),  # Increase axis text font size
    legend.text = element_text(size = 12)  # Increase legend text size
  ) +
  scale_color_manual(values = c("1" = "blue", "0" = "red"))  # Define specific colors for classes

# Save the plot as "RandomForestComparisonGraph.png" in the current working directory
ggsave("RandomForestComparisonGraph.png", plot = plot_obj, width = 16, height = 6, dpi = 300)  # Adjusted width for better layout

# Print the location where the graph is saved
cat("Plot saved to:", file.path(getwd(), "RandomForestComparisonGraph.png"), "\n")
