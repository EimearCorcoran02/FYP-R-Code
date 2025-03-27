# Load the necessary libraries
if (!require("caret")) install.packages("caret", dependencies = TRUE)
library(caret)
if (!require("mice")) install.packages("mice", dependencies = TRUE)
library(mice)
if (!require("randomForest")) install.packages("randomForest", dependencies = TRUE)
library(randomForest)

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
  train_data$Target <- factor(train_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))
  test_data$Target <- factor(test_data$Target, levels = c(1, 0), labels = c("Class1", "Class0"))
  
  
  

  
  # Train the Random Forest model using randomForest()
  rf_model <- randomForest(
    Target ~ .,
    data = train_data,
    ntree = 1000,  # Number of trees
    importance = TRUE,  # Calculate variable importance
    mtry=round(sqrt(no_col)),
    #classwt = c('Class0'=1, 'Class1'=10)
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
importance_matrix <- matrix(nrow = length(model_list), ncol = ncol(train_data) - 1)  # Assuming the last column is 'Target'
colnames(importance_matrix) <- names(train_data)[-ncol(train_data)]

# Loop through each model and store their importance
for (j in 1:length(model_list)) {
  # Fill each row with the importance data from each model
  importance_matrix[j, ] <- importance(model_list[[j]], type = 1)[, "MeanDecreaseAccuracy"]
}

# Calculate the average importance across all models (mean of each column)
average_importance <- matrix(nrow=1,ncol = ncol(train_data) - 1)
average_importance[1,] <- colMeans(importance_matrix, na.rm = TRUE)
colnames(average_importance) <-colnames(importance_matrix)

average_importance_t=t(average_importance)

average_importance_t[order(average_importance_t[,1],decreasing=TRUE),]

##############################################
# Make sure ggplot2 is installed and loaded
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
library(ggplot2)

# Use the last trained rf_model and its corresponding training data (from your loop)
# Alternatively, you can choose a specific model from model_list, e.g., model_list[[1]]
test_probs <- predict(rf_model, train_data, type = "prob")[, "Class1"]

# Create a data frame for plotting
plot_df <- data.frame(
  Index = 1:nrow(train_data),
  PredictedProb = test_probs,
  ActualClass = train_data$Target
)

# Create the plot object with a white background and set y-axis limits
plot_obj <- ggplot(plot_df, aes(x = Index, y = PredictedProb, color = ActualClass)) +
  geom_point() +
  labs(
    title = "Training Set: Predicted Probability of Success vs. Actual Class",
    x = "Observation Index",
    y = "Predicted Probability for Success"
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", colour = "white"), # Set panel background to white
    plot.background = element_rect(fill = "white", colour = "white") # Set plot background to white
  ) +
  scale_y_continuous(limits = c(0, 1)) # Set y-axis to go from 0 to 1

# Save the plot as "RandomForestWithoutSMOTEGraph.png" in the current working directory
ggsave("RandomForestWithoutSMOTEGraph2.png", plot = plot_obj, width = 8, height = 6, dpi = 300)

# Print the location where the graph is saved
cat("Plot saved to:", file.path(getwd(), "RandomForestWithoutSMOTEGraph2.png"), "\n")
