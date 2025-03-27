##########BRIEF DESCRIPTION OF DATA SETS#################
#data1: Initial cleaned dataset after removing censored/redundant data and converting most columns to numeric.
#data 2: removing those with high collinearity and lower predictive power
#data3: Imputed dataset after handling missing data with MICE.
#data4:  keeping only the columns that have predictive power based on their p-values from univariate binomial regression

##########################setting up##########################

# Load necessary libraries
library(dplyr)
library(mice)
library(caret)
library(e1071) 
#library(tidymodels)
#library(themis)
#install.packages("themis")
library(smotefamily)#





# Load the dataset
data <- read.csv("C:/Users/Eimear/OneDrive - University College Cork/College/FYP/processandlaboratory.csv")
names(data)
#changing some variables to factors
data$weekend <- factor(data$weekend, levels = c("yes", "no"))


# Create data1 by removing specific columns - ones which have censored data, are directly correlated to target, and ones which have been replaced with calculated columns
data1 <- data %>% dplyr::select(-Drug_release_average, -Drug_release_min, -Residual_solvent, -Total_impurities, -Impurity_O, -Impurity_L, -batch_yield, -tbl_yield, -start, -Reformatted.Start,-strength)

#############converting to numeric###############
#converting everything except to weekend to numeric (so we can get cor)

# Loop through each column in the dataset
for (col in colnames(data1)) {
  
  # Skip the "weekend" column
  if (col != "weekend") {
    
    # Check if the column is a factor or character
    if (is.factor(data1[[col]]) || is.character(data1[[col]])) {
      
      # Convert to numeric (for factors or characters)
      data1[[col]] <- as.numeric(as.character(data1[[col]]))
      
      # Print the status of the conversion
      cat(col, ": Converted to numeric\n")
      
    } else if (!is.numeric(data1[[col]])) {
      
      # Print a message if it can't be converted
      cat(col, ": Could not be converted\n")
      
    }
  }
}

# Verify the conversion
str(data1)  # Check the structure of the dataset after conversion

##############################

###Check if the correlation is greater than 0.7#################

# Vector of columns with missing data
columns_with_missing_data <- c("api_water", "api_total_impurities", "api_l_impurity", "api_content", "api_ps01", "api_ps05", "api_ps09", "tbl_min_weight", "tbl_max_weight")

#vector of columns which are factors
columns_which_are_factors <- c("weekend")

# Initialize an empty data frame to store results
correlation_results <- data.frame(Variable1 = character(), Variable2 = character(), Correlation = numeric(), 
                                  P_Value_Variable1 = numeric(), P_Value_Variable2 = numeric(), 
                                  Higher_P_Value_Variable = character(), stringsAsFactors = FALSE)

# Get all numeric columns from data1 excluding those with missing data and those which are factors
numeric_columns <- colnames(data1)[sapply(data1, is.numeric)  & (!colnames(data1) %in% columns_which_are_factors)]



# Loop through each pair of numeric variables to check for collinearity
for (i in 1:(length(numeric_columns) - 1)) {
  for (j in (i + 1):length(numeric_columns)) {
    var1 <- numeric_columns[i]
    var2 <- numeric_columns[j]
    
    # Confirm that both columns are numeric
    if (is.numeric(data1[[var1]]) & is.numeric(data1[[var2]])) {
      x=2
      # Calculate the correlation
      correlation <- tryCatch({
        cor(data1[[var1]], data1[[var2]], use = "complete.obs")
      }, error = function(e) {
        NA  # If there's an error, return NA and continue
      })
      
      # Check if the correlation is greater than 0.7 and is not NA
      if (!is.na(correlation) && abs(correlation) > 0.7) {
        # Get p-values for both variables using univariate logistic regression
        p_value_var1 <- summary(glm(Target ~ data1[[var1]], data = data1, family = binomial))$coefficients[2, 4]
        p_value_var2 <- summary(glm(Target ~ data1[[var2]], data = data1, family = binomial))$coefficients[2, 4]
        
        # Determine which variable has the higher p-value
        higher_p_value_var <- ifelse(p_value_var1 > p_value_var2, var1, var2)
        
        # Store results in the data frame
        correlation_results <- rbind(correlation_results, data.frame(Variable1 = var1, Variable2 = var2, 
                                                                     Correlation = correlation, 
                                                                     P_Value_Variable1 = p_value_var1, 
                                                                     P_Value_Variable2 = p_value_var2, 
                                                                     Higher_P_Value_Variable = higher_p_value_var))
      }
    } else {
      message(paste("Skipping non-numeric column pair:", var1, "and", var2))
    }
  }
}


# Display the results table
print(correlation_results)

# Create data2 by removing variables identified with high collinearity and lower predictive power
columns_to_remove <- unique(correlation_results$Higher_P_Value_Variable)
data2 <- data1 %>% dplyr::select(-one_of(columns_to_remove))

# Verify the new dataset
print(names(data2))


#############Finding blacks############
#how many in data2 have empty cells?
# Initialize an empty list to store columns with missing (NA) values
na_values_list <- list()

# Loop through each column in data2
for (col in colnames(data2)) {
  # Count the number of NA values in the column
  na_count <- sum(is.na(data2[[col]]))
  
  # If there are any NA values, add to the list
  if (na_count > 0) {
    na_values_list[[col]] <- na_count
  }
}

# Print the result
print(na_values_list)

#####################################
####looking at densities of columns remaining with missing values###### 
#searchterminr55555

# Set up the 2x2 plotting area
par(mfrow = c(2, 2))  # 2 rows and 2 columns

# Create a list of the columns to plot
columns_to_plot <- c("api_water", "api_total_impurities", "api_l_impurity", "api_ps05")

# Loop through the columns and create histograms
for (column in columns_to_plot) {
  if (column %in% colnames(data2)) {
    # Create the histogram
    hist(data2[[column]], 
         main = paste("Histogram of", column), 
         xlab = column, 
         col = "#29285D", 
         border = "black", 
         breaks = 30)  # You can adjust the number of breaks as needed
  } 
}

# Reset plotting parameters to default
par(mfrow = c(1, 1))  # Reset to single plot

######################Imputing####################################

library(mice)
library(dplyr)
library(MASS)

# Step 1: Perform MICE imputation on data2, creating 5 imputed datasets
imputed_data <- mice(data2, m = 10, method = 'cart', maxit = 50, seed = 500) # Adjust m to 5 for now

######################Univariate Analysis##########################################

# Step 2: Univariate analysis on each imputed dataset to get significant predictors
p_value_results <- data.frame(Variable = character(), P_Value = numeric(), stringsAsFactors = FALSE)

# Loop through each imputed dataset
for (i in 1:10) {  # Looping for 5 imputations
  # Get the ith imputed dataset
  imputed_dataset <- complete(imputed_data, action = i)
  
  # Loop through each column in the imputed dataset (excluding 'Target')
  for (col in colnames(imputed_dataset)) {
    if (col != "Target") {
      
      # Perform univariate binomial regression for each predictor
      model <- glm(Target ~ imputed_dataset[[col]], data = imputed_dataset, family = binomial)
      
      # Extract the p-value for the predictor variable
      p_value <- summary(model)$coefficients[2, 4]  # 2nd row is the predictor, 4th column is the p-value
      
      # Append the result to the data frame
      p_value_results <- rbind(p_value_results, data.frame(Variable = col, P_Value = p_value))
    }
  }
}
####################graphing histograms for imputed values#######################
#####rsearchterm777777

# Define columns that had missing values and need imputation
columns_to_plot <- c("api_water", "api_total_impurities", "api_l_impurity", "api_ps05")

# Set up the plotting area with one plot per covariate (1 original + 5 imputations = 6 plots per row)
par(mfrow = c(length(columns_to_plot), 6))  # Adjust number of rows and columns based on number of variables

# Loop through each covariate
for (column in columns_to_plot) {
  # Calculate xlim based on the range of the original data and imputed datasets
  min_value <- min(c(data2[[column]], sapply(1:10, function(i) complete(imputed_data, action = i)[[column]])), na.rm = TRUE)
  max_value <- max(c(data2[[column]], sapply(1:10, function(i) complete(imputed_data, action = i)[[column]])), na.rm = TRUE)
  
  # Original Data Histogram
  hist(data2[[column]], 
       main = paste("Original:", column), 
       xlab = column, 
       col = "lightblue", 
       border = "black", 
       breaks = 30, 
       xlim = c(min_value, max_value))
  
  # Loop through each imputed dataset (from 1 to 10 imputations)
  for (i in 1:10) {
    imputed_dataset <- complete(imputed_data, action = i)
    
    # Imputed Data Histogram for each imputation
    hist(imputed_dataset[[column]], 
         main = paste("Imputed:", column, "Imputation", i), 
         xlab = column, 
         col = "#62A25D", 
         border = "black", 
         breaks = 30, 
         add = FALSE, 
         xlim = c(min_value, max_value))
  }
}

# Reset plotting layout to default (single plot)
par(mfrow = c(1, 1))  # Reset to single plot

#############analysis of imputed data#############################

####searchterm8888888

# Function to calculate and print mean and standard deviation for each covariate and each imputation
calculate_stats <- function(data_before, data_after, columns) {
  for (column in columns) {
    # Calculate mean and standard deviation for the original data
    mean_before <- mean(data_before[[column]], na.rm = TRUE)
    sd_before <- sd(data_before[[column]], na.rm = TRUE)
    
    cat("\nColumn:", column, "\n")
    cat("Original Mean:", mean_before, " | Original SD:", sd_before, "\n")
    
    # Loop through each imputed dataset to calculate mean and standard deviation
    for (i in 1:10) {
      imputed_dataset <- complete(imputed_data, action = i)
      
      # Calculate mean and standard deviation for each imputed dataset
      mean_after <- mean(imputed_dataset[[column]], na.rm = TRUE)
      sd_after <- sd(imputed_dataset[[column]], na.rm = TRUE)
      
      cat("Imputation", i, "Mean:", mean_after, " | Imputation", i, "SD:", sd_after, "\n")
    }
  }
}

# Call the function to calculate and print statistics
calculate_stats(data2, data3, columns_to_plot)

###############Filtering cols with p-value<=0.1###########################
# Step 3: Filter for columns with p-value <= 0.1 (significant predictors)
columns_to_keep <- p_value_results$Variable[p_value_results$P_Value <= 0.1]

#columns_to_keep <- as.character(columns_to_keep)

# Step 4: Create data4 for each imputation with "Target" and the selected significant predictors
data4_list <- list()

# Loop through each imputation (1 to 5)
for (i in 1:10) {
  
  # Get the ith imputed dataset
  imputed_dataset <- complete(imputed_data, action = i)
  
  # Ensure columns_to_keep is a character vector
  #columns_to_keep <- as.character(columns_to_keep)
  
  # Create data4 with "Target" and the selected significant predictors
  data4 <- imputed_dataset %>% dplyr::select(Target, all_of(columns_to_keep))
  
  # Store the resulting dataset in the list
  data4_list[[i]] <- data4
}
#############################################

# Step 3: Filter for columns with p-value <= 0.1 (significant predictors)
columns_to_keep <- p_value_results$Variable[p_value_results$P_Value <= 0.1]

#columns_to_keep <- as.character(columns_to_keep)

# Step 4: Create data4 for each imputation with "Target" and the selected significant predictors
data4_list <- list()

# Loop through each imputation (1 to 5)
for (i in 1:10) {
  
  # Get the ith imputed dataset
  imputed_dataset <- complete(imputed_data, action = i)
  
  # Ensure columns_to_keep is a character vector
  #columns_to_keep <- as.character(columns_to_keep)
  
  # Create data4 with "Target" and the selected significant predictors
  data4 <- imputed_dataset %>% dplyr::select(Target, all_of(columns_to_keep))
  
  # Store the resulting dataset in the list
  data4_list[[i]] <- data4
}


#########GRAPHING FOR POWERPOINT###########


# Load required libraries
if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
if (!require("gridExtra")) install.packages("gridExtra", dependencies = TRUE)
library(ggplot2)
library(gridExtra)

# Define the variables with missing values
missing_vars <- c("api_water", "api_total_impurities", "api_l_impurity", "api_ps05")

# Extract original dataset (before imputation)
original_data <- imputed_data$data  # This extracts the dataset before MICE imputation

# Extract first imputed dataset
imputed_first <- complete(imputed_data, action = 1)

# Set the path to the Downloads folder (Update path accordingly)
download_path <- "C:\\Users\\Eimear\\OneDrive - University College Cork\\College\\FYP\\Test"

# Define color scheme
original_color <- "#29285D"    # Dark Blue
imputed_color <- "#62A25D"     # Green for First Imputation

# Initialize list to store plots
plots_list <- list()

# Loop through each variable and generate side-by-side histogram plots
for (var in missing_vars) {
  
  # Create a combined dataset for plotting
  plot_data <- data.frame(
    Value = c(original_data[[var]], imputed_first[[var]]),
    Source = rep(c("Original", "First Imputation"), each = nrow(original_data))
  )
  
  # Create the histogram plot
  plot <- ggplot(plot_data, aes(x = Value, fill = Source)) +
    geom_histogram(alpha = 0.6, position = "identity", bins = 30, color = "black") +
    scale_fill_manual(values = c("Original" = original_color, "First Imputation" = imputed_color)) +
    labs(
      title = paste("Histogram Comparison of", var, "Before and After Imputation"),
      x = var,
      y = "Frequency",
      fill = "Dataset"
    ) +
    theme_minimal() +
    theme(legend.position = "top")
  
  # Add plot to list
  plots_list[[var]] <- plot
}

# Arrange plots as a 2x2 grid and save as a single PDF
pdf_filename <- file.path(download_path, "Combined_Histograms_Comparison.pdf")
pdf(pdf_filename, width = 16, height = 12)
grid.arrange(grobs = plots_list, ncol = 2, nrow = 2)
dev.off()

print(paste("Saved combined histogram comparison in:", pdf_filename))

