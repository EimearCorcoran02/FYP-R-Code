library(mice)

# Assuming 'imputed_data' is your MICE output
completed_data <- complete(imputed_data, action = 1)  # Get the first imputed dataset

# Identify numeric columns for histogram plotting
numeric_columns <- sapply(completed_data, is.numeric)
numeric_data <- completed_data[, numeric_columns]

# Number of numeric columns
num_cols <- ncol(numeric_data)

# Split the columns into two halves
first_half <- numeric_data[, 1:ceiling(num_cols / 2)]
second_half <- numeric_data[, (ceiling(num_cols / 2) + 1):num_cols]

# Function to plot histograms for a given set of columns
plot_histograms <- function(data_subset, title) {
  num_cols_subset <- ncol(data_subset)
  
  # Set up the plotting area for a grid
  plot_rows <- ceiling(sqrt(num_cols_subset))
  plot_cols <- ceiling(num_cols_subset / plot_rows)
  par(mfrow = c(plot_rows, plot_cols))  # Set grid layout
  
  # Loop through each column and plot histogram
  for (i in 1:num_cols_subset) {
    hist(data_subset[[i]], 
         main = paste("Histogram of", colnames(data_subset)[i]), 
         xlab = colnames(data_subset)[i], 
         col = "lightblue", 
         border = "black")
  }
  
  # Reset par to default settings after plotting
  par(mfrow = c(1, 1))
}
 
# Plot histograms for the first half
plot_histograms(first_half, "First Half of Columns") #REF779812

# Plot histograms for the second half
plot_histograms(second_half, "Second Half of Columns") #REF779812


#Graphing columns with missing values
# Select specific columns for histogram plotting
specific_columns <- completed_data[, c("api_water", "api_total_impurities", "api_ps05", "api_l_impurity")]

# Set up the plotting area
png("beforeimputgraph.png", width = 800, height = 600)  # Save the plot as a PNG file

# Set the layout for a 2x2 grid
par(mfrow = c(2, 2))

# Loop through each selected column and plot histograms with larger text
for (col in colnames(specific_columns)) {
  hist(specific_columns[[col]], 
       main = paste("Histogram of", col), 
       xlab = col, 
       col = "#29285D", 
       border = "black",
       cex.main = 1.5,  # Increase main title text size
       cex.lab = 1.5,   # Increase axis labels text size
       cex.axis = 1.2   # Increase axis tick labels text size
  )
}

# Reset par to default settings and close the graphics device
dev.off()

# Get and print the current working directory to see where the file is saved
getwd()


# Assuming 'completed_data' is your dataset and it includes 'api_l_impurity'
api_l_impurity <- completed_data$api_l_impurity

# Calculate the first and third quartiles
Q1 <- quantile(api_l_impurity, 0.25)
Q3 <- quantile(api_l_impurity, 0.75)

# Calculate the interquartile range
IQR <- Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Identify outliers
outliers <- api_l_impurity[api_l_impurity < lower_bound | api_l_impurity > upper_bound]

# Print outliers
print(outliers)

# Count the number of outliers
num_outliers <- length(outliers)
print(paste("Number of outliers in 'api_l_impurity':", num_outliers))