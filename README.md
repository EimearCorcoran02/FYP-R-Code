# Thesis Code Repository: Investigating Yield Adequacy in Tablet Compression

This repository contains the code used in my thesis, **"Investigating the Factors Influencing Yield Adequacy in the Tablet Compression Stage for a Cholesterol-Lowering Drug Using Logistic Regression and Machine Learning Techniques."** The code includes scripts for data cleaning, exploratory analysis, and the implementation of various machine learning models in R.



## Repository Structure

- **HistogramsFYP.R**  
  Generates histograms and visualizations for exploratory data analysis.

- **K-Nearest Neighbours FYP.R**  
  Implements the K-Nearest Neighbours model.

- **K-Nearest Neighbours with SMOTE FYP.R**  
  Applies KNN with SMOTE for handling class imbalance.

- **Logistic Regression with Missing Columns Removed.R**  
  Logistic Regression analysis after removing columns with missing data.

- **Logistic Regression with Missing Rows Removed.R**  
  Logistic Regression analysis after removing rows with missing data.

- **Logistic Regression with Smote.R**  
  Logistic Regression model using SMOTE for class imbalance.

- **Logistic Regression without Smote.R**  
  Logistic Regression model without applying SMOTE.

- **Neural Networks .R**  
  Implements Neural Network modeling for yield prediction.

- **Random forest with SMOTE.R**  
  Applies Random Forest with SMOTE to improve class balance.

- **Data Cleaning.R**  
  Performs data cleaning, conversion, and imputation using MICE.

## Prerequisites

Ensure you have R (version ≥ 4.0) installed along with the following packages:

- dplyr
- mice
- caret
- class
- ROSE
- randomForest
- nnet
- ggplot2
- gridExtra
- smotefamily
- e1071

You can install them by running the following in R:

```r
install.packages(c("dplyr", "mice", "caret", "class", "ROSE", "randomForest", "nnet", "ggplot2", "gridExtra", "smotefamily", "e1071"))
