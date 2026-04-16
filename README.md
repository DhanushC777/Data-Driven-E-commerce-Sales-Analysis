# E-Commerce Sales Analysis Project

## Project Overview

This project performs a complete data analysis pipeline on an e-commerce dataset using Python. It includes data cleaning, exploratory data analysis (EDA), statistical testing, and machine learning models to extract business insights and predict sales performance.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn

## Dataset

The dataset file used for this project is:
`ecommerce_sales_analysis.csv`

Key columns include:

- Order Date
- Sales
- Quantity
- Profit
- Category
- Region

## Project Workflow

### 1. Data Loading

- Read the dataset using Pandas
- Display the first rows and dataset columns

### 2. Data Cleaning

- Removed duplicate records
- Filled missing numeric values using median imputation
- Converted the order date column to a datetime format

### 3. Feature Engineering

- Extracted the month from the order date for time-based analysis

### 4. Exploratory Data Analysis (EDA)

Performed several analyses and visualizations, including:

- Monthly sales trend line plot
- Sales distribution histogram
- Regression plot for quantity vs sales
- Correlation heatmap for numerical features
- Pairplot for relationship exploration
- Box plot for sales distribution by region
- Bar chart for sales by category
- Scatter plot for profit vs sales

### 5. Outlier Treatment

- Used the interquartile range (IQR) method to identify and remove extreme sales values
- Improved model reliability and data quality

### 6. Statistical Analysis

- Checked data distribution with the Shapiro-Wilk test
- Compared group means using t-tests
- Examined relationships between categorical variables with chi-square tests

### 7. Machine Learning Models

#### Linear Regression

- Predicted sales using features such as quantity and profit
- Evaluated model performance with mean squared error (MSE)

#### Logistic Regression

- Created a high-sales classification label
- Predicted whether a sale value is above the median
- Evaluated performance with accuracy and classification metrics

## Key Insights

- Sales vary by month, showing seasonal patterns
- Quantity and profit have a strong influence on sales
- Some product categories and regions contribute more revenue than others
- Statistical analysis confirms relationships between key factors
- Predictive models help forecast sales behavior

## How to Run the Project

1. Install dependencies:

```
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

2. Place `ecommerce_sales_analysis.csv` in the project folder

3. Run the Python script:

```
python CA2_PROJECT.py
```

## Future Improvements

- Add advanced machine learning models such as Random Forest or XGBoost
- Create a web interface or dashboard for results
- Add time series forecasting
- Expand feature engineering and model tuning

## Author

Dhanush
