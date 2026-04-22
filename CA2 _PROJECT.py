import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("ecommerce_sales_analysis.csv")

print(df.head())
print(df.columns)

# Cleaning
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# Feature
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Month'] = df['Order Date'].dt.month

# EDA
print(df.describe())
print(df.corr(numeric_only=True))

# Line plot
monthly = df.groupby('Month')['Sales'].mean().reset_index()
plt.figure()
sns.lineplot(x='Month', y='Sales', data=monthly, marker='o')
for i in range(len(monthly)):
    plt.text(monthly['Month'][i], monthly['Sales'][i],
             round(monthly['Sales'][i], 2), ha='center')
plt.title("Monthly Sales")
plt.show()

# Histogram
plt.figure()
sns.histplot(df['Sales'], kde=True)
plt.title("Sales Distribution")
plt.show()

# Regression plot
plt.figure()
sns.regplot(x='Quantity', y='Sales', data=df)
plt.title("Quantity vs Sales")
plt.show()

# Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation")
plt.show()

# Pairplot
sns.pairplot(df[['Sales', 'Quantity', 'Profit']], diag_kind='kde')
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x='Region', y='Sales', data=df)
plt.title("Sales by Region")
plt.show()

# Bar chart
category_sales = df.groupby('Category')['Sales'].sum().reset_index()
plt.figure()
sns.barplot(x='Category', y='Sales', data=category_sales)
plt.title("Category Sales")
plt.show()

# Scatter
plt.figure()
sns.scatterplot(x='Profit', y='Sales', data=df)
plt.title("Profit vs Sales")
plt.show()

# Pie chart
counts = df['Category'].value_counts()
plt.figure()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("Category Share")
plt.show()

# Outliers
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Sales'] >= Q1 - 1.5 * IQR) &
        (df['Sales'] <= Q3 + 1.5 * IQR)]

# Stats tests
sample = df['Sales'].sample(min(5000, len(df)))
print("Shapiro:", stats.shapiro(sample)[1])

regions = df['Region'].dropna().unique()
if len(regions) >= 2:
    g1 = df[df['Region'] == regions[0]]['Sales']
    g2 = df[df['Region'] == regions[1]]['Sales']
    print("T-test:", stats.ttest_ind(g1, g2)[1])

cont = pd.crosstab(df['Category'], df['Region'])
print("Chi-square:", stats.chi2_contingency(cont)[1])

# Linear Regression
X = df[['Quantity', 'Profit']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, pred))
