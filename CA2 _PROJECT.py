import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report


#LOAD DATA
df = pd.read_csv("ecommerce_sales_analysis.csv")

print(df.head())
print(df.columns)


#DATA CLEANING
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

#FEATURE ENGINEERING
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Month'] = df['Order Date'].dt.month

#EDA
print(df.describe())
print(df.corr(numeric_only=True))


# 1.LINE PLOT (Monthly Sales Trend)
monthly = df.groupby('Month')['Sales'].mean().reset_index()

plt.figure()
sns.lineplot(x='Month', y='Sales', data=monthly, marker='o')
plt.title("Average Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Average Sales")

for i in range(len(monthly)):
    plt.text(monthly['Month'][i], monthly['Sales'][i], 
             round(monthly['Sales'][i],2), ha='center')

plt.show()


# 2.HISTOGRAM
plt.figure()
sns.histplot(df['Sales'], kde=True)
plt.title("Distribution of Sales")
plt.xlabel("Sales Value")
plt.ylabel("Frequency")
plt.show()


# 3.REGRESSION PLOT
plt.figure()
sns.regplot(x='Quantity', y='Sales', data=df)
plt.title("Relationship between Quantity and Sales")
plt.xlabel("Quantity Sold")
plt.ylabel("Sales")
plt.show()


# 4.CORRELATION HEATMAP
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Between Variables")
plt.show()


# 5.PAIRPLOT (NEW ADDED)
sns.pairplot(df[['Sales', 'Quantity', 'Profit']], diag_kind='kde')
plt.suptitle("Pairwise Relationships Between Key Variables", y=1.02)
plt.show()


# 6.BOX PLOT
plt.figure()
sns.boxplot(x='Region', y='Sales', data=df)
plt.title("Sales Distribution Across Regions")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.show()


# 7.BAR CHART
category_sales = df.groupby('Category')['Sales'].sum().reset_index()

plt.figure()
sns.barplot(x='Category', y='Sales', data=category_sales)
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.show()


# 8.SCATTER PLOT
plt.figure()
sns.scatterplot(x='Profit', y='Sales', data=df)
plt.title("Profit vs Sales Relationship")
plt.xlabel("Profit")
plt.ylabel("Sales")
plt.show()


# 9.PIE CHART
category_counts = df['Category'].value_counts()

plt.figure()
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
plt.title("Category Distribution (%)")
plt.show()


#OUTLIER REMOVAL
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Sales'] >= Q1 - 1.5*IQR) & (df['Sales'] <= Q3 + 1.5*IQR)]


#STATISTICAL TESTS

# Shapiro Test
sample = df['Sales'].sample(min(5000, len(df)))
print("Shapiro p-value:", stats.shapiro(sample)[1])

# T-Test
regions = df['Region'].unique()
if len(regions) >= 2:
    g1 = df[df['Region'] == regions[0]]['Sales']
    g2 = df[df['Region'] == regions[1]]['Sales']
    print("T-test p-value:", stats.ttest_ind(g1, g2)[1])

# Chi-Square Test
cont = pd.crosstab(df['Category'], df['Region'])
print("Chi-square p-value:", stats.chi2_contingency(cont)[1])


#LINEAR REGRESSION
X = df[['Quantity', 'Profit']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, pred))


#LOGISTIC REGRESSION
df['High_Sales'] = (df['Sales'] > df['Sales'].median()).astype(int)

X = df[['Quantity', 'Profit']]
y = df['High_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
