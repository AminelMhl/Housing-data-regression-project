import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
df = pd.read_csv("./dataset/Housing.csv")

# Convert all columns to numeric, coercing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Drop non-numeric columns
df = df.dropna(axis=1, how='all')

# Display first few rows
print(df.head())

# Basic info and statistics
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if any)
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# Check for outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Ensure target variable is numerical
y = df['price']
if y.dtype == 'object':
    y = pd.to_numeric(y, errors='coerce')

# Drop target variable from features
X = df.drop(columns=['price'])

# Convert categorical variables to dummy variables (if needed)
X = pd.get_dummies(X, drop_first=True)

# Ensure all features are numerical
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Drop rows with NaN values (if any)
X = X.dropna()
y = y.dropna()

# Exploratory Data Analysis (EDA)
sns.pairplot(df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Define dependent and independent variables
y = df['price']  # Assuming 'price' is the target variable
X = df.drop(columns=['price'])  # Drop target variable from features

# Convert categorical variables to dummy variables (if needed)
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add constant for intercept in regression model
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit OLS Regression Model
model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())

# Predictions
y_pred = model.predict(X_test_sm)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Residual Plot
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()

sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()

# Feature Importance Plot
importance = model.params[1:]
plt.figure(figsize=(10,6))
sns.barplot(x=importance.index, y=importance.values)
plt.xticks(rotation=45)
plt.title("Feature Importance in Regression Model")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.show()
