import pandas as pd

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

df.to_csv("./dataset/cleaned_housing.csv", index=False)

