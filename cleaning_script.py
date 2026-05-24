import pandas as pd
import numpy as np

# -----------------------------------
# STEP 1: LOAD DATASET
# -----------------------------------

# Replace with your dataset file name
df = pd.read_csv("SampleSuperstore.csv", encoding='latin1')

print("Dataset Loaded Successfully\n")

# -----------------------------------
# STEP 2: VIEW BASIC INFORMATION
# -----------------------------------

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# -----------------------------------
# STEP 3: CHECK MISSING VALUES
# -----------------------------------

print("\nMissing Values:\n")
print(df.isnull().sum())

# Fill missing numerical values with mean
numerical_columns = df.select_dtypes(include=np.number).columns

for col in numerical_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing categorical values with mode
categorical_columns = df.select_dtypes(include='object').columns

for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# -----------------------------------
# STEP 4: REMOVE DUPLICATES
# -----------------------------------

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates Removed Successfully")

# -----------------------------------
# STEP 5: STANDARDIZE TEXT COLUMNS
# -----------------------------------

for col in categorical_columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].str.title()

print("\nText Formatting Standardized")

# -----------------------------------
# STEP 6: DATE FORMAT CONVERSION
# -----------------------------------

date_columns = ['Order Date', 'Ship Date']

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

print("\nDate Columns Converted")

# -----------------------------------
# STEP 7: OUTLIER DETECTION
# -----------------------------------

print("\nStatistical Summary:\n")
print(df.describe())

# Example using IQR method for Sales column
if 'Sales' in df.columns:

    Q1 = df['Sales'].quantile(0.25)
    Q3 = df['Sales'].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[
        (df['Sales'] >= lower_limit) &
        (df['Sales'] <= upper_limit)
    ]

    print("\nOutliers Removed From Sales Column")

# -----------------------------------
# STEP 8: FEATURE ENGINEERING
# -----------------------------------

# Create Profit Category column
if 'Profit' in df.columns:

    df['Profit Category'] = df['Profit'].apply(
        lambda x: 'Profit' if x > 0 else 'Loss'
    )

print("\nNew Feature Created")

# -----------------------------------
# STEP 9: CHECK CLEANED DATA
# -----------------------------------

print("\nCleaned Dataset Preview:\n")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)

# -----------------------------------
# STEP 10: SAVE CLEANED DATASET
# -----------------------------------

df.to_csv("Sample - Superstore.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")

# -----------------------------------
# TASK COMPLETED
# -----------------------------------

print("\nData Cleaning and Transformation Completed Successfully!")