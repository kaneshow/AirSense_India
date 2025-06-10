# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# Step 2: Download Dataset from Kaggle
dataset_path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
print("Dataset downloaded to:", dataset_path)

# Step 3: Load the 'city_day.csv' File
data = pd.read_csv(f"{dataset_path}/city_day.csv")
print("First few rows:\n", data.head())

# Step 4: Handle Missing Values
print("\nMissing Values:\n", data.isnull().sum())
data.fillna(data.mean(numeric_only=True), inplace=True)

# Step 5: Convert 'Date' Column to Datetime Format
data['Date'] = pd.to_datetime(data['Date'])

# Step 6: Select and Rename Relevant Columns
data.rename(columns={'O3': 'Ozone'}, inplace=True)
features = ['Date', 'PM2.5', 'PM10', 'NO2', 'CO', 'Ozone']
data = data[features]

# Step 7: Feature Engineering - Create AQI (Simple Average)
data['AQI'] = (data['PM2.5'] + data['PM10'] + data['NO2']) / 3

# Step 8: Data Integrity Checks
print("\nData Types:\n", data.dtypes)
print("\nNegative Values Detected:\n", data[data[['PM2.5', 'PM10', 'NO2', 'CO', 'Ozone']] < 0].dropna(how='all'))

# Step 9: Summary Statistics
print("\nSummary Statistics:\n", data.describe())

# Step 10: Time Series Visualization of Pollutants
data.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(data['PM2.5'], label='PM2.5', color='blue')
plt.plot(data['PM10'], label='PM10', color='orange')
plt.title('Pollution Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Outlier Detection using IQR (PM2.5)
Q1 = data['PM2.5'].quantile(0.25)
Q3 = data['PM2.5'].quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data['PM2.5'] < (Q1 - 1.5 * IQR)) | (data['PM2.5'] > (Q3 + 1.5 * IQR))]
print("\nPM2.5 Outliers:\n", outliers)

# Step 12: Log Transformation for Skewed Data
data['PM2.5_log'] = np.log(data['PM2.5'] + 1)

# Step 13: Bar Chart of Average Pollution Levels
avg_pollution = data[['PM2.5', 'PM10', 'NO2']].mean()
avg_pollution.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Average Pollution Levels')
plt.ylabel('Concentration')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 14: Histogram of PM2.5 Levels
plt.figure(figsize=(10, 5))
sns.histplot(data['PM2.5'], bins=30, kde=True, color='purple')
plt.title('Distribution of PM2.5 Levels')
plt.xlabel('PM2.5 Concentration')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
