import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

# 1. Data Loading & Initial Analysis
def summarize_data(df):
    print("Veri setinin boyutu:", df.shape)
    print("\nVeri türleri:")
    print(df.dtypes)
    print("\nSınıf dağılımı:")
    print(df['quality'].value_counts())
    print("\nEksik veriler:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0])

# Load data
df = pd.read_csv("path/to/WineQT_missing.csv")
summarize_data(df)

# 2. Missing Value Treatment
# Standardize data before KNN imputation
scaler = StandardScaler()
X = df.drop(columns=['quality', "Id"])
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# KNN imputation
imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(X_scaled)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Inverse transform to original scale
X_recovered = scaler.inverse_transform(X_imputed)
X_recovered = pd.DataFrame(X_recovered, columns=X.columns)
X_recovered[['quality', "Id"]] = df[['quality', "Id"]]

# 3. Feature Engineering
def create_features(df):
    df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']
    df['fixed_volatile_ratio'] = df['fixed acidity'] / df['volatile acidity']
    df['sulfur_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
    return df

X_recovered = create_features(X_recovered)

# 4. Data Visualization
def plot_distributions(df):
    plt.figure(figsize=(15, 10))
    
    # Quality Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='quality')
    plt.title('Wine Quality Distribution')
    
    # Alcohol vs Quality
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='quality', y='alcohol')
    plt.title('Alcohol Content by Quality')
    
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

plot_distributions(X_recovered)

# 5. Outlier Detection & Removal
# Using LOF for outlier detection
X = X_recovered.drop(columns=['quality', "Id"])
y = X_recovered['quality']

lof = LocalOutlierFactor(n_neighbors=5)
outlier_labels = lof.fit_predict(X)
non_outlier_mask = outlier_labels != -1

# Remove outliers
X_cleaned = X[non_outlier_mask]
y_cleaned = y[non_outlier_mask]

print("Total samples:", len(X))
print("Non-outlier samples:", len(X_cleaned))
print("Outliers removed:", len(X) - len(X_cleaned))

# 6. Final Clean Dataset
clean_data = pd.concat([X_cleaned, y_cleaned], axis=1)
clean_data.to_csv("path/to/WineQT_cleaned.csv", index=False)

# Final visualization to verify cleaning
plot_distributions(clean_data)