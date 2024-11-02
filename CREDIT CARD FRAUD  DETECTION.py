#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('fraudTrain.csv')
data = pd.read_csv('fraudTest.csv')

# Convert `trans_date_trans_time` to datetime and extract features
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
data['trans_hour'] = data['trans_date_trans_time'].dt.hour
data['trans_day'] = data['trans_date_trans_time'].dt.day
data['trans_month'] = data['trans_date_trans_time'].dt.month

# Drop columns that may not be relevant for fraud detection
data = data.drop(['trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 
                  'city', 'state', 'zip', 'lat', 'long', 'dob', 'trans_num'], axis=1)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in ['merchant', 'category', 'gender', 'job']:
    data[column] = label_encoder.fit_transform(data[column])

# Define features and target variable
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Standardize only the numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Function to train and evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

# Logistic Regression
log_reg = LogisticRegression()
evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")

# Decision Tree Classifier
dtree = DecisionTreeClassifier(random_state=42)
evaluate_model(dtree, X_train, X_test, y_train, y_test, "Decision Tree")

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")


# In[ ]:




