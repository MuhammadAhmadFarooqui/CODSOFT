#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = r'C:\Users\Muhammad Ahmad\Downloads\archive (9)\Churn_Modelling.csv'
data = pd.read_csv(file_path)

# Explore and preprocess the data
# Selecting relevant features
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = data[features]
y = data['Exited']  # Target column representing churn (1 for churn, 0 for not churn)

# Encoding categorical features
label_encoder = LabelEncoder()
X['Geography'] = label_encoder.fit_transform(X['Geography'])
X['Gender'] = label_encoder.fit_transform(X['Gender'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Evaluate the Logistic Regression model
print("Logistic Regression Classifier")
accuracy_logreg = accuracy_score(y_test, y_pred_logreg) * 100
print(f"Accuracy: {accuracy_logreg:.2f}%")
print(classification_report(y_test, y_pred_logreg))

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Classifier")
accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
print(f"Accuracy: {accuracy_rf:.2f}%")
print(classification_report(y_test, y_pred_rf))

# Train a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test)

# Evaluate the Gradient Boosting model
print("Gradient Boosting Classifier")
accuracy_gb = accuracy_score(y_test, y_pred_gb) * 100
print(f"Accuracy: {accuracy_gb:.2f}%")
print(classification_report(y_test, y_pred_gb))


# In[ ]:




