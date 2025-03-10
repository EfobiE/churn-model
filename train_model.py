import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Telco_customer_churn.csv',encoding='utf-8')

# Quick overview of the dataset
print(df.head())

#pre-processing

# Label Encoding for binary columns
label_encoder = LabelEncoder()#Encode(literally count) target labels with value between 0 and n_classes-1.
df['Churn Label '] = label_encoder.fit_transform(df['Churn Label '])  # Target variable (Yes/No to 1/0)
df['Churn Value '] = label_encoder.fit_transform(df['Churn Value ']) 
df['Paperless Billing '] = label_encoder.fit_transform(df['Paperless Billing '])
df['Senior Citizen '] = label_encoder.fit_transform(df['Senior Citizen '])
df['Partner '] = label_encoder.fit_transform(df['Partner '])
df['Dependents '] = label_encoder.fit_transform(df['Dependents '])
df['Phone Service '] = label_encoder.fit_transform(df['Phone Service '])

# One-hot encoding for other categorical features
df = pd.get_dummies(df, drop_first=True)#This is one-hot encoding, converts categorical variables (rows) into multiple binary columns
scaler = StandardScaler()#Standardize features by removing the mean and scaling to unit variance. normalizer
numerical_columns = ['Tenure Months ', 'Monthly Charges ','Churn Reason_Competitor offered higher download speeds','Payment Method            _Electronic check          ',
        'Internet Service _Fiber optic      ','Dependents ', 'CLTV ','Contract       _Two year       ','Churn Reason_Attitude of support person',
        'Churn Reason_Competitor offered more data','Churn Reason_Competitor made better offer','Churn Reason_Product dissatisfaction',
       'Churn Reason_Lack of self-service on Website',
       'Churn Reason_Network reliability',
       'Churn Reason_Service dissatisfaction', 'Latitude  ']  #These are the inputs for the prediction model
# Example. columns[27]="Total Charges "
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])#fit:to compute the mean and std dev for a given feature.
#transform:to perform scaling using mean and std dev calculated in fit.

# Define features (X) and target (y)
df.drop('Churn Label ', axis=1)
df.drop('Churn Value ', axis=1)
X = df[numerical_columns] 
print("Defining Features:", X.columns)
#tenure months 28,monthly charges 29,total charges 27,latitude 7, longitude 8, senior citizen 10, Partner 11, paperless billing 24
y = df['Churn Label ']

# Train-test split
#test data: a subset of the training dataset that is utilized to give an accurate evaluation of a final model fit
#default, 25% of our data is test set and 75% data goes into training tests. in this case 20% of data is test set
#random_state acts like a numpy seed, used for data reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")


# Initialize and train the model
#rfc is a supervised Machine learning algorithm used for classification, regression, and other tasks using decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.values , y_train.values)

# Get feature importances from the trained RandomForest model
feature_importances = model.feature_importances_

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]
top_n = 25  # Keep the top 25 features

# Select the most important features
important_features = X_train.columns[sorted_indices[:top_n]]
print("Top Features:", important_features)

# Predict on test data
y_pred = model.predict(X_test.values)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix: shows how well a classification model is performing by comparing its predictions to the actual results
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("figures/confusion_matrix.png")
plt.close()
# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC score for evaluating binary classifiers
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Feature importance plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(35,5))
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance")
plt.savefig("figures/feature_importance.png")
plt.close()
print(f"Feature Importance Figure Saved and Closed")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train , y_train)

print(f"Best Parameters: {grid_search.best_params_}")

import joblib

# Save model and scaler
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')