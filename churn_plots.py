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

#The folder to save plots
sf = "plots"

# Countplot for Churn
plt.figure(figsize=(6,4))
sns.countplot(x='Churn Value ', data=df, palette='coolwarm')
plt.title('Overall Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig(sf+"/Overall Churn Distribution.png")
plt.close()

#Churn by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Gender ', hue='Churn Value ', data=df, palette='coolwarm')
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title="Churn", labels=["No", "Yes"])
plt.savefig(sf+"/Churn Rate by Gender.png")
plt.close()

#Churn by Contract Type
plt.figure(figsize=(8,5))
sns.countplot(x='Contract       ', hue='Churn Value ', data=df, palette='coolwarm')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.legend(title="Churn", labels=["No", "Yes"])
plt.savefig(sf+"/Churn by Contract Type.png")
plt.close()

#Churn by Monthly Charges
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn Value ', y='Monthly Charges ', data=df, palette='coolwarm')
plt.title('Churn by Monthly Charges')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Monthly Charges')
plt.savefig(sf+"/Churn by Monthly Charges.png")
plt.close()

#Churn by Tenure
plt.figure(figsize=(8,5))
sns.histplot(df[df['Churn Value '] == 1]['Tenure Months '], bins=30, kde=True, color='red', label='Churned')
sns.histplot(df[df['Churn Value '] == 0]['Tenure Months '], bins=30, kde=True, color='blue', label='Stayed')
plt.title('Tenure Distribution (Churned vs. Stayed)')
plt.xlabel('Tenure (Months)')
plt.ylabel('Count')
plt.legend()
plt.savefig(sf+"/Tenure Distribution.png")
plt.close()

#Churn by IST
plt.figure(figsize=(8,5))
sns.countplot(x='Internet Service ', hue='Churn Value ', data=df, palette='coolwarm')
plt.title('Churn by Internet Service Type')
plt.xlabel('Internet Service Type')
plt.ylabel('Count')
plt.legend(title="Churn", labels=["No", "Yes"])
plt.savefig(sf+"/Churn by Internet Service Type.png")
plt.close()

#Chun by Payment Method
plt.figure(figsize=(10,15))
sns.countplot(x='Payment Method            ', hue='Churn Value ', data=df, palette='coolwarm')
plt.xticks(rotation=15)
plt.title('Churn by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.legend(title="Churn", labels=["No", "Yes"])
plt.savefig(sf+"/Churn by Payment Method.png")
plt.close()