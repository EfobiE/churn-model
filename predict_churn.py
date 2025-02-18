import joblib
import pandas as pd

# Load dataset, model and scaler
df = pd.read_csv('Telco_customer_churn.csv',encoding='utf-8')
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

#Prediction
tenure = 2
monthly_charges = 40.00
total_charges = 200.00

new_customer = pd.DataFrame({'Tenure Months ': [tenure], 'Monthly Charges ': [monthly_charges], df.columns[27]: [total_charges]})

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)
print("Predicted Churn:", "Yes" if prediction[0] == 1 else "No")

