import joblib
import pandas as pd

# Load dataset, model and scaler
df = pd.read_csv('Telco_customer_churn.csv',encoding='utf-8')
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

#Prediction
tenure = 2
monthly_charges = 10.00
total_charges = 20.00

new_customer = pd.DataFrame({'Tenure Months ':[tenure], 'Monthly Charges ':[monthly_charges],
                             'Churn Reason_Competitor offered higher download speeds':[0],
              'Payment Method            _Electronic check          ':[0],
        'Internet Service _Fiber optic      ':[1],'Dependents ':[4], 'CLTV ':[5717],'Contract       _Two year       ':[0],
        'Churn Reason_Attitude of support person':[0],
        'Churn Reason_Competitor offered more data':[0],'Churn Reason_Competitor made better offer':[0],
        'Churn Reason_Product dissatisfaction':[0],
       'Churn Reason_Lack of self-service on Website':[0],
       'Churn Reason_Network reliability':[0],
       'Churn Reason_Service dissatisfaction':[0], 'Latitude  ':[40.584991]})
new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)
print("Predicted Churn:", "Yes" if prediction[0] == 1 else "No")

