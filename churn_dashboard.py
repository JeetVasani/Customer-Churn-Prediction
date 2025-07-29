import streamlit as st
import requests

st.title("Customer Churn Prediction")

st.write("Enter customer features (in the same order as the dataset columns):")
st.write("The features should be in the following order: ")
st.write("call_failure, complains, subscription_length, charge_amount, seconds_of_use, frequency_of_use, frequency_of_sms, distinct_called_numbers, age_group, tariff_plan, status, age, customer_value")

features_input = st.text_area(
    "Enter features",
    placeholder="e.g., 8, 0, 38, 0, 4370, 71, 5, 17, 3, 1, 1, 30, 197.64"
)

if st.button("Predict Churn"):
    if features_input:
        try:
            features_list = [float(x.strip()) for x in features_input.split(",")]
            if len(features_list) != 13:
                st.warning("Please enter exactly 13 features.")

            else:
                response = requests.post(
                    "http://127.0.0.1:8000/predict/",
                    json=dict(zip(
                        ["call_failure", "complains", "subscription_length", "charge_amount", "seconds_of_use",
                         "frequency_of_use", "frequency_of_sms", "distinct_called_numbers", "age_group", "tariff_plan",
                         "status", "age", "customer_value"], features_list)
                    )
                )
                if response.status_code == 200:
                    prediction = response.json()
                    st.write(f"Churn Prediction: {'Churn' if prediction['Churn Prediction'] == '1' else 'No Churn'}")
                else:
                    st.error(f"Error: Unable to get prediction. Status code: {response.status_code}")
        except ValueError:
            st.warning("Please enter valid numerical values for all features.")
    else:
        st.warning("Please enter the customer features.")