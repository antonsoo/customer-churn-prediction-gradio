import gradio as gr
import pandas as pd
import joblib
import logging
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the saved model, scaler, and other necessary objects
# (Make sure these files are uploaded to your Hugging Face Spaces repository)
try:
    best_model = joblib.load('model/churn_prediction_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    median_total_charges = joblib.load('model/median_total_charges.pkl')
    X_train = pd.read_csv('model/X_train.csv')  # Load X_train for SHAP explainer

    # Initialize the SHAP explainer (assuming best_model is a tree-based model)
    if hasattr(best_model, "predict_proba"):
        explainer = shap.Explainer(best_model, X_train)
    else:
        explainer = None

    logging.info("Model, scaler, median_total_charges, and X_train loaded successfully.")
except Exception as e:
    logging.error(f"Error loading saved objects: {e}")
    # Handle the error appropriately (e.g., raise an exception or use default values)

# Define the prediction function for Gradio
def predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                  StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    
    # ... (rest of your code) ...

    try:
        # Make predictions using the best model
        prediction = best_model.predict(input_data)[0]
        probability = best_model.predict_proba(input_data)[0][1]

        logging.info(f"Prediction: {prediction}")
        logging.info(f"Probability: {probability}")

        # Generate SHAP explanation if the explainer is loaded
        if explainer is not None:
            try:
                shap_values = explainer(input_data)
                shap_html = shap.force_plot(explainer.expected_value[1], shap_values.values[0][:, 1], input_data, matplotlib=False, show=False, link="logit")
            except Exception as e:
                logging.error(f"Error generating SHAP explanation: {e}")
                shap_html = None
        else:
            shap_html = None

        # Return the prediction, probability, and SHAP explanation
        if prediction == 1:
            return f"Churn: Yes (Probability: {probability:.2f})", shap_html
        else:
            return f"Churn: No (Probability: {probability:.2f})", shap_html

    except Exception as e:
        error_message = f"Error during prediction: {e}"
        logging.error(error_message)
        return error_message, None

# Define the input components for Gradio
inputs = [
    gr.Radio(choices=["Female", "Male"], label="Gender"),
    gr.Number(label="Senior Citizen", minimum=0, maximum=1, step=1),
    gr.Radio(choices=["No", "Yes"], label="Partner"),
    gr.Radio(choices=["No", "Yes"], label="Dependents"),
    gr.Slider(label="Tenure (months)", minimum=0, maximum=72, step=1),
    gr.Radio(choices=["No", "Yes"], label="Phone Service"),
    gr.Radio(choices=["No", "No phone service", "Yes"], label="Multiple Lines"),
    gr.Radio(choices=["DSL", "Fiber optic", "No"], label="Internet Service"),
    gr.Radio(choices=["No", "No internet service", "Yes"], label="Online Security"),
    gr.Radio(choices=["No", "No internet service", "Yes"], label="Online Backup"),
    gr.Radio(choices=["No", "No internet service", "Yes"], label="Device Protection"),
    gr.Radio(choices=["No", "No internet service", "Yes"], label="Tech Support"),
    gr.Radio(choices=["No", "No internet service", "Yes"], label="Streaming TV"),
    gr.Radio(choices=["No", "No internet service", "Yes"], label="Streaming Movies"),
    gr.Radio(choices=["Month-to-month", "One year", "Two year"], label="Contract"),
    gr.Radio(choices=["No", "Yes"], label="Paperless Billing"),
    gr.Radio(choices=["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], label="Payment Method"),
    gr.Number(label="Monthly Charges"),
    gr.Number(label="Total Charges")
]

# Define the output components for Gradio
output1 = gr.Textbox(label="Churn Prediction")
output2 = gr.HTML(label="SHAP Explanation")

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs=[output1, output2],
    title="Telecom Customer Churn Prediction",
    description="Enter customer information to predict churn.",
    live=False,
    examples=[
        # Sample input data (replace with actual data from X_train)
        ['Male', 0, 'Yes', 'No', 5, 'Yes', 'No', 'DSL', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Month-to-month', 'Yes', 'Electronic check', 70.0, 350.0],
        ['Female', 1, 'No', 'No', 10, 'Yes', 'Yes', 'Fiber optic', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'Month-to-month', 'No', 'Mailed check', 95.0, 950.0],
        ['Male', 0, 'No', 'No', 1, 'No', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'No', 'Month-to-month', 'No', 'Electronic check', 25.0, 25.0],
        ['Female', 0, 'Yes', 'Yes', 24, 'Yes', 'No', 'No', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'No internet service', 'Two year', 'No', 'Mailed check', 20.0, 480.0],
        ['Male', 1, 'Yes', 'No', 72, 'Yes', 'Yes', 'Fiber optic', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Two year', 'Yes', 'Credit card (automatic)', 115.0, 8280.0]
    ],
    theme=gr.themes.Soft()
)

# Launch the interface
iface.launch()
