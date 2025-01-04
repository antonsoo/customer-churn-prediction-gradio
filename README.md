# Telecom Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. Customer churn, the rate at which customers stop doing business with a company, is a critical metric for businesses as it is often more cost-effective to retain existing customers than to acquire new ones.

The primary goal of this project is to develop a predictive model that can accurately identify customers who are likely to churn. This enables the telecommunications company to take proactive steps to retain these customers, such as offering targeted promotions, improving customer service, or adjusting pricing plans.

This project utilizes the Telco Customer Churn dataset from Kaggle, and it implements an end-to-end machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and deployment.

**Key Features:**

*   **Interactive Gradio App:** A user-friendly web application built with Gradio that allows users to input customer data and receive churn predictions in real time.
*   **Model Interpretability:** Uses SHAP (SHapley Additive exPlanations) values to provide insights into the factors driving each prediction, making the model more transparent and understandable.
*   **Deployment on Hugging Face Spaces:** The Gradio app is deployed on Hugging Face Spaces, making it easily accessible for demonstration and testing.
*   **Comprehensive Model Evaluation:** The project includes a thorough evaluation of different machine learning models using various metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
*   **Engineered Features:** The project involves creating new features from the existing data to potentially improve the model's predictive power.

## Dataset

The project uses the **Telco Customer Churn** dataset from Kaggle: [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). This dataset contains information about customer demographics, services used, contract details, and whether or not the customer churned.

**Here is a brief description of the key features in the dataset:**

*   **customerID:** Unique identifier for each customer.
*   **gender:** Customer's gender (Male, Female).
*   **SeniorCitizen:** Whether the customer is a senior citizen (0: No, 1: Yes).
*   **Partner:** Whether the customer has a partner (Yes, No).
*   **Dependents:** Whether the customer has dependents (Yes, No).
*   **tenure:** Number of months the customer has stayed with the company.
*   **PhoneService:** Whether the customer has a phone service (Yes, No).
*   **MultipleLines:** Whether the customer has multiple phone lines (Yes, No, No phone service).
*   **InternetService:** Type of internet service (DSL, Fiber optic, No).
*   **OnlineSecurity:** Whether the customer has online security (Yes, No, No internet service).
*   **OnlineBackup:** Whether the customer has online backup (Yes, No, No internet service).
*   **DeviceProtection:** Whether the customer has device protection (Yes, No, No internet service).
*   **TechSupport:** Whether the customer has tech support (Yes, No, No internet service).
*   **StreamingTV:** Whether the customer has streaming TV (Yes, No, No internet service).
*   **StreamingMovies:** Whether the customer has streaming movies (Yes, No, No internet service).
*   **Contract:** Type of contract (Month-to-month, One year, Two year).
*   **PaperlessBilling:** Whether the customer uses paperless billing (Yes, No).
*   **PaymentMethod:** Payment method used by the customer (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
*   **MonthlyCharges:** The amount charged to the customer monthly.
*   **TotalCharges:** The total amount charged to the customer.
*   **Churn:** Whether the customer churned (Yes, No) - this is the target variable.

## Methodology

The project follows a standard machine learning workflow:

1. **Data Loading and Preprocessing:**
    *   The Telco Customer Churn dataset is loaded from a CSV file.
    *   Data cleaning steps are performed to handle missing values (if any) and ensure data quality.
    *   `TotalCharges` column is converted to numeric type.
    *   The `customerID` column is dropped as it is not relevant for modeling.

2. **Feature Engineering:**
    *   New features are engineered to potentially improve model performance:
        *   `Tenure_to_MonthlyCharges_Ratio`
        *   `TotalCharges_to_MonthlyCharges_Ratio`
        *   `TotalCharges_to_Tenure_Ratio`

3. **Data Splitting:**
    *   The dataset is split into training and testing sets using an 80/20 split ratio.

4. **One-Hot Encoding:**
    *   Categorical features are one-hot encoded to convert them into a numerical representation that can be used by machine learning models.

5. **Feature Scaling:**
    *   Numerical features are scaled using `StandardScaler` to ensure that they have a similar range of values.

6. **Model Selection and Training:**
    *   Three different classification models are trained and evaluated:
        *   Logistic Regression
        *   Random Forest
        *   XGBoost
    *   `GridSearchCV` is used to find the best hyperparameters for each model based on 5-fold cross-validation.

7. **Model Evaluation:**
    *   The best-performing model (based on cross-validation scores) is selected for further evaluation.
    *   The model is evaluated on the training set using metrics like precision, recall, F1-score, and confusion matrix.

8. **Deployment with Gradio:**
    *   An interactive web application is built using Gradio to allow users to input customer data and receive churn predictions.
    *   The Gradio app includes a feature to generate SHAP (SHapley Additive exPlanations) values, providing interpretability to the model's predictions.

## Results

The XGBoost model achieved the highest cross-validation score during training and was selected as the best model.

**Training Set Evaluation:**

*   **Precision:** \[Insert Precision Score Here]
*   **Recall:** \[Insert Recall Score Here]
*   **F1-score:** \[Insert F1-Score Here]
*   **Accuracy:** \[Insert Accuracy Score Here]

A detailed classification report and confusion matrix are also provided in the code.

**Feature Importances:**

The feature importances from the XGBoost model are displayed in the output of Cell 10. The most important features for churn prediction were found to be \[List the top 5-10 most important features].

**SHAP Explanations:**

The Gradio app includes a SHAP plot that visualizes the contribution of each feature to individual predictions. This allows users to understand why the model made a particular prediction for a given customer.

## Installation

To run this project, you will need to install the following Python libraries:
  ```bash
  pip install pandas scikit-learn xgboost joblib gradio shap
  ```

You will also need to download the Telco Customer Churn dataset from Kaggle and place it in the data directory.

## Usage

1. Clone the repository:
  ```bash
  git clone https://github.com/antonsoo/customer-churn-prediction-gradio
  cd https://github.com/antonsoo/customer-churn-prediction-gradio
  ```

  
