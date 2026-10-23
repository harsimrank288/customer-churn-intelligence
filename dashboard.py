
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# Page configuration
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Churn Insights",
        "Predict Churn",
        "Prediction History"
    ]
)


# Load the original dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    ).fillna(0)

    return df

try:
    df = load_data()
except FileNotFoundError:
    df = None
    st.error(f"Dataset not found at: {DATA_PATH}")

st.title("📊 Customer Churn Prediction & Retention Intelligence")
st.write(
    "Predict customer churn risk using Machine Learning "
    "and explore customer retention insights."
)



# --------------------------------
# OVERVIEW PAGE
# --------------------------------
if page == "Overview":
    st.header("Project Overview")

    st.write(
        "This application uses Machine Learning to estimate "
        "whether a customer may leave a company."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Model", "Logistic Regression")
    col2.metric("Backend", "FastAPI")
    col3.metric("Database", "SQLite")

    st.info("Select Predict Churn from the sidebar to begin.")
# --------------------------------
# CHURN INSIGHTS PAGE
# --------------------------------
elif page == "Churn Insights":
    st.header("📊 Customer Churn Insights")

    if df is None:
        st.warning("Please check that the dataset exists in the data folder.")

    else:
        # Calculate churn statistics
        total_customers = len(df)
        churned_customers = (df["Churn"] == "Yes").sum()
        retained_customers = (df["Churn"] == "No").sum()
        churn_rate = churned_customers / total_customers * 100

        # Summary cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churned Customers", f"{churned_customers:,}")
        col3.metric("Retained Customers", f"{retained_customers:,}")
        col4.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.divider()

        # Churn distribution
        st.subheader("1. Overall Churn Distribution")

        churn_counts = (
            df["Churn"]
            .value_counts()
            .rename_axis("Churn")
            .reset_index(name="Customers")
        )

        fig1 = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            title="Customers Who Left vs Stayed",
            hole=0.4
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Churn by contract
        st.subheader("2. Churn by Contract Type")

        contract_churn = (
            df.groupby("Contract", observed=True)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig2 = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate",
            title="Churn Rate by Contract Type",
            labels={"Churn Rate": "Churn Rate (%)"},
            text_auto=".2f"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Churn by internet service
        st.subheader("3. Churn by Internet Service")

        internet_churn = (
            df.groupby("InternetService", observed=True)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig3 = px.bar(
            internet_churn,
            x="InternetService",
            y="Churn Rate",
            title="Churn Rate by Internet Service",
            labels={
                "InternetService": "Internet Service",
                "Churn Rate": "Churn Rate (%)"
            },
            text_auto=".2f"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Churn by payment method
        st.subheader("4. Churn by Payment Method")

        payment_churn = (
            df.groupby("PaymentMethod", observed=True)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig4 = px.bar(
            payment_churn,
            x="PaymentMethod",
            y="Churn Rate",
            title="Churn Rate by Payment Method",
            labels={
                "PaymentMethod": "Payment Method",
                "Churn Rate": "Churn Rate (%)"
            },
            text_auto=".2f"
        )

        fig4.update_layout(xaxis_tickangle=-25)

        st.plotly_chart(fig4, use_container_width=True)

        # Tenure versus churn
        st.subheader("5. Customer Tenure by Churn Status")

        fig5 = px.box(
            df,
            x="Churn",
            y="tenure",
            color="Churn",
            title="How Long Do Customers Stay?",
            labels={
                "Churn": "Customer Churn Status",
                "tenure": "Tenure (Months)"
            }
        )

        st.plotly_chart(fig5, use_container_width=True)

        # Monthly charges versus churn
        st.subheader("6. Monthly Charges by Churn Status")

        fig6 = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges: Churned vs Retained Customers",
            labels={
                "Churn": "Customer Churn Status",
                "MonthlyCharges": "Monthly Charges"
            }
        )

        st.plotly_chart(fig6, use_container_width=True)

        st.info(
            "Business insight: These charts show associations in the "
            "historical dataset. They do not prove that a particular "
            "feature causes customer churn."
        )

# --------------------------------
# PREDICT CHURN PAGE
# --------------------------------
elif page == "Predict Churn":
    st.header("🔍 Predict Customer Churn")

    st.write("Enter the customer's details below.")

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )

            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:

        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=customer_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                st.subheader("Prediction Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Predicted Churn",
                    result["predicted_churn"]
                )

                col2.metric(
                    "Churn Probability",
                    f'{result["churn_probability"]}%'
                )

                col3.metric(
                    "Risk Level",
                    result["risk_level"]
                )

                if result["risk_level"] == "High":
                   st.error("🔴 High Churn Risk") 
                   
                   st.subheader("Recommended Retention Actions") 
                   st.write("- Review the customer's service experience.") 
                   st.write("- Consider a personalized retention offer.") 
                   st.write("- Contact the customer to understand their concerns.")
                elif result["risk_level"] == "Medium":
                    st.warning("🟡 Medium Churn Risk") 

                    st.subheader("Recommended Retention Actions") 
                    st.write("- Monitor customer engagement.") 
                    st.write("- Review contract and service preferences.") 
                    st.write("- Consider a customer satisfaction follow-up.")
                else:
                    st.success("🟢 Low Churn Risk") 
                    st.subheader("Recommended Actions") 
                    st.write("- Maintain consistent service quality.") 
                    st.write("- Continue regular customer engagement.")

            else:
                st.error(
                    f"API error ({response.status_code}): "
                    f"{response.text}"
                )

        except requests.RequestException as error:
            st.error(
                "Could not connect to the FastAPI backend. "
                "Make sure the API is running."
            )
            st.caption(str(error))

# --------------------------------
# PREDICTION HISTORY PAGE
# --------------------------------
elif page == "Prediction History":
    st.header("📜 Prediction History")

    try:
        response = requests.get(
            f"{API_URL}/history",
            timeout=15
        )

        if response.status_code == 200:
            history = response.json()

            st.metric(
                "Total Predictions",
                history["total_predictions"]
            )

            if history["predictions"]:
                history_df = pd.DataFrame(history["predictions"])

                # Convert stored customer JSON into separate columns
                if "customer_data" in history_df.columns:
                    customer_details = pd.json_normalize(
                        history_df["customer_data"].apply(
                            lambda value: (
                                json.loads(value)
                                if isinstance(value, str)
                                else value
                            )
                        )
                    )

                    history_df = pd.concat(
                        [
                            history_df.drop(columns=["customer_data"]),
                            customer_details
                        ],
                        axis=1
                    )

                # Select useful columns for display
                preferred_columns = [
                    "id",
                    "created_at",
                    "gender",
                    "tenure",
                    "Contract",
                    "MonthlyCharges",
                    "predicted_churn",
                    "churn_probability",
                    "risk_level"
                ]

                available_columns = [
                    column for column in preferred_columns
                    if column in history_df.columns
                ]

                history_df = history_df[available_columns]

                # Rename columns for readability
                history_df = history_df.rename(
                    columns={
                        "id": "Prediction ID",
                        "created_at": "Date",
                        "gender": "Gender",
                        "tenure": "Tenure (Months)",
                        "Contract": "Contract Type",
                        "MonthlyCharges": "Monthly Charges",
                        "predicted_churn": "Predicted Churn",
                        "churn_probability": "Churn Probability (%)",
                        "risk_level": "Risk Level"
                    }
                )

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Download prediction history
                csv_data = history_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="📥 Download Prediction History",
                    data=csv_data,
                    file_name="customer_churn_prediction_history.csv",
                    mime="text/csv"
                )

            else:
                st.info("No predictions have been saved yet.")

        else:
            st.error(
                f"Could not load history: {response.status_code}"
            )

    except requests.RequestException:
        st.error(
            "Could not connect to the API. "
            "Make sure FastAPI is running."
        )

