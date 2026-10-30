
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Configure API URL
API_URL = os.getenv("API_URL")

if not API_URL:
    try:
        API_URL = st.secrets["API_URL"]
    except Exception:
        API_URL = "http://127.0.0.1:8000"

API_URL = API_URL.rstrip("/")


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


# Load dataset from Hugging Face

DATA_URL = (
    "https://huggingface.co/datasets/"
    "harsimrank288/customer-churn-data/"
    "resolve/main/Telco-Customer-Churn.csv"
)

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_URL)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    ).fillna(0)

    return df


try:
    df = load_data()

except Exception as error:
    df = None
    st.error(f"Could not load dataset: {error}")


# --------------------------------
# OVERVIEW PAGE
# --------------------------------
if page == "Overview":

    st.title("📊 Customer Churn Intelligence")
    st.write(
        "An end-to-end Machine Learning application "
        "to analyse customer behaviour and predict churn."
    )

    st.divider()

    if df is not None:

        # Calculate customer statistics
        total_customers = len(df)
        churned_customers = (df["Churn"] == "Yes").sum()
        retained_customers = (df["Churn"] == "No").sum()

        churn_rate = (
            churned_customers / total_customers
        ) * 100

        # KPI Cards
        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

        col2.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

        col3.metric(
            "Retained Customers",
            f"{retained_customers:,}"
        )

        col4.metric(
            "Overall Churn Rate",
            f"{churn_rate:.2f}%"
        )

    else:
        st.warning("Dataset could not be loaded.")
    

    
    # Churn Distribution Chart
    st.subheader("Customer Churn Distribution")

    churn_counts = (
    df["Churn"]
    .map({"Yes": "Churned", "No": "Retained"})
    .value_counts()
    .rename_axis("Customer Status")
    .reset_index(name="Customers")
    )

    fig = px.pie(
    churn_counts,
    names="Customer Status",
    values="Customers",
    hole=0.55,
    title="Churned vs Retained Customers"
)

    fig.update_traces(
    textinfo="percent+label",
    textposition="inside"
)

    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # Technology Stack
    st.subheader("Technology Stack")

    col1, col2, col3 = st.columns(3)

    col1.info("🤖 Machine Learning\n\nLogistic Regression")
    col2.info("⚡ Backend\n\nFastAPI")
    col3.info("🗄️ Database\n\nSQLite")

    st.success(
        "Select Predict Churn from the sidebar "
        "to analyse an individual customer's churn risk."
    )


# --------------------------------
# CHURN INSIGHTS PAGE
# --------------------------------
elif page == "Churn Insights":

    st.header("📊 Customer Churn Insights")

    if df is None:

        st.warning("Please check that the dataset exists.")

    else:

        # ==========================================
        # CUSTOMER OVERVIEW
        # ==========================================

        total_customers = len(df)
        churned_customers = (df["Churn"] == "Yes").sum()
        retained_customers = (df["Churn"] == "No").sum()

        churn_rate = churned_customers / total_customers * 100

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Churned Customers", f"{churned_customers:,}")
        col3.metric("Retained Customers", f"{retained_customers:,}")
        col4.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.divider()

        # Helper function for calculating churn rates
        def calculate_churn_rate(column, data=None):

            if data is None:
                data = df

            return (
                data.groupby(column, observed=True)["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
                .reset_index(name="Churn Rate")
            )

        # Helper function for finding the highest churn category
        def highest_churn(data, category_column):

            return data.loc[data["Churn Rate"].idxmax()]

        # ==========================================
        # SECTION A: CUSTOMER OVERVIEW
        # ==========================================

        st.subheader("📌 Section A: Customer Overview")

        col1, col2 = st.columns(2)

        # Chart 1: Overall Churn Distribution

        with col1:

            st.markdown("### 1. Overall Churn Distribution")

            churn_counts = (
                df["Churn"]
                .value_counts()
                .rename_axis("Churn")
                .reset_index(name="Customers")
            )

            churn_counts["Churn"] = churn_counts["Churn"].map({
                "Yes": "Churned",
                "No": "Retained"
            })

            fig1 = px.pie(
                churn_counts,
                names="Churn",
                values="Customers",
                hole=0.5,
                title="Churned vs Retained Customers"
            )

            fig1.update_traces(
                textinfo="percent+label",
                textposition="inside"
            )

            st.plotly_chart(fig1, use_container_width=True)

            st.info(
                f"💡 {churned_customers:,} customers have left "
                f"the company. The overall churn rate is "
                f"{churn_rate:.2f}%. The company should develop "
                "effective customer retention strategies."
            )

        # Chart 2: Churn by Contract Type

        with col2:

            st.markdown("### 2. Churn by Contract Type")

            contract_churn = calculate_churn_rate("Contract")

            fig2 = px.bar(
                contract_churn,
                x="Contract",
                y="Churn Rate",
                title="Churn Rate by Contract Type",
                text_auto=".2f",
                labels={"Churn Rate": "Churn Rate (%)"}
            )

            st.plotly_chart(fig2, use_container_width=True)

            highest = highest_churn(contract_churn, "Contract")

            st.info(
                f"💡 Customers with {highest['Contract']} "
                f"contracts have the highest churn rate of "
                f"{highest['Churn Rate']:.2f}%. The company "
                "should focus on improving customer engagement."
            )

        st.divider()

        # ==========================================
        # SECTION B: SERVICE AND CONTRACT ANALYSIS
        # ==========================================

        st.subheader("📌 Section B: Service and Contract Analysis")

        col1, col2 = st.columns(2)

        # Chart 3: Churn by Internet Service

        with col1:

            st.markdown("### 3. Churn by Internet Service")

            internet_churn = calculate_churn_rate("InternetService")

            fig3 = px.bar(
                internet_churn,
                x="InternetService",
                y="Churn Rate",
                title="Churn Rate by Internet Service",
                text_auto=".2f",
                labels={
                    "InternetService": "Internet Service",
                    "Churn Rate": "Churn Rate (%)"
                }
            )

            st.plotly_chart(fig3, use_container_width=True)

            highest = highest_churn(internet_churn, "InternetService")

            st.info(
                f"💡 Customers using {highest['InternetService']} "
                f"have the highest churn rate of "
                f"{highest['Churn Rate']:.2f}%. The company "
                "should investigate service quality and pricing."
            )

        # Chart 4: Churn by Payment Method

        with col2:

            st.markdown("### 4. Churn by Payment Method")

            payment_churn = calculate_churn_rate("PaymentMethod")

            fig4 = px.bar(
                payment_churn,
                x="PaymentMethod",
                y="Churn Rate",
                title="Churn Rate by Payment Method",
                text_auto=".2f",
                labels={
                    "PaymentMethod": "Payment Method",
                    "Churn Rate": "Churn Rate (%)"
                }
            )

            fig4.update_layout(xaxis_tickangle=-25)

            st.plotly_chart(fig4, use_container_width=True)

            highest = highest_churn(payment_churn, "PaymentMethod")

            st.info(
                f"💡 Customers using {highest['PaymentMethod']} "
                f"have the highest churn rate of "
                f"{highest['Churn Rate']:.2f}%. The company "
                "should investigate payment-related issues."
            )

        col1, col2 = st.columns(2)

        # Chart 5: Churn by Online Security

        with col1:

            st.markdown("### 5. Churn by Online Security")

            security_churn = calculate_churn_rate("OnlineSecurity")

            fig5 = px.bar(
                security_churn,
                x="OnlineSecurity",
                y="Churn Rate",
                title="Churn Rate by Online Security",
                text_auto=".2f",
                labels={
                    "OnlineSecurity": "Online Security",
                    "Churn Rate": "Churn Rate (%)"
                }
            )

            st.plotly_chart(fig5, use_container_width=True)

            highest = highest_churn(security_churn, "OnlineSecurity")

            st.info(
                f"💡 Customers in the {highest['OnlineSecurity']} "
                f"category have the highest churn rate of "
                f"{highest['Churn Rate']:.2f}%. The company "
                "should review security service adoption."
            )

        # Chart 6: Churn by Tech Support

        with col2:

            st.markdown("### 6. Churn by Tech Support")

            support_churn = calculate_churn_rate("TechSupport")

            fig6 = px.bar(
                support_churn,
                x="TechSupport",
                y="Churn Rate",
                title="Churn Rate by Tech Support",
                text_auto=".2f",
                labels={
                    "TechSupport": "Tech Support",
                    "Churn Rate": "Churn Rate (%)"
                }
            )

            st.plotly_chart(fig6, use_container_width=True)

            highest = highest_churn(support_churn, "TechSupport")

            st.info(
                f"💡 Customers in the {highest['TechSupport']} "
                f"category have the highest churn rate of "
                f"{highest['Churn Rate']:.2f}%. The company "
                "should investigate customer support needs."
            )

        st.divider()

        # ==========================================
        # SECTION C: CUSTOMER BEHAVIOUR ANALYSIS
        # ==========================================

        st.subheader("📌 Section C: Customer Behaviour Analysis")

        col1, col2 = st.columns(2)

        # Chart 7: Customer Tenure by Churn Status

        with col1:

            st.markdown("### 7. Customer Tenure by Churn Status")

            fig7 = px.box(
                df,
                x="Churn",
                y="tenure",
                color="Churn",
                title="Customer Tenure Distribution",
                labels={
                    "Churn": "Customer Churn Status",
                    "tenure": "Tenure (Months)"
                }
            )

            st.plotly_chart(fig7, use_container_width=True)

            average_tenure = df.groupby("Churn")["tenure"].mean()

            st.info(
                f"💡 Churned customers stay for an average of "
                f"{average_tenure['Yes']:.1f} months, compared "
                f"to {average_tenure['No']:.1f} months for "
                "retained customers. Early customer engagement "
                "may help improve retention."
            )

        # Chart 8: Churn by Tenure Group

        with col2:

            st.markdown("### 8. Churn by Tenure Group")

            df_tenure = df.copy()

            df_tenure["Tenure Group"] = pd.cut(
                df_tenure["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=[
                    "0-12 Months",
                    "13-24 Months",
                    "25-48 Months",
                    "49-72 Months"
                ]
            )

            tenure_churn = calculate_churn_rate(
                "Tenure Group",
                data=df_tenure
            )

            fig8 = px.bar(
                tenure_churn,
                x="Tenure Group",
                y="Churn Rate",
                title="Churn Rate Across Tenure Groups",
                text_auto=".2f",
                labels={"Churn Rate": "Churn Rate (%)"}
            )

            st.plotly_chart(fig8, use_container_width=True)

            highest = highest_churn(tenure_churn, "Tenure Group")

            st.info(
                f"💡 Customers in the {highest['Tenure Group']} "
                f"group have the highest churn rate of "
                f"{highest['Churn Rate']:.2f}%. Targeted "
                "retention campaigns may help reduce customer loss."
            )

        col1, col2 = st.columns(2)

        # Chart 9: Monthly Charges by Churn Status

        with col1:

            st.markdown("### 9. Monthly Charges by Churn Status")

            fig9 = px.box(
                df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges Distribution",
                labels={
                    "Churn": "Customer Churn Status",
                    "MonthlyCharges": "Monthly Charges"
                }
            )

            st.plotly_chart(fig9, use_container_width=True)

            average_charges = df.groupby("Churn")["MonthlyCharges"].mean()

            st.info(
                f"💡 Churned customers pay an average of "
                f"{average_charges['Yes']:.2f} per month, "
                f"compared to {average_charges['No']:.2f} "
                "for retained customers. The company should "
                "review pricing and available packages."
            )

        # Chart 10: Churn by Senior Citizen Status

        with col2:

            st.markdown("### 10. Churn by Senior Citizen Status")

            df_senior = df.copy()

            df_senior["Customer Group"] = df_senior[
                "SeniorCitizen"
            ].map({
                0: "Non-Senior Citizens",
                1: "Senior Citizens"
            })

            senior_churn = (
                df_senior.groupby("Customer Group", observed=True)["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
                .reset_index(name="Churn Rate")
            )

            fig10 = px.bar(
                senior_churn,
                x="Customer Group",
                y="Churn Rate",
                title="Churn Rate by Customer Age Category",
                text_auto=".2f",
                labels={"Churn Rate": "Churn Rate (%)"}
            )

            st.plotly_chart(fig10, use_container_width=True)

            highest = highest_churn(senior_churn, "Customer Group")

            st.info(
                f"💡 {highest['Customer Group']} have the "
                f"higher churn rate of "
                f"{highest['Churn Rate']:.2f}%. The company "
                "should understand the needs of different "
                "customer groups."
            )

        # ==========================================
        # FINAL BUSINESS NOTE
        # ==========================================

        st.divider()

        st.success(
            "📈 Analysis Complete! These visualisations help "
            "identify customer groups associated with higher "
            "churn rates and support data-driven retention strategies."
        )

        st.caption(
            "Note: These charts show associations in historical "
            "data. They do not prove that a particular customer "
            "characteristic causes churn."
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

