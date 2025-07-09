import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/DanielAmes/telco-applet/main/telco_churn_streamlit_app/telco_churn.csv"
    df = pd.read_csv(url)
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.dropna(inplace=True)
    return df

df = load_data()



def preprocess_features(df):
    df_model = df.copy()
    for col in df_model.select_dtypes(include="object").columns:
        if col != "customerID":
            df_model[col] = LabelEncoder().fit_transform(df_model[col])

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df_model[num_cols] = scaler.fit_transform(df_model[num_cols])
    return df_model





tab1, tab2, tab3 = st.tabs(["Data","Prediction Models", "Visualizations"])

with tab1:
    st.subheader("The Telco Customer Churn Dataset")
    st.markdown("""
<div style="text-align: justify">
Simulated by IBM for the purpose of demonstrating churn analysis, the telco customer churn dataset contains 7043 fictional customers of a fictional telecommunications company. The response variable <i>churn</i> is a binary variable measuring whether or not a customer churned in fixed observational period rather than a continuous variable recording the duration after which a customer churned, so binary prediction models are appropriate.  Because of its origin in simulation, the data is quite clean, and therefore little pre-processing was necessary.
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    if st.checkbox("Show preview of raw data", value=True):
        st.dataframe(df.head())
       
    st.markdown("---")
    
    st.subheader("Variable Descriptions")

    st.markdown("""
- **customerID**: Unique customer identifier  
- **gender**: Customer’s gender (Male/Female)  
- **SeniorCitizen**: 1 if customer is a senior, 0 otherwise  
- **Partner**: Whether the customer has a partner (Yes/No)  
- **Dependents**: Whether the customer has dependents (Yes/No)  
- **tenure**: Number of months the customer has stayed  
- **PhoneService**: Whether the customer has phone service  
- **MultipleLines**: Whether the customer has multiple phone lines  
- **InternetService**: Type of internet service (DSL, Fiber, No)  
- **OnlineSecurity**, **OnlineBackup**, **DeviceProtection**, **TechSupport**: Online services subscribed to  
- **StreamingTV**, **StreamingMovies**: Streaming services subscribed to
- **Contract**: Length of contract (Month-to-month, One year, Two year)  
- **PaperlessBilling**: Whether the customer uses paperless billing  
- **PaymentMethod**: How the customer pays  
- **MonthlyCharges**: Current monthly payment ($)
- **TotalCharges**: Amount paid over tenure ($)
- **Churn**: Whether the customer has churned in the observational period (1 = Yes, 0 = No)
""")


with tab2:
    st.subheader("Select a Prediction Model:")
    model_choice = st.selectbox(
        "",
        ["Lasso Logistic Regression", "Random Forest"]
    )
    @st.cache_resource
    def train_model(df, model_choice):
        df_model = preprocess_features(df)
        X = df_model.drop(columns=["Churn", "customerID"])
        y = df_model["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if model_choice == "Lasso Logistic Regression":
            model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model choice")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return model, df_model, acc, report, cm, X.columns



    model, df_model, acc, report, cm, feature_names = train_model(df, model_choice) # type: ignore

    st.markdown("---")
    st.subheader("Customer Example Selection")

    mode = st.radio("Choose input mode:", ["Select Existing", "Pick Random", "Create New"])

    if mode == "Select Existing":
        customer_id = st.selectbox("Select a customer ID", df["customerID"].unique())
        cust_data = df[df["customerID"] == customer_id]
    elif mode == "Pick Random":
        import random
        random_customer_id = random.choice(df["customerID"].tolist())
        cust_data = df[df["customerID"] == random_customer_id]
        st.write(f"**Randomly selected customer ID:** `{random_customer_id}`")
    elif mode == "Create New":
        st.info("Enter custom values for a new customer:")

        cust_data = df.sample(1).copy()
        cust_data["customerID"] = "custom_001" #placeholder
        cust_data["Churn"] = 0  #placeholder

        cust_data["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
        cust_data["MonthlyCharges"] = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
        cust_data["TotalCharges"] = st.slider("Total Charges ($)", 0.0, 10000.0, 1000.0)

        cust_data["gender"] = st.selectbox("Gender", ["Male", "Female"])
        cust_data["Partner"] = st.selectbox("Partner", ["Yes", "No"])
        cust_data["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
        cust_data["PhoneService"] = st.selectbox("Phone Service", ["Yes", "No"])
        cust_data["MultipleLines"] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        cust_data["InternetService"] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        cust_data["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        cust_data["OnlineBackup"] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        cust_data["DeviceProtection"] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        cust_data["TechSupport"] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        cust_data["StreamingTV"] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        cust_data["StreamingMovies"] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        cust_data["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        cust_data["PaperlessBilling"] = st.selectbox("Paperless Billing", ["Yes", "No"])
        cust_data["PaymentMethod"] = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])


    cust_model_input = preprocess_features(cust_data).drop(columns=["Churn", "customerID"])
    pred = model.predict(cust_model_input)[0]
    prob = model.predict_proba(cust_model_input)[0][1]

    #st.subheader("Customer Information")
    st.write(cust_data)

    st.subheader("Prediction of the Model")
    if pred == 1:
        st.error(f"This customer is likely to **Churn** ({prob:.2%})")
    else:
        st.success(f"This customer is likely to **Stay** ({1 - prob:.2%})")




    st.markdown("---")
    st.subheader("Model Evaluation on Test Set")





    st.write(f"**Accuracy**: {acc:.2%}")
    #st.write("**Classification Report:**")
    #st.json(report)

    st.write("**Confusion Matrix**:")
    st.write(pd.DataFrame(cm,
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"]
    ))



    st.markdown("---")
    if model_choice == "Lasso Logistic Regression":
        st.subheader("Feature Weights (Lasso Coefficients)")
        coefs = pd.Series(model.coef_[0], index=feature_names, name="Coefficient")
        st.write(coefs.sort_values(key=abs, ascending=False))

    elif model_choice == "Random Forest":
        st.subheader("Feature Importance")
        importances = pd.Series(model.feature_importances_, index=feature_names, name="Importance")
        st.bar_chart(importances.sort_values(ascending=False))

with tab3:
    #st.subheader("Churn Rate by Length of Tenure")

    bins_1 = [0, 6, 12, 18, 24, 30, 36, 42, 48, df["tenure"].max()]
    labels_1 = ["0–6","7-12", "13–18", "19-24", "25–30","31-36","37-42","43-48", "49+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins_1, labels=labels_1, right=False)

    churn_by_tenure = df.groupby("tenure_group")["Churn"].mean()

    fig_1, ax_1 = plt.subplots()
    churn_by_tenure.plot(kind="bar", ax=ax_1)
    plt.title("Churn Rate by Tenure")
    plt.ylabel("Churn Rate")
    plt.xlabel("Tenure (months)")
    st.pyplot(fig_1)

    #st.subheader("Churn Rate by Monthly Charge")
    bins_2 = [0, 20, 40, 60, 80, 100, df["MonthlyCharges"].max()]
    labels_2 = ["0–20", "21–40", "41–60", "61–80", "81–100", "101+"]

    df["charge_group"] = pd.cut(df["MonthlyCharges"], bins=bins_2, labels=labels_2, right=False)

    charges_churn = df.groupby("charge_group")["Churn"].mean()

    fig_2, ax_2 = plt.subplots()
    charges_churn.plot(kind="bar", ax=ax_2)
    plt.ylabel("Churn Rate")
    plt.xlabel("Monthly Charges (US$)")
    plt.title("Churn Rate by Monthly Charges")
    st.pyplot(fig_2)


