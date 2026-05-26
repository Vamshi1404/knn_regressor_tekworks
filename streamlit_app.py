import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from knn_regressor_analysis import (

    clean_data,

    encode_data,

    split_data,

    scale_data,

    train_model,

    evaluate_model
)

# Streamlit page settings
st.set_page_config(

    page_title=
    "KNN Regressor Demo",

    layout="wide"
)

st.title(
    "Machine Learning "
    "KNN Regressor Demo"
)

# Upload dataset
uploaded_file = st.file_uploader(

    "Upload CSV Dataset",

    type=["csv"]
)

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(
        uploaded_file
    )

    st.header(
        "Dataset Overview"
    )

    # Display dataset
    st.dataframe(
        df.head()
    )

    # Dataset shape
    st.write(
        "Dataset Shape:",
        df.shape
    )

    # Missing values
    st.write(
        "Missing Values"
    )

    st.write(
        df.isnull().sum()
    )

    # Clean dataset
    df = clean_data(df)

    st.success(
        "Data Cleaned Successfully"
    )

    st.header(
        "Exploratory "
        "Data Analysis"
    )

    # Numeric columns
    numeric_cols = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    # Select column
    selected_col = st.selectbox(

        "Select Numeric Column",

        numeric_cols
    )

    # Histogram
    fig, ax = plt.subplots()

    ax.hist(
        df[selected_col],
        bins=20
    )

    ax.set_xlabel(
        selected_col
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    # Correlation matrix
    st.header(
        "Correlation Matrix"
    )

    st.dataframe(
        df[numeric_cols].corr()
    )

    # Select target
    target = st.selectbox(

        "Select Target Column",

        numeric_cols
    )

    # Encode data
    data = encode_data(df)

    # Split data
    X_train, X_test, y_train, y_test = (
        split_data(
            data,
            target
        )
    )

    # Scale data
    X_train, X_test = scale_data(

        X_train,

        X_test
    )

    # Train model
    model, best_params = train_model(

        X_train,

        y_train
    )

    # Evaluate model
    (
        predictions,
        mae,
        mse,
        rmse,
        r2

    ) = evaluate_model(

        model,

        X_test,

        y_test
    )

    # Best hyperparameters
    st.header(
        "Best Hyperparameters"
    )

    st.write(
        best_params
    )

    # Evaluation metrics
    st.header(
        "Model Evaluation"
    )

    st.write(
        f"MAE : {mae:.2f}"
    )

    st.write(
        f"MSE : {mse:.2f}"
    )

    st.write(
        f"RMSE : {rmse:.2f}"
    )

    st.write(
        f"R² Score : {r2:.2f}"
    )

    # Actual vs Predicted
    st.header(
        "Actual vs Predicted"
    )

    fig2, ax2 = plt.subplots()

    ax2.scatter(
        y_test,
        predictions
    )

    ax2.set_xlabel(
        "Actual"
    )

    ax2.set_ylabel(
        "Predicted"
    )

    st.pyplot(fig2)

else:

    st.info(
        "Please Upload "
        "a CSV File"
    )