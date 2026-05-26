import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Validate dataframe
def validate_dataframe(df):

    if df is None:

        raise ValueError(
            "DataFrame cannot be None"
        )

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "Input must be a pandas DataFrame"
        )

    if df.empty:

        raise ValueError(
            "DataFrame is empty"
        )

    return True


# Clean missing values and duplicates
def clean_data(df):

    validate_dataframe(df)

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    categorical_cols = df.select_dtypes(include="object").columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    return df

# Convert categorical columns into numbers
def encode_data(df):

    validate_dataframe(df)

    data = df.copy()

    for col in data.columns:

        # Encode only non-numeric columns
        if not pd.api.types.is_numeric_dtype(data[col]):

            data[col] = data[col].astype(str)

            le = LabelEncoder()

            data[col] = le.fit_transform(data[col])

    return data


# Split dataset
def split_data(data, target):

    X = data.drop(columns=[target])

    y = data[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# Scale features
def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test


# Train KNN Regressor model
def train_model(X_train, y_train):

    model = KNeighborsRegressor(
        n_neighbors=5
    )

    model.fit(X_train, y_train)

    return model


# Evaluate model
def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    return predictions, mae, mse, rmse, r2