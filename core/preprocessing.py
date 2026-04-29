import pandas as pd
from io import BytesIO
import streamlit as st

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def preprocess_data(df_bytes, sensitive_features_tuple, target,
                    impute_num='median', impute_cat='mode'):
    df = pd.read_csv(BytesIO(df_bytes))
    sensitive_features = list(sensitive_features_tuple)

    sensitive_original = {col: df[col].copy() for col in sensitive_features if col in df.columns}

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                if impute_num == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif impute_num == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
            else:
                if impute_cat == 'mode':
                    df[col].fillna(
                        df[col].mode()[0] if not df[col].mode().empty else "Missing",
                        inplace=True
                    )
                else:
                    df[col].fillna("Missing", inplace=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_encode = [c for c in categorical_cols if c != target and c not in sensitive_features]
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    if target not in df_encoded.columns:
        raise ValueError(
            f"Target column '{target}' not found after one-hot encoding. "
            "Ensure the target is not accidentally included in cols_to_encode."
        )

    missing_sensitive = [s for s in sensitive_features if s not in df_encoded.columns]
    if missing_sensitive:
        raise ValueError(
            f"Sensitive feature(s) {missing_sensitive} not found in the encoded "
            "DataFrame. This should not happen — please check for duplicate column "
            "names or data-loading issues."
        )

    return df_encoded, sensitive_original