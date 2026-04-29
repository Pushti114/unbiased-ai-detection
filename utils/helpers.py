import pandas as pd
from sklearn.preprocessing import LabelEncoder

def detect_target_type(df, target):
    if df[target].nunique() == 2:
        return "binary"
    elif pd.api.types.is_numeric_dtype(df[target]):
        return "numeric"
    else:
        return "multiclass"

def transform_target(df, target, target_type, binarize_option="median", custom_threshold=None):
    df = df.copy()
    is_binary = False
    if target_type == "binary":
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target])
        is_binary = True
    elif target_type == "numeric":
        if binarize_option == "none":
            is_binary = False
        else:
            if binarize_option == "median":
                threshold = df[target].median()
            elif binarize_option == "mean":
                threshold = df[target].mean()
            elif binarize_option == "custom":
                threshold = custom_threshold if custom_threshold is not None else df[target].median()
            else:
                threshold = df[target].median()
            df[target] = (df[target] > threshold).astype(int)
            is_binary = True
    else:
        most_common = df[target].value_counts().idxmax()
        df[target] = (df[target] == most_common).astype(int)
        is_binary = True
    return df, is_binary