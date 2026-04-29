import numpy as np
from core.metrics import mutual_info_score_custom

def detect_proxy_features(df, sensitive_features, threshold=0.3, max_features=20):
    proxy_results = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_features = numeric_cols + categorical_cols
    features_to_check = [f for f in all_features if f not in sensitive_features][:max_features]
    for sens in sensitive_features:
        if sens not in df.columns:
            proxy_results[sens] = {}
            continue

        sens_series = df[sens]

        if sens_series.nunique() <= 1:
            proxy_results[sens] = {}
            continue

        associations = {}
        for col in features_to_check:
            if col == sens:
                continue
            col_series = df[col] if col in numeric_cols else df[col].astype(str)

            if col_series.nunique() <= 1:
                continue

            try:
                strength = mutual_info_score_custom(sens_series, col_series)
                if pd.notna(strength) and strength > threshold:
                    associations[col] = round(float(strength), 3)
            except Exception:
                continue
        proxy_results[sens] = associations
    return proxy_results