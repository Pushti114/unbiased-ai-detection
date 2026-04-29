import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.size == 0:
        return 0.0
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0.0
    r, k = confusion_matrix.shape
    if r < 2 or k < 2:
        return 0.0
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else 0.0

def mutual_info_score_custom(x, y):
    if pd.api.types.is_numeric_dtype(x) and x.nunique() <= 1:
        return 0.0
    if pd.api.types.is_numeric_dtype(y) and y.nunique() <= 1:
        return 0.0
    if not pd.api.types.is_numeric_dtype(x) and x.nunique() <= 1:
        return 0.0
    if not pd.api.types.is_numeric_dtype(y) and y.nunique() <= 1:
        return 0.0

    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        corr = x.corr(y)
        return abs(corr) if pd.notna(corr) else 0.0
    elif pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_numeric_dtype(y):
        y_enc = LabelEncoder().fit_transform(y.astype(str))
        mi = mutual_info_classif(x.values.reshape(-1, 1), y_enc, discrete_features=False)[0]
        counts = np.bincount(y_enc)
        probs = counts / counts.sum()
        ent = stats.entropy(probs)
        return mi / ent if ent > 0 else 0.0
    elif not pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        return mutual_info_score_custom(y, x)
    else:
        return cramers_v(x, y)