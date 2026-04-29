def compute_fairness_metrics(df_encoded, target, sensitive_features, sensitive_originals):
    results = {}
    y_true = df_encoded[target]
    for sens in sensitive_features:
        if sens not in sensitive_originals:
            results[sens] = {"warning": f"'{sens}' not found in preprocessed data; skipping."}
            continue

        groups = sensitive_originals[sens].dropna().unique()

        if len(groups) < 2:
            results[sens] = {
                "warning": (
                    f"Only one group found in '{sens}' "
                    f"(value: '{groups[0]}' if len(groups) else 'none'). "
                    "Demographic Parity and Disparate Impact require ≥ 2 groups."
                )
            }
            continue

        if y_true.nunique() <= 1:
            results[sens] = {
                "warning": (
                    "Target column has zero variance (all values identical). "
                    "Fairness metrics are undefined."
                )
            }
            continue

        rates = {}
        for g in groups:
            mask = (sensitive_originals[sens] == g)
            subset_y = y_true[mask]
            if len(subset_y) == 0:
                continue
            rates[str(g)] = round(float(subset_y.mean()), 3)

        if len(rates) < 2:
            results[sens] = {
                "warning": (
                    f"Fewer than 2 non-empty groups found for '{sens}' after "
                    "filtering. Cannot compute parity metrics."
                )
            }
            continue

        max_rate = max(rates.values())
        min_rate = min(rates.values())
        dp_diff = round(max_rate - min_rate, 3)

        if max_rate == 0:
            disparate_impact = 1.0
        else:
            disparate_impact = round(min_rate / max_rate, 3)

        results[sens] = {
            "positive_rates": rates,
            "demographic_parity_difference": dp_diff,
            "disparate_impact": disparate_impact,
            "bias_flag": disparate_impact < 0.8,
            "note": "For Equal Opportunity/Odds, a trained model is required."
        }
    return results