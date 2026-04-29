import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from io import BytesIO

from core.preprocessing import load_data, preprocess_data
from core.fairness import compute_fairness_metrics
from core.proxy_detection import detect_proxy_features
from core.metrics import cramers_v, mutual_info_score_custom
from utils.helpers import detect_target_type, transform_target


st.set_page_config(page_title="Unbiased AI Detection", layout="wide")

if "bias_results" not in st.session_state:
    st.session_state.bias_results = None
if "proxy_results" not in st.session_state:
    st.session_state.proxy_results = None
if "fairness_metrics" not in st.session_state:
    st.session_state.fairness_metrics = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "sensitive_cols" not in st.session_state:
    st.session_state.sensitive_cols = None

# =========================
# FEATURE 3 — VISUALIZATIONS
# =========================

def plot_group_rates(rates_dict, title, bias_flag):
    fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(rates_dict))))
    groups = list(rates_dict.keys())
    values = list(rates_dict.values())
    bar_color = "#E24B4A" if bias_flag else "#1D9E75"
    bars = ax.barh(groups, values, color=bar_color, height=0.5)
    ax.set_xlabel("Positive Rate")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, max(values) * 1.25 if values else 1)
    if values:
        threshold_line = 0.8 * max(values)
        ax.axvline(threshold_line, color="#BA7517", linestyle="--",
                   linewidth=1.2, label=f"80% rule ({threshold_line:.2f})")
        ax.legend(fontsize=8)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_fairness_summary(fairness_metrics):
    features, di_vals, dp_vals = [], [], []
    for sens, m in fairness_metrics.items():
        if "warning" in m:
            continue
        features.append(sens)
        di_vals.append(m["disparate_impact"])
        dp_vals.append(m["demographic_parity_difference"])
    if not features:
        return
    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(max(5, len(features) * 1.4), 3.5))
    ax.bar(x - 0.2, di_vals, 0.35, label="Disparate Impact", color="#378ADD")
    ax.bar(x + 0.2, dp_vals, 0.35, label="DP Difference", color="#D85A30")
    ax.axhline(0.8, color="#BA7517", linestyle="--", linewidth=1, label="DI threshold (0.8)")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Fairness metrics summary", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_proxy_heatmap(proxy_dict):
    data = []
    for sens, assoc in proxy_dict.items():
        for feat, val in assoc.items():
            data.append([sens, feat, val])
    if not data:
        st.info("No strong proxy associations found.")
        return
    df_assoc = pd.DataFrame(data, columns=["Sensitive Feature", "Proxy Feature", "Strength"])
    pivot = df_assoc.pivot(
        index="Sensitive Feature", columns="Proxy Feature", values="Strength"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2),
                                    max(3, pivot.shape[0] * 1.0)))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Association strength", "shrink": 0.7}
    )
    ax.set_title("Proxy feature associations with sensitive attributes", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =========================
# FEATURE 4 — ACTIONABLE NEXT STEPS
# =========================
def show_actionable_next_steps(bias_results, proxy_results, fairness_metrics, target_col):
    st.subheader("🛠️ Actionable Next Steps")

    any_action = False

    biased_di      = [f for f, d in (bias_results or {}).items() if d.get("bias")]
    biased_dp      = [f for f, m in (fairness_metrics or {}).items()
                      if m.get("demographic_parity_difference", 0) > 0.1]
    proxy_found    = {s: list(a.keys())
                      for s, a in (proxy_results or {}).items() if a}

    if biased_di or biased_dp:
        any_action = True
        affected = list(set(biased_di + biased_dp))
        with st.expander("⚖️ Reweighing  (pre-processing — apply before training)"):
            st.markdown(
                f"**Triggered by:** Disparate Impact < 0.8 or Demographic Parity Difference > 0.1  \n"
                f"**Affected features:** `{'`, `'.join(affected)}`\n\n"
                "Reweighing assigns per-sample weights so that every "
                "group × label combination is equally represented in training. "
                "No data is removed or modified — only weights change."
            )
            st.code(f"""\
# pip install aif360
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

dataset = BinaryLabelDataset(
    df=df_train,
    label_names=["{target_col}"],
    protected_attribute_names={affected}
)

rw = Reweighing(
    unprivileged_groups=[{{"{affected[0]}": 0}}],
    privileged_groups  =[{{"{affected[0]}": 1}}]
)
dataset_rw = rw.fit_transform(dataset)

model.fit(X_train, y_train,
          sample_weight=dataset_rw.instance_weights)
""", language="python")
            st.caption(
                "📦 Library: `pip install aif360`  |  "
                "📖 [AIF360 docs](https://aif360.readthedocs.io/)"
            )

    if biased_di:
        any_action = True
        with st.expander("🔧 Disparate Impact Remover  (pre-processing — transform features)"):
            st.markdown(
                f"**Triggered by:** Disparate Impact < 0.8  \n"
                f"**Affected features:** `{'`, `'.join(biased_di)}`\n\n"
                "Repairs numeric feature distributions to be more uniform across "
                "groups while preserving rank order within each group. "
                "`repair_level=1.0` gives full repair; lower values are a softer fix."
            )
            st.code(f"""\
# pip install aif360
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset

dataset = BinaryLabelDataset(
    df=df_train,
    label_names=["{target_col}"],
    protected_attribute_names={biased_di}
)

di_remover = DisparateImpactRemover(repair_level=0.8)
dataset_repaired = di_remover.fit_transform(dataset)

df_repaired = dataset_repaired.convert_to_dataframe()[0]
""", language="python")
            st.caption(
                "📦 Library: `pip install aif360`  |  "
                "📖 [AIF360 docs](https://aif360.readthedocs.io/)"
            )

    if proxy_found:
        any_action = True
        with st.expander("🚫 Remove or decorrelate proxy features  (pre-processing — feature selection)"):
            for sens, cols in proxy_found.items():
                st.markdown(
                    f"**Sensitive feature:** `{sens}`  \n"
                    f"**Proxy columns:** `{'`, `'.join(cols)}`\n\n"
                    f"These columns are strongly associated with `{sens}` and can allow "
                    "indirect discrimination even when the sensitive column itself is excluded."
                )
            st.markdown("**Option 1 — Drop proxies (simplest):**")
            all_proxy_cols = list({c for cols in proxy_found.values() for c in cols})
            st.code(f"df = df.drop(columns={all_proxy_cols})", language="python")
            st.markdown("**Option 2 — Decorrelate (preserve signal, remove sensitive component):**")
            first_sens = list(proxy_found.keys())[0]
            first_cols = proxy_found[first_sens]
            st.code(f"""\
from sklearn.linear_model import LinearRegression

for col in {first_cols}:
    reg = LinearRegression().fit(df[["{first_sens}"]], df[col])
    df[col] = df[col] - reg.predict(df[["{first_sens}"]])
""", language="python")
            st.caption(
                "Audit before dropping — confirm the proxy carries no legitimate "
                "predictive signal independent of the sensitive attribute."
            )

    if biased_dp:
        any_action = True
        with st.expander("🎯 Per-group threshold optimisation  (post-processing — no retraining needed)"):
            st.markdown(
                f"**Triggered by:** Demographic Parity Difference > 0.1  \n"
                f"**Affected features:** `{'`, `'.join(biased_dp)}`\n\n"
                "After training, apply different decision thresholds per group "
                "so that positive prediction rates are equalised. "
                "No retraining required."
            )
            st.code(f"""\
# pip install aif360
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

cpp = CalibratedEqOddsPostprocessing(
    unprivileged_groups=[{{"{biased_dp[0]}": 0}}],
    privileged_groups  =[{{"{biased_dp[0]}": 1}}],
    cost_constraint="fnr"
)
cpp.fit(dataset_val_true, dataset_val_pred)
dataset_fair = cpp.predict(dataset_test_pred)
""", language="python")
            st.caption(
                "📦 Library: `pip install aif360`  |  "
                "📖 [AIF360 docs](https://aif360.readthedocs.io/)"
            )

    if not any_action:
        st.success("✅ No significant bias detected. No remediation steps needed at this time.")

# =========================
# MAIN APP
# =========================
st.title("🔍 Unbiased AI Detection System")
st.markdown("Upload a dataset to detect potential bias and proxy features.")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Data Preprocessing")
    impute_num = st.selectbox("Numeric imputation", ["median", "mean", "zero"], index=0)
    impute_cat = st.selectbox("Categorical imputation", ["mode", "missing"], index=0)

    st.subheader("🔐 Gemini API (Optional)")
    use_gemini = st.checkbox("Enable Gemini Explanation")
    api_key = None
    if use_gemini:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            st.success("API key set")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Column", columns)
    sensitive_features = st.multiselect("Select Sensitive Features", columns)

    target_type = detect_target_type(df, target_column)
    st.write(f"Detected target type: **{target_type}**")

    binarize_option = "none"
    custom_threshold = None
    if target_type == "numeric":
        binarize_choice = st.radio(
            "How to handle numeric target?",
            ["Keep as regression (no fairness metrics for classification)",
             "Binarize for classification"],
            index=1
        )
        if binarize_choice.startswith("Keep"):
            binarize_option = "none"
        else:
            threshold_method = st.selectbox("Threshold method", ["median", "mean", "custom"])
            if threshold_method == "custom":
                custom_threshold = st.number_input(
                    "Enter threshold value", value=float(df[target_column].median())
                )
            binarize_option = threshold_method
    elif target_type == "multiclass":
        st.info("Multiclass target will be binarized: most frequent class vs rest.")

    if st.button("🚀 Run Bias Detection"):
        if not sensitive_features:
            st.error("Please select at least one sensitive feature.")
        else:
            with st.spinner("Processing data and computing metrics…"):

                df_transformed, is_binary_target = transform_target(
                    df, target_column, target_type, binarize_option, custom_threshold
                )
                st.session_state.target_col      = target_column
                st.session_state.sensitive_cols  = sensitive_features

                csv_bytes = uploaded_file.getvalue()
                try:
                    df_encoded, sensitive_originals = preprocess_data(
                        csv_bytes,
                        tuple(sensitive_features),
                        target_column,
                        impute_num=impute_num,
                        impute_cat=impute_cat
                    )
                except ValueError as exc:
                    st.error(f"❌ Preprocessing failed: {exc}")
                    st.stop()

                df_encoded, _ = transform_target(
                    df_encoded, target_column, target_type, binarize_option, custom_threshold
                )
                st.session_state.df_processed = df_encoded

                if is_binary_target:
                    fairness_metrics = compute_fairness_metrics(
                        df_encoded, target_column, sensitive_features, sensitive_originals
                    )
                    st.session_state.fairness_metrics = fairness_metrics
                else:
                    st.warning("Target is numeric (regression). Classification fairness metrics not computed.")
                    st.session_state.fairness_metrics = None

                proxy_results = detect_proxy_features(df_encoded, sensitive_features)
                st.session_state.proxy_results = proxy_results

                if is_binary_target and st.session_state.fairness_metrics:
                    bias_results = {}

                    for sens, metrics in st.session_state.fairness_metrics.items():
                        if "positive_rates" in metrics:
                            bias_results[sens] = {
                                "rates": metrics["positive_rates"],
                                "disparate_impact": metrics["disparate_impact"],
                                "bias": metrics["bias_flag"]
                            }

                    st.session_state.bias_results = bias_results
                else:
                    st.session_state.bias_results = None

            st.success("Analysis complete!")

    if st.session_state.fairness_metrics or st.session_state.proxy_results:
        st.header("📊 Analysis Results")

        if st.session_state.fairness_metrics:
            st.subheader("Fairness Metrics (Classification)")

            plot_fairness_summary(st.session_state.fairness_metrics)

            for sens, metrics in st.session_state.fairness_metrics.items():
                if "warning" in metrics:
                    st.warning(f"{sens}: {metrics['warning']}")
                    continue

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{sens}**")
                    st.metric("Demographic Parity Diff",
                              metrics["demographic_parity_difference"])
                    st.metric("Disparate Impact", metrics["disparate_impact"])
                    if metrics["bias_flag"]:
                        st.error("⚠️ Potential bias detected")
                    else:
                        st.success("✅ No significant bias")
                with col2:
                    plot_group_rates(
                        metrics["positive_rates"],
                        f"Positive Rates by {sens}",
                        metrics["bias_flag"]
                    )
                st.markdown("---")

        if st.session_state.proxy_results:
            st.subheader("🔗 Proxy Feature Detection")
            if any(st.session_state.proxy_results.values()):
                plot_proxy_heatmap(st.session_state.proxy_results)
                with st.expander("View raw proxy associations"):
                    st.json(st.session_state.proxy_results)
            else:
                st.info("No strong proxy associations found (threshold > 0.3).")

        show_actionable_next_steps(
            st.session_state.bias_results,
            st.session_state.proxy_results,
            st.session_state.fairness_metrics,
            st.session_state.target_col or target_column
        )

        if use_gemini and api_key:
            st.subheader("🤖 Gemini AI Explanation")
            if st.button("Generate Explanation"):
                if not st.session_state.fairness_metrics and not st.session_state.proxy_results:
                    st.error("No results to explain. Run bias detection first.")
                else:
                    with st.spinner("Generating explanation…"):
                        try:
                            from google import genai
                            client = genai.Client(api_key=api_key)
                            prompt = f"""
You are an AI fairness expert. Analyze the following bias detection results.

Fairness Metrics:
{st.session_state.fairness_metrics if st.session_state.fairness_metrics else 'Not computed (regression target)'}

Proxy Associations:
{st.session_state.proxy_results}

Explain:
- What biases exist and which groups are affected.
- Which features act as proxies for sensitive attributes.
- Why these biases may occur.
- Actionable steps to mitigate bias (reweighing, disparate impact remover, feature removal).
Keep the explanation concise and non-technical where possible.
"""
                            response = client.models.generate_content(
                                model="gemma-4-31b-it",
                                contents=prompt
                            )
                            st.markdown("### 📄 Explanation")
                            st.write(response.text)
                        except Exception as e:
                            st.error(f"Error generating explanation: {e}")

else:
    st.info("👆 Please upload a CSV file to begin.")