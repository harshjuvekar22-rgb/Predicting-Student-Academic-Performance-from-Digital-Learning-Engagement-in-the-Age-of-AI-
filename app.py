"""Streamlit app: digital learning engagement + AI use -> predicted academic performance."""
import numpy as np
import pandas as pd
import streamlit as st

from edu_model import (GROUPS, MODEL_NAMES, REPORT_DIR, ROOT, TARGET, compare_categories, load_data,
                       load_or_train, suggest_gains, sweep_numeric)

st.set_page_config(page_title="Digital Engagement & AI – Performance Predictor", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run may retrain, ~1 min)…")
def get_models():
    return load_or_train()


models, metrics = get_models()
numeric, categorical = metrics["numeric"], metrics["categorical"]
FEATS = numeric + categorical
RISK = metrics.get("risk_threshold", 55.0)


@st.cache_data
def get_data(rel):
    p = ROOT / rel
    return load_data(p if p.exists() else None)


df = get_data(metrics["data_file"])

st.title("🎓 Predicting Student Academic Performance from Digital Learning Engagement in the Age of AI")
st.caption("Regression + deep-learning models. ⚠️ If you are using the bundled dataset, it is **synthetic** "
           "(OULAD-style) and meant for demonstration; do not use predictions for real decisions.")


# ------------------------------------------------------------------ sidebar (built from the data)
def widget(col):
    s = df[col]
    if col in categorical:
        opts = sorted(s.dropna().unique().tolist())
        return st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(s.mode()[0]), key=col)
    lo, hi, med = float(s.min()), float(s.max()), float(s.median())
    label = col.replace("_", " ").title()
    if pd.api.types.is_integer_dtype(s):
        return st.slider(label, int(lo), int(hi), int(med), key=col)
    step = 0.1 if (hi - lo) <= 100 else 1.0
    return st.slider(label, round(lo, 1), round(hi, 1), round(med, 1), step, key=col)


inp = {}
with st.sidebar:
    st.header("Student profile")
    placed = set()
    for group, cols in GROUPS.items():
        cols = [c for c in cols if c in FEATS]
        if cols:
            with st.expander(group, expanded=(group != "Background")):
                for c in cols:
                    inp[c] = widget(c); placed.add(c)
    rest = [c for c in FEATS if c not in placed]
    if rest:
        with st.expander("Other"):
            for c in rest:
                inp[c] = widget(c)

row = pd.DataFrame([inp])[FEATS]
click_cols = [c for c in numeric if c.startswith("clicks_") and c != "clicks_first_4_weeks"]
if click_cols and "clicks_first_4_weeks" in inp and inp["clicks_first_4_weeks"] > sum(inp[c] for c in click_cols):
    st.sidebar.warning("Clicks in first 4 weeks exceed total clicks – please adjust.")

tab_pred, tab_whatif, tab_perf, tab_data, tab_about = st.tabs(
    ["🔮 Predict", "🧪 What-if", "📊 Model performance", "🗂 Data explorer", "ℹ️ About"])

# ------------------------------------------------------------------ predict
with tab_pred:
    chosen = st.selectbox("Model", list(MODEL_NAMES), index=list(MODEL_NAMES).index("dnn"),
                          format_func=lambda k: MODEL_NAMES[k])
    pred = float(np.clip(models[chosen].predict(row)[0], 0, 100))
    rmse = metrics["results"][chosen]["RMSE"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted final score", f"{pred:.1f} / 100")
    c2.metric("Typical error (RMSE)", f"± {rmse:.1f}")
    c3.metric("Cohort average", f"{df[TARGET].mean():.1f}", f"{pred - df[TARGET].mean():+.1f}")
    c4.metric("Status", "⚠️ At risk" if pred < RISK else "✅ On track", f"threshold {RISK:.0f}", delta_color="off")
    st.progress(min(max(pred / 100, 0.0), 1.0))

    st.subheader("Biggest associated gains if engagement reached the cohort's 75th percentile")
    gains = suggest_gains(models["dnn"], row, df, numeric)
    if gains.empty:
        st.write("This profile is already at or above the 75th percentile on all actionable features.")
    else:
        st.dataframe(gains.style.format({"current": "{:.1f}", "cohort 75th pct": "{:.1f}", "predicted gain": "{:+.2f}"}),
                     hide_index=True)
    st.subheader("All models side by side")
    st.bar_chart(pd.DataFrame({MODEL_NAMES[k]: [float(np.clip(m.predict(row)[0], 0, 100))]
                               for k, m in models.items()}, index=["Predicted score"]).T)
    st.info("Predictions are statistical associations, not causal effects. They should support – never replace – "
            "educator judgement, and must not be used for grading or other high-stakes decisions.")

# ------------------------------------------------------------------ what-if
with tab_whatif:
    st.markdown("Change **one** input while all others stay as set in the sidebar.")
    left, right = st.columns(2)
    with left:
        feat = st.selectbox("Numeric feature to sweep", numeric,
                            index=numeric.index("active_weeks_pct") if "active_weeks_pct" in numeric else 0)
        vals = np.linspace(float(df[feat].quantile(0.01)), float(df[feat].quantile(0.99)), 40)
        curves = pd.DataFrame({"Deep NN": sweep_numeric(models["dnn"], row, feat, vals),
                               "Linear": sweep_numeric(models["linear"], row, feat, vals)}, index=np.round(vals, 2))
        st.line_chart(curves, x_label=feat, y_label="Predicted score")
    with right:
        cat = st.selectbox("Categorical feature to compare", categorical,
                           index=categorical.index("ai_use_style") if "ai_use_style" in categorical else 0)
        cats = sorted(df[cat].unique().tolist())
        st.bar_chart(pd.DataFrame({"Predicted score": compare_categories(models["dnn"], row, cat, cats)}, index=cats))

# ------------------------------------------------------------------ performance
with tab_perf:
    res = pd.DataFrame(metrics["results"]).T.set_index("name")[
        ["MAE", "RMSE", "R2", "CV_R2_mean", "CV_R2_std", "risk_precision", "risk_recall"]].astype(float)
    st.dataframe(res.style.format("{:.3f}"))
    st.caption(f"Train n = {metrics['n_train']}, test n = {metrics['n_test']} "
               f"({metrics.get('n_at_risk_test', '?')} below the risk threshold of {RISK:.0f}). "
               f"Deep network stopped after {metrics.get('dnn_epochs', '?')} epochs (early stopping).")
    st.subheader("Permutation importance (deep network)")
    st.bar_chart(pd.Series(metrics["importance"]).sort_values(ascending=False))
    for img in ["dnn_pred_vs_actual.png", "dnn_training_curve.png", "model_comparison.png"]:
        if (REPORT_DIR / img).exists():
            st.image(str(REPORT_DIR / img))

# ------------------------------------------------------------------ data
with tab_data:
    st.write(f"{len(df):,} rows × {df.shape[1]} columns · file: `{metrics['data_file']}`")
    st.dataframe(df.head(200))
    if categorical:
        g = st.selectbox("Mean final score by", categorical, key="grp")
        st.bar_chart(df.groupby(g)[TARGET].mean())
    st.download_button("Download dataset (CSV)", df.to_csv(index=False), "dataset.csv")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Topic:** *Predicting Student Academic Performance from Digital Learning Engagement in the Age of AI.*

**Task:** supervised regression – predict `final_score` (0–100) from LMS engagement, assessment behaviour,
background and (in the synthetic dataset) AI-use variables.

**Models:** Linear & Ridge regression, Random Forest, Gradient Boosting and a deep feed-forward neural network
(128→64→32, ReLU, Adam, L2, early stopping, log-scaled click counts).

**Data:** see `data/generate_dataset.py` (synthetic) and `prepare_oulad.py` (real OULAD).
""")
