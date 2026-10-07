"""Shared code: column inference, preprocessing, model zoo, load-or-train, what-if helpers."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "student_engagement_ai.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET = "final_score"
SEED = 42
RISK_THRESHOLD = 55.0          # predicted/actual score below this = "at risk" (adjust to your institution)

MODEL_NAMES = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "rf": "Random Forest",
    "gbr": "Gradient Boosting",
    "dnn": "Deep Neural Network (MLP)",
}

# sidebar grouping in the app (unknown columns go to "Other")
GROUPS = {
    "Background": ["course_module", "highest_education", "age_band", "num_of_prev_attempts", "studied_credits"],
    "Digital engagement (LMS)": ["active_days", "active_weeks_pct", "clicks_content", "clicks_resource",
                                 "clicks_forum", "clicks_quiz", "clicks_homepage", "clicks_other",
                                 "clicks_first_4_weeks"],
    "Assessment behaviour": ["assessments_submitted_pct", "submission_timeliness_days"],
    "AI use": ["ai_use_style", "ai_assisted_study_hours_weekly", "ai_feedback_interactions_weekly"],
}
NON_ACTIONABLE = {"num_of_prev_attempts", "studied_credits", "clicks_first_4_weeks", "clicks_homepage", "clicks_other"}


def infer_columns(df, target=TARGET):
    feats = [c for c in df.columns if c != target]
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in feats if c not in cat]
    return num, cat


def make_preprocessor(numeric, categorical):
    # click counts are heavy-tailed -> log1p before scaling (helps the neural network a lot)
    log_cols = [c for c in numeric if c.startswith("clicks_")]
    plain = [c for c in numeric if c not in log_cols]
    log_pipe = Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())])
    return ColumnTransformer([
        ("num", StandardScaler(), plain),
        ("log", log_pipe, log_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def build_models(numeric, categorical):
    dnn = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers = deep feed-forward network
        activation="relu", solver="adam", alpha=1e-3, learning_rate_init=1e-3,
        batch_size=64, max_iter=400, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=SEED,
    )
    regs = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        "gbr": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=SEED),
        "dnn": TransformedTargetRegressor(regressor=dnn, transformer=StandardScaler()),
    }
    return {k: Pipeline([("prep", make_preprocessor(numeric, categorical)), ("model", v)])
            for k, v in regs.items()}


def load_data(path=None):
    return pd.read_csv(path or DATA_PATH, keep_default_na=False, na_values=[""])


def load_or_train():
    """Load saved models; if missing/incompatible (e.g. other scikit-learn version), retrain."""
    try:
        models = joblib.load(MODEL_DIR / "models.joblib")
        metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
        return models, metrics
    except Exception:
        import train
        return train.run(save_plots=False)


# ---------------------------------------------------------------- what-if helpers
def sweep_numeric(model, row, col, values):
    grid = pd.concat([row] * len(values), ignore_index=True)
    grid[col] = values
    return np.clip(model.predict(grid), 0, 100)


def compare_categories(model, row, col, categories):
    grid = pd.concat([row] * len(categories), ignore_index=True)
    grid[col] = categories
    return np.clip(model.predict(grid), 0, 100)


def suggest_gains(model, row, df, numeric, q=0.75, top=5):
    """For each actionable numeric feature currently below the cohort's q-quantile, predict the
    score if the student moved up to that quantile. Associative, NOT causal."""
    base = float(np.clip(model.predict(row)[0], 0, 100))
    out = []
    for c in numeric:
        if c in NON_ACTIONABLE:
            continue
        target = float(df[c].quantile(q))
        if float(row[c].iloc[0]) < target:
            r = row.copy(); r[c] = target
            out.append((c, float(row[c].iloc[0]), target, float(np.clip(model.predict(r)[0], 0, 100)) - base))
    out.sort(key=lambda t: -t[3])
    return pd.DataFrame(out[:top], columns=["feature", "current", "cohort 75th pct", "predicted gain"])
