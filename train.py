"""Train + evaluate regression and deep-learning models.

Run:  python train.py                                  # synthetic dataset
      python train.py --data data/oulad_features.csv   # real OULAD table from prepare_oulad.py
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from edu_model import (DATA_PATH, MODEL_DIR, MODEL_NAMES, REPORT_DIR, RISK_THRESHOLD, ROOT, SEED,
                       TARGET, build_models, infer_columns, load_data)


def run(data_path=None, save_plots=True):
    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    data_path = Path(data_path or DATA_PATH).resolve()
    df = load_data(data_path).dropna(subset=[TARGET])
    numeric, categorical = infer_columns(df)
    feats = numeric + categorical
    X, y = df[feats], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = build_models(numeric, categorical)
    try:
        data_file = str(data_path.relative_to(ROOT))
    except ValueError:
        data_file = str(data_path)
    metrics = {"n_train": len(X_tr), "n_test": len(X_te), "data_file": data_file,
               "numeric": numeric, "categorical": categorical, "risk_threshold": RISK_THRESHOLD,
               "results": {}}
    preds = {}
    kf = KFold(5, shuffle=True, random_state=SEED)

    for key, pipe in models.items():
        cv = cross_val_score(pipe, X_tr, y_tr, cv=kf, scoring="r2")
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te); preds[key] = p
        act_risk, pred_risk = (y_te < RISK_THRESHOLD).to_numpy(), p < RISK_THRESHOLD
        tp = int((act_risk & pred_risk).sum())
        metrics["results"][key] = {
            "name": MODEL_NAMES[key],
            "MAE": float(mean_absolute_error(y_te, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, p))),
            "R2": float(r2_score(y_te, p)),
            "CV_R2_mean": float(cv.mean()), "CV_R2_std": float(cv.std()),
            "risk_recall": tp / max(int(act_risk.sum()), 1),
            "risk_precision": tp / max(int(pred_risk.sum()), 1),
        }
        r = metrics["results"][key]
        print(f"{r['name']:28s} MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} R2={r['R2']:.4f} "
              f"CV-R2={r['CV_R2_mean']:.4f}±{r['CV_R2_std']:.4f} "
              f"risk P/R={r['risk_precision']:.2f}/{r['risk_recall']:.2f}")

    imp = permutation_importance(models["dnn"], X_te, y_te, n_repeats=8, random_state=SEED, scoring="r2")
    importance = pd.DataFrame({"feature": feats, "importance": imp.importances_mean}).sort_values(
        "importance", ascending=False)
    metrics["importance"] = importance.set_index("feature")["importance"].to_dict()
    mlp = models["dnn"].named_steps["model"].regressor_    # fitted MLP inside TransformedTargetRegressor
    metrics["dnn_loss_curve"] = [float(v) for v in mlp.loss_curve_]
    metrics["dnn_val_curve"] = [float(v) for v in mlp.validation_scores_]
    metrics["dnn_epochs"] = int(mlp.n_iter_)
    metrics["n_at_risk_test"] = int((y_te < RISK_THRESHOLD).sum())

    joblib.dump(models, MODEL_DIR / "models.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if save_plots:
        make_plots(y_te, preds, importance, metrics)
    print("Saved models/ and reports/")
    return models, metrics


def make_plots(y_te, preds, importance, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    p = preds["dnn"]
    ax[0].scatter(y_te, p, s=8, alpha=0.35)
    lim = [min(y_te.min(), p.min()), max(y_te.max(), p.max())]
    ax[0].plot(lim, lim, "r--")
    ax[0].set(xlabel="Actual final score", ylabel="Predicted", title="Deep NN: predicted vs actual")
    ax[1].hist(y_te - p, bins=40, color="tab:green", alpha=0.8)
    ax[1].set(xlabel="Residual", ylabel="Count", title="Deep NN: residuals")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "dnn_pred_vs_actual.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    top = importance.head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="tab:purple")
    ax.set(xlabel="Drop in R² when shuffled", title="Permutation importance (Deep NN)")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics["dnn_loss_curve"], label="Training loss"); ax.set(xlabel="Epoch", ylabel="Loss")
    ax2 = ax.twinx(); ax2.plot(metrics["dnn_val_curve"], color="tab:orange"); ax2.set_ylabel("Validation R²")
    ax.set_title("Deep NN training curve")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "dnn_training_curve.png", dpi=150); plt.close(fig)

    names = [v["name"] for v in metrics["results"].values()]
    r2 = [v["R2"] for v in metrics["results"].values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, r2, color="tab:blue"); ax.set(ylabel="Test R²", title="Model comparison", ylim=(min(r2) - 0.05, 1))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); fig.savefig(REPORT_DIR / "model_comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV with a final_score column")
    run(ap.parse_args().data)
