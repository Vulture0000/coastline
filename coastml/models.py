"""Stage 6-7: ML modeling (Linear Regression, Random Forest, XGBoost) with
leave-one-storm-out validation, performance metrics, SHAP and feature importance.

Artifacts written to outputs/:
  model_metrics.csv / .json   - headline + timing metrics per model
  loso_per_storm.csv          - per-storm held-out metrics per model
  predictions_loso.csv        - out-of-fold predictions (all models)
  feature_importance.csv      - global importance (SHAP |mean| or fallback)
  shap_summary_<model>.png    - SHAP summary plots
  models/<name>.pkl           - refit models on full data (for the dashboard)
"""
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from . import config as C
from .dataset import load_master

MODELS = {
    "LinearRegression": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        random_state=C.RANDOM_SEED, n_jobs=-1),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, random_state=C.RANDOM_SEED, verbosity=0),
}


def _metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(df):
    X = df[C.FEATURES].to_numpy()
    y = df[C.TARGET].to_numpy()
    storms = df.storm.to_numpy()

    results, per_storm, oof = {}, [], {m: np.zeros(len(y)) for m in MODELS}
    shap_data = {}

    for name, factory in MODELS.items():
        # leave-one-storm-out cross-validation
        t0 = time.perf_counter()
        train_times, pred_times = [], []
        for s in sorted(set(storms)):
            tr, te = storms != s, storms == s
            m = factory()
            t1 = time.perf_counter()
            m.fit(X[tr], y[tr])
            train_times.append(time.perf_counter() - t1)
            t2 = time.perf_counter()
            p = m.predict(X[te])
            pred_times.append(time.perf_counter() - t2)
            oof[name][te] = p
            per_storm.append({"model": name, "held_out_storm": s,
                              **_metrics(y[te], p)})
        # refit on all data for deployment + explanation
        t3 = time.perf_counter()
        full = factory()
        full.fit(X, y)
        total_train = time.perf_counter() - t3
        t4 = time.perf_counter()
        full.predict(X[:1])
        single_pred_ms = (time.perf_counter() - t4) * 1000.0

        oof_m = _metrics(y, oof[name])
        results[name] = {
            **{f"loso_{k}": v for k, v in oof_m.items()},
            "train_time_s": float(np.sum(train_times) + total_train),
            "mean_fold_train_s": float(np.mean(train_times)),
            "mean_fold_predict_s": float(np.mean(pred_times)),
            "single_prediction_ms": float(single_pred_ms),
        }
        shap_data[name] = full

    oof_df = df.copy()
    for name in MODELS:
        oof_df[f"pred_{name}"] = np.round(oof[name], 2)

    best = min(results, key=lambda m: results[m]["loso_rmse"])
    for name in results:
        results[name]["is_best"] = name == best
    return results, pd.DataFrame(per_storm), oof_df, shap_data, best


def explain(shap_data, df, best):
    """Paper-style SHAP figures: beeswarm summary + dependence plots (top-3
    features, auto interaction colouring) + global importance (mean |SHAP|)."""
    imp_rows, paths = [], {}
    X = df[C.FEATURES].to_numpy()
    shap = None
    try:
        import shap
    except Exception as e:
        print(f"  (shap unavailable: {e})")
    for name, model in shap_data.items():
        sv = None
        if shap is not None:
            for factory in (lambda: shap.TreeExplainer(model),
                            lambda: shap.LinearExplainer(model, X)):
                try:
                    sv = factory().shap_values(X)
                    break
                except Exception:
                    sv = None
        if sv is not None:
            mean_abs = np.abs(sv).mean(axis=0)
            for f, v in zip(C.FEATURES, mean_abs):
                imp_rows.append({"model": name, "feature": f,
                                 "importance": float(v), "method": "shap"})
            entry = {}
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.figure(figsize=(9, 6))
                shap.summary_plot(sv, X, feature_names=C.FEATURES, show=False,
                                  plot_size=None)
                p = os.path.join(C.OUT_DIR, f"shap_summary_{name}.png")
                plt.tight_layout()
                plt.savefig(p, dpi=130, bbox_inches="tight")
                plt.close("all")
                entry["summary"] = os.path.basename(p)
                top3 = [C.FEATURES[i] for i in np.argsort(mean_abs)[::-1][:3]]
                entry["dependence"] = {}
                for f in top3:
                    plt.figure(figsize=(7, 5.5))
                    shap.dependence_plot(f, sv, X, feature_names=C.FEATURES,
                                         interaction_index="auto", show=False)
                    fp = os.path.join(C.OUT_DIR,
                                      f"shap_dependence_{name}_{f}.png")
                    plt.tight_layout()
                    plt.savefig(fp, dpi=130, bbox_inches="tight")
                    plt.close("all")
                    entry["dependence"][f] = os.path.basename(fp)
            except Exception as e:  # plots are optional
                print(f"  (shap plot skipped for {name}: {e})")
            paths[name] = entry
        else:
            imp = getattr(model, "feature_importances_", None)
            if imp is None:
                coef = np.abs(model.coef_)
                imp = coef / coef.sum()
            for f, v in zip(C.FEATURES, imp):
                imp_rows.append({"model": name, "feature": f,
                                 "importance": float(v), "method": "builtin"})
    with open(os.path.join(C.OUT_DIR, "shap_plots.json"), "w") as fh:
        json.dump(paths, fh, indent=2)
    return pd.DataFrame(imp_rows), paths


def paper_plots(metrics_df, per_storm, imp_df, best, n_pred):
    """Publication-style figures: mean |SHAP| bar chart + grouped MAE bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: mean |SHAP| per feature (best model)
    sub = imp_df[imp_df.model == best].sort_values("importance", ascending=False)
    if len(sub):
        vals = sub.importance.to_numpy()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.Blues(np.linspace(0.85, 0.35, len(vals)))
        y = np.arange(len(sub))[::-1]
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(sub.feature, fontsize=9)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.01, yi, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Mean |SHAP| (m of shoreline retreat)")
        ax.set_title(f"Global feature importance — {best}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(C.OUT_DIR, "mean_shap_bar.png"), dpi=200)
        plt.close(fig)

    # Fig 2: grouped MAE bars (fold mean ± SD vs pooled out-of-fold)
    models = metrics_df.model.tolist()
    fold_mean, fold_sd, pooled = [], [], []
    for m in models:
        f = per_storm[per_storm.model == m].mae
        fold_mean.append(f.mean())
        fold_sd.append(f.std(ddof=1))
        pooled.append(metrics_df.loc[metrics_df.model == m, "loso_mae"].iloc[0])
    x = np.arange(len(models))
    w = 0.35
    n_folds = per_storm.held_out_storm.nunique()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w / 2, fold_mean, w, yerr=fold_sd, capsize=5, color="#1f4e79",
           label=f"Leave-one-storm-out CV MAE (mean ± SD, {n_folds} folds)")
    ax.bar(x + w / 2, pooled, w, color="#c55a11",
           label=f"Pooled out-of-fold MAE ({n_pred} predictions)")
    for xi, v in zip(x - w / 2, fold_mean):
        ax.text(xi, 0.02, f"{v:.3f}", ha="center", va="bottom", color="white",
                fontsize=8)
    for xi, v in zip(x + w / 2, pooled):
        ax.text(xi, 0.02, f"{v:.3f}", ha="center", va="bottom", color="black",
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean absolute error (m of shoreline retreat)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=True,
              fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "model_mae_comparison.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(C.OUT_DIR, "models"), exist_ok=True)
    df = load_master()
    print(f"training on {len(df)} rows, {len(C.FEATURES)} features, "
          f"{df.storm.nunique()} storms (leave-one-storm-out)")
    results, per_storm, oof_df, shap_data, best = train_and_evaluate(df)
    imp, shap_paths = explain(shap_data, df, best)

    metrics_df = pd.DataFrame([{"model": k, **v} for k, v in results.items()])
    cols = ["model", "loso_rmse", "loso_mae", "loso_r2", "train_time_s",
            "mean_fold_train_s", "mean_fold_predict_s", "single_prediction_ms", "is_best"]
    metrics_df = metrics_df[cols].sort_values("loso_rmse")
    metrics_df.to_csv(os.path.join(C.OUT_DIR, "model_metrics.csv"), index=False)
    with open(os.path.join(C.OUT_DIR, "model_metrics.json"), "w") as fh:
        json.dump({"best_model": best, "shap_plots": shap_paths,
                   "metrics": results}, fh, indent=2)
    per_storm.to_csv(os.path.join(C.OUT_DIR, "loso_per_storm.csv"), index=False)
    oof_df.to_csv(os.path.join(C.OUT_DIR, "predictions_loso.csv"), index=False)
    imp.to_csv(os.path.join(C.OUT_DIR, "feature_importance.csv"), index=False)
    for name, model in shap_data.items():
        with open(os.path.join(C.OUT_DIR, "models", f"{name}.pkl"), "wb") as fh:
            pickle.dump(model, fh)
    paper_plots(metrics_df, per_storm, imp, best, len(df))

    print("\n=== leave-one-storm-out performance ===")
    print(metrics_df.to_string(index=False))
    print(f"\nbest model: {best}")
    print("artifacts ->", C.OUT_DIR)


if __name__ == "__main__":
    main()
