"""Stage 7: composite explainability-analysis figure for the paper.

Writes outputs/explainability_analysis.png (2x3 panels, 300 dpi):
  (a) global mean |SHAP| importance - best model
  (b) SHAP beeswarm summary - best model
  (c-e) dependence plots for the top-3 drivers - best model
  (f) share of total mean |SHAP| per feature - all three models

Usage:  .venv/bin/python -m coastml.explain_figure
Reads data/master_dataset.csv + outputs/models/*.pkl (no retraining).
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

from . import config as C

FIG_NAME = "explainability_analysis.png"
MODEL_NAMES = ["LinearRegression", "RandomForest", "XGBoost"]


def _load():
    df = pd.read_csv(C.CSV_PATH)
    with open(os.path.join(C.OUT_DIR, "model_metrics.json")) as fh:
        best = json.load(fh)["best_model"]
    models = {}
    for name in MODEL_NAMES:
        p = os.path.join(C.OUT_DIR, "models", f"{name}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as fh:
                models[name] = pickle.load(fh)
    if not models:
        raise SystemExit("no pickled models in outputs/models - run coastml.models first")
    return df, models, best


def _shap(model, X):
    import shap
    for factory in (lambda: shap.TreeExplainer(model),
                    lambda: shap.LinearExplainer(model, X)):
        try:
            return factory().shap_values(X)
        except Exception:
            continue
    return None


def _panel_label(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=10.5,
                 fontweight="bold", pad=8)


def _figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    df, models, best = _load()
    X = df[C.FEATURES].to_numpy()
    sv = {name: _shap(m, X) for name, m in models.items()}
    sv = {k: v for k, v in sv.items() if v is not None}
    if best not in sv:
        best = next(iter(sv))
    sb = sv[best]

    mean_abs = np.abs(sb).mean(axis=0)
    order = np.argsort(mean_abs)          # ascending for barh
    top3 = [C.FEATURES[i] for i in np.argsort(mean_abs)[::-1][:3]]

    fig = plt.figure(figsize=(16.5, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.55,
                          left=0.115, right=0.98, top=0.90, bottom=0.10)

    # (a) global importance bar chart -----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    vals = mean_abs[order]
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(vals)))[::-1]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([C.FEATURES[i] for i in order], fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(v + vals.max() * 0.015, yi, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_xlabel("Mean |SHAP| (m of shoreline retreat)", fontsize=9)
    ax.set_xlim(0, vals.max() * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "(a)", f"Global importance — {best}")

    # (b) beeswarm summary ----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    exp = shap.Explanation(values=sb, data=X, feature_names=C.FEATURES)
    shap.plots.beeswarm(exp, ax=ax, show=False, plot_size=None,
                        max_display=len(C.FEATURES))
    ax.set_xlabel("SHAP value (m of retreat)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.title.set_fontsize(10.5)
    ax.set_title(f"(b)  SHAP summary — {best}", loc="left", fontweight="bold")
    ax.set_ylabel("")

    # (c-e) dependence plots for the top-3 drivers -----------------------
    letters = ["(c)", "(d)", "(e)"]
    for k, f in enumerate(top3):
        ax = fig.add_subplot(gs[1, k])
        shap.dependence_plot(f, sb, X, feature_names=C.FEATURES,
                             interaction_index="auto", ax=ax, show=False,
                             dot_size=18)
        ax.set_title(f"{letters[k]}  {f}", loc="left", fontsize=10.5,
                     fontweight="bold", pad=8)
        ax.set_xlabel(f, fontsize=9)
        ax.set_ylabel(f"SHAP value for {f}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    # (f) cross-model share of total |SHAP| ------------------------------
    ax = fig.add_subplot(gs[0, 2])
    shares, present = {}, []
    for name, s in sv.items():
        ma = np.abs(s).mean(axis=0)
        sh = ma / ma.sum() * 100.0
        shares[name] = {C.FEATURES[i]: sh[i] for i in range(len(C.FEATURES))}
        present.append(name)
    feat_order = sorted(C.FEATURES, key=lambda f: -max(shares[n][f] for n in present))[:6]
    w = 0.26
    xs = np.arange(len(feat_order))
    palette = {"LinearRegression": "#1f4e79", "RandomForest": "#c55a11",
               "XGBoost": "#548235"}
    for k, name in enumerate(present):
        h = [shares[name][f] for f in feat_order]
        ax.bar(xs + (k - (len(present) - 1) / 2) * w, h, w,
               color=palette.get(name, f"C{k}"), label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(feat_order, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("Share of total mean |SHAP| (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, ncol=1 if len(present) > 2 else 1)
    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "(f)", "Driver share across models")

    fig.suptitle("Explainability analysis — SHAP attributions of predicted "
                 "shoreline retreat (leave-one-storm-out best model)",
                 fontsize=13, fontweight="bold", y=0.965)
    return fig


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = _figure()
    out = os.path.join(C.OUT_DIR, FIG_NAME)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figure ->", out)
    return out


def main():
    build()


if __name__ == "__main__":
    main()
