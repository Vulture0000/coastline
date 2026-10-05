"""Stage 8: Streamlit dashboard - outputs, evaluation and decision support.

Run:  streamlit run coastml/app.py
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from . import config as C
    from .dataset import derive_features, load_master
except ImportError:
    from coastml import config as C
    from coastml.dataset import derive_features, load_master

st.set_page_config(page_title="Coastal Erosion ML Pipeline", page_icon="🌊", layout="wide")


@st.cache_data
def load_all():
    df = load_master()
    metrics = pd.read_csv(os.path.join(C.OUT_DIR, "model_metrics.csv"))
    per_storm = pd.read_csv(os.path.join(C.OUT_DIR, "loso_per_storm.csv"))
    importance = pd.read_csv(os.path.join(C.OUT_DIR, "feature_importance.csv"))
    preds = pd.read_csv(os.path.join(C.OUT_DIR, "predictions_loso.csv"))
    with open(os.path.join(C.OUT_DIR, "model_metrics.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(C.OUT_DIR, "dataset_qc.json")) as fh:
        qc = json.load(fh)
    shap_plots = {}
    sp_path = os.path.join(C.OUT_DIR, "shap_plots.json")
    if os.path.exists(sp_path):
        with open(sp_path) as fh:
            shap_plots = json.load(fh)
    models = {}
    for name in metrics.model:
        with open(os.path.join(C.OUT_DIR, "models", f"{name}.pkl"), "rb") as fh:
            models[name] = pickle.load(fh)
    return df, metrics, per_storm, importance, preds, meta, qc, models, shap_plots


df, metrics, per_storm, importance, preds, meta, qc, models, shap_plots = load_all()
best = meta["best_model"]

st.title("🌊 Coastal Erosion ML Pipeline — Coromandel Coast")
st.caption("Sentinel-2 style shoreline retreat modelling: 8-stage pipeline "
           "(acquisition → preprocessing → shoreline extraction → feature engineering → "
           "master dataset → ML modeling → evaluation → outputs). "
           "Validation: leave-one-storm-out cross-validation.")

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox(
        "Model", list(metrics.model),
        index=list(metrics.model).index(best))
    storms = sorted(df.storm.unique())
    storm = st.selectbox("Storm event", storms, index=storms.index("Gaja") if "Gaja" in storms else 0)
    st.divider()
    mrow = metrics[metrics.model == model_name].iloc[0]
    st.metric("LOSO R²", f"{mrow.loso_r2:.3f}")
    st.metric("LOSO RMSE", f"{mrow.loso_rmse:.2f} m")
    imp_sel = (importance[importance.model == model_name]
               .sort_values("importance", ascending=False).reset_index(drop=True))
    st.caption("**Key drivers of this model:**")
    for _, r in imp_sel.head(5).iterrows():
        st.caption(f"- {r['feature']} — {r['importance']:.3f}")
    st.caption(f"Best model by LOSO RMSE: **{best}**")

# model card: reacts instantly to sidebar changes on every tab
card = st.columns(4)
card[0].metric("Selected model", model_name,
               delta="BEST by LOSO RMSE" if model_name == best else None)
card[1].metric("LOSO R² / RMSE (m)", f"{mrow.loso_r2:.3f} / {mrow.loso_rmse:.2f}")
card[2].metric("LOSO MAE (m)", f"{mrow.loso_mae:.2f}")
card[3].metric("Top driver", imp_sel.iloc[0]["feature"])
st.caption(f"Key drivers for **{model_name}**: "
           + ", ".join(imp_sel["feature"].head(3))
           + f" · storm context: **{storm}**")
st.divider()

tab_data, tab_models, tab_explain, tab_predict, tab_maps = st.tabs(
    ["📊 Dataset", "🏆 Model performance", " Explainability", "🎯 Prediction & decision support", "🗺️ Erosion maps"])

# ---------------------------------------------------------------- dataset
with tab_data:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", qc["rows"])
    c2.metric("Sections", qc["sections"])
    c3.metric("Storms", len(qc["storms"]))
    c4.metric("Features", len(C.FEATURES))
    c5.metric("QC", "✅ PASS" if qc["checks_passed"] else "❌ FAIL")
    st.caption("Master dataset (section × storm) stored in SQLite "
               f"(`data/master.db`) and CSV. Nulls: {qc['nulls']}, "
               f"duplicate keys: {qc['duplicate_keys']}.")

    cA, cB = st.columns([1, 1])
    with cA:
        st.subheader("Observed retreat by storm")
        fig = px.box(df, x="storm", y=C.TARGET, color="storm",
                     labels={C.TARGET: "shoreline retreat (m)"})
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        st.subheader("Mean retreat per section")
        sec = df.groupby(["section_id", "lat", "lon"], as_index=False)[C.TARGET].mean()
        sec["marker"] = sec[C.TARGET].clip(lower=0.0) + 2.0
        fig = px.scatter_map(
            sec, lon="lon", lat="lat", color=C.TARGET, size="marker",
            color_continuous_scale="YlOrRd",
            labels={C.TARGET: "mean retreat (m)"}, zoom=6,
            map_style="open-street-map", title="Coastal sections (Chennai → Point Calimere)")
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Master dataset preview"):
        st.dataframe(df.sort_values(["storm", "section_id"]), use_container_width=True)

# ---------------------------------------------------------------- models
with tab_models:
    st.subheader("Leave-one-storm-out cross-validation performance")
    show = metrics.rename(columns={
        "loso_rmse": "RMSE (m)", "loso_mae": "MAE (m)", "loso_r2": "R²",
        "train_time_s": "Total train (s)", "mean_fold_train_s": "Fold train (s)",
        "mean_fold_predict_s": "Fold predict (s)", "single_prediction_ms": "1 pred (ms)",
        "is_best": "Best"})
    show = show.style.apply(
        lambda r: ["background-color: rgba(255, 214, 102, 0.45)"] * len(r)
        if r["model"] == model_name else [""] * len(r), axis=1)
    st.dataframe(show, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    for col, metric_col, title in (
            (c1, "loso_r2", "R² (higher is better)"),
            (c2, "loso_rmse", "RMSE in m (lower is better)"),
            (c3, "loso_mae", "MAE in m (lower is better)")):
        fig = px.bar(metrics, x="model", y=metric_col, color="model", title=title)
        col.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Speed comparison")
        fig = px.bar(metrics, x="model", y="single_prediction_ms", color="model",
                     title="Single prediction latency (ms, log scale)", log_y=True)
        c1.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Per-storm held-out R²")
        piv = per_storm.pivot(index="held_out_storm", columns="model", values="r2")
        fig = px.imshow(piv, text_auto=".2f", color_continuous_scale="RdYlGn",
                        aspect="auto", title="R² per held-out storm")
        c2.plotly_chart(fig, use_container_width=True)
    with st.expander("Per-storm metrics table"):
        st.dataframe(per_storm, use_container_width=True, hide_index=True)

    mae_png = os.path.join(C.OUT_DIR, "model_mae_comparison.png")
    if os.path.exists(mae_png):
        st.subheader("Which model works better? — MAE comparison")
        st.image(mae_png, caption="Grouped leave-one-storm-out CV MAE (mean ± SD over "
                                  "storm folds) vs pooled out-of-fold MAE per model")

    st.subheader("Out-of-fold predictions vs observed")
    c1, c2, c3 = st.columns(3)
    for col, m in zip((c1, c2, c3), metrics.model):
        fig = px.scatter(preds, x=C.TARGET, y=f"pred_{m}", color="storm",
                         title=f"{m} (LOSO)")
        lim = [0, max(preds[C.TARGET].max(), preds[f"pred_{m}"].max()) * 1.05]
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                      line=dict(dash="dash", color="gray"))
        fig.update_layout(xaxis_title="observed retreat (m)",
                          yaxis_title="predicted retreat (m)", showlegend=False)
        col.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- explain
with tab_explain:
    st.subheader(f"Global feature importance — {model_name}")
    imp = importance[importance.model == model_name].sort_values("importance", ascending=True)
    fig = px.bar(imp, x="importance", y="feature", orientation="h", color="feature",
                 title=f"{imp.method.iloc[0].upper()} importance")
    st.plotly_chart(fig, use_container_width=True)
    if model_name == best:
        shap_bar = os.path.join(C.OUT_DIR, "mean_shap_bar.png")
        if os.path.exists(shap_bar):
            st.image(shap_bar, caption=f"Mean |SHAP| per feature (m of shoreline "
                                       f"retreat) — {best}")
    png_meta = shap_plots.get(model_name, {})
    summary = png_meta.get("summary")
    if summary and os.path.exists(os.path.join(C.OUT_DIR, summary)):
        st.subheader(f"SHAP summary (beeswarm) — {model_name}")
        st.image(os.path.join(C.OUT_DIR, summary),
                 caption=f"Fig: global feature importance & impact on {model_name} output")
    deps = png_meta.get("dependence", {})
    if deps:
        st.subheader(f"SHAP dependence plots — {model_name}")
        cols = st.columns(min(len(deps), 3))
        for col, (feat, fname) in zip(cols, deps.items()):
            fp = os.path.join(C.OUT_DIR, fname)
            if os.path.exists(fp):
                col.image(fp, caption=f"contribution of {feat} (colour = interaction feature)")
    elif not summary:
        st.info(f"No SHAP figures for {model_name}; see coefficient importance above.")
    with st.expander("All-model importance table"):
        st.dataframe(importance, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- predict
with tab_predict:
    st.subheader("Scenario prediction & decision support")
    c1, c2 = st.columns([1, 2])
    with c1:
        sections = sorted(df.section_id.unique())
        section_id = st.selectbox("Coastal section", sections)
        base = df[(df.section_id == section_id) & (df.storm == storm)].iloc[0]
        st.caption(f"Section {section_id} @ ({base.lat:.4f}, {base.lon:.4f})")
        wind = st.slider("Storm wind (m/s)", 5.0, 50.0, float(base.storm_wind_ms), 0.5,
                         key=f"wind_{storm}_{section_id}")
        wave_h = st.slider("Wave height (m)", 0.3, 7.0, float(base.wave_height_m), 0.1,
                           key=f"wave_{storm}_{section_id}")
        mw = st.slider("Mangrove width (m)", 0.0, 3000.0, float(base.mangrove_width_m), 50.0,
                       key=f"mw_{storm}_{section_id}")
        st.toggle("Use stored storm values", value=False,
                  key=f"use_stored_{storm}_{section_id}",
                  help="When on, ignores sliders and predicts the recorded event.")
    row = base.to_dict()
    if st.session_state.get(f"use_stored_{storm}_{section_id}"):
        wind, wave_h, mw = row["storm_wind_ms"], row["wave_height_m"], row["mangrove_width_m"]
    row.update(storm_wind_ms=wind, wave_height_m=wave_h, mangrove_width_m=mw)
    row.update(derive_features(row))
    X = np.array([[row[f] for f in C.FEATURES]])
    pred = float(models[model_name].predict(X)[0])
    obs = float(base[C.TARGET])

    def risk(v):
        return ("Low" if v < 5 else "Moderate" if v < 15 else
                "High" if v < 30 else "Very high")

    with c2:
        cA, cB, cC = st.columns(3)
        cA.metric("Predicted retreat", f"{pred:.1f} m",
                  delta=f"{pred - obs:+.1f} m vs observed")
        cB.metric("Risk class", risk(pred))
        cC.metric("Recommended buffer", f"{pred * 1.5 + 10:.0f} m",
                  help="1.5x predicted retreat + 10 m safety setback")
        pressure = min(max(pred, 0.0) / 40.0, 1.0)
        st.progress(pressure,
                    text=f"Erosion pressure {pressure:.0%} of 40 m scale")
        st.caption("Decision support: sections in High / Very high class priority for "
                   "mangrove restoration, dune strengthening and setback enforcement.")

    st.subheader(f"Top-10 highest-risk sections — {storm} ({model_name})")
    sub = df[df.storm == storm].copy()
    Xs = sub[C.FEATURES].to_numpy()
    sub["predicted"] = models[model_name].predict(Xs)
    sub["risk"] = sub.predicted.apply(risk)
    sub["buffer_m"] = (sub.predicted * 1.5 + 10).round(0)
    st.dataframe(
        sub.sort_values("predicted", ascending=False).head(10)[
            ["section_id", "lat", "lon", "predicted", C.TARGET, "risk", "buffer_m"]
        ].rename(columns={C.TARGET: "observed"}),
        use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- maps
with tab_maps:
    st.subheader(f"Erosion & buffer map — {storm}")
    sub = df[df.storm == storm].copy()
    sub["predicted"] = models[model_name].predict(sub[C.FEATURES].to_numpy())
    sub["buffer_m"] = (sub.predicted * 1.5 + 10).clip(lower=2.0)
    fig = px.scatter_map(
        sub, lon="lon", lat="lat", color="predicted", size="buffer_m",
        color_continuous_scale="YlOrRd", zoom=6, map_style="open-street-map",
        labels={"predicted": "predicted retreat (m)", "buffer_m": "buffer (m)"},
        hover_data=["section_id", C.TARGET, "buffer_m"])
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=sub.section_id, y=sub.predicted, name="predicted",
                               marker_color="crimson"))
        fig.add_scatter(x=sub.section_id, y=sub[C.TARGET], mode="lines",
                        name="observed")
        fig.update_layout(title=f"{storm}: predicted vs observed by section",
                          xaxis_title="section", yaxis_title="retreat (m)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        mean_storm = df.groupby("storm")[C.TARGET].mean().reset_index()
        fig = px.bar(mean_storm, x="storm", y=C.TARGET, color="storm",
                     title="Mean retreat per storm event",
                     labels={C.TARGET: "mean retreat (m)"})
        st.plotly_chart(fig, use_container_width=True)
