# app.py
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Project Identity ----------
PROJECT_NAME = "RiverGuard Logistic (RGL): Water Pollution Predictor"
FEATURES = ["DOmgl", "BODmgl", "CODmgl", "SSmgl", "pH", "NH3Nmgl"]
TARGET = "RIVERSTATUS"  # 0 = Clean/Safe, 1 = Polluted

st.set_page_config(page_title=PROJECT_NAME, layout="wide")
st.title(PROJECT_NAME)

# ---------- Quick Start ----------
with st.expander("📘 Quick Start — How to use this app", expanded=True):
    st.markdown(
        """
        **Inputs:** `DOmgl`, `BODmgl`, `CODmgl`, `SSmgl`, `pH`, `NH3Nmgl` (and `RIVERSTATUS` if training/evaluating).  
        **Outputs:**  
        - **𝑧 = ln(p/(1−p))** (logit)  
        - **e^𝑧**  
        - **k = e^𝑧/(1+e^𝑧)** (predicted probability; **k** used to avoid confusion with the \(p\) in the logit)  
        
        **Two modes:**
        1. **Manual coefficients** — Enter β₀ and β’s, then score a single row or an uploaded file.
        2. **Train from data** — Upload CSV/Excel with all columns (including binary `RIVERSTATUS` 0/1). 
           The app trains a logistic model, shows metrics, and exports raw-space coefficients.

        **Graphs:**
        - Single row: probability meter and predicted status  
        - Batch: bar chart of predicted water status; if `RIVERSTATUS` exists, Persistency (%) and confusion table

        **Tip:** Use the **threshold** in the sidebar to decide when **k** becomes "Polluted" (default 0.50).
        """
    )

# ---------- About ----------
with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
        **RiverGuard Logistic (RGL)** is a decision-support tool that uses **logistic regression** 
        to estimate the probability that a river sample is **polluted (1)** vs **clean/safe (0)** 
        given water-quality indicators.

        The model computes a **logit** score `z = β₀ + Σ βᵢ xᵢ`, then converts it to a probability **k** via the logistic function.  
        You can supply your own coefficients (e.g., from studies) or train from a labelled dataset.

        **Note:** Logistic regression models linear relationships in the log-odds. Validate results with held-out data and domain knowledge, 
        and be mindful of data quality and class labelling standards (e.g., WQI).
        """
    )

# ---------- Variable Glossary ----------
st.markdown("### Variable glossary (scientific names)")
gloss = pd.DataFrame({
    "Column": ["DOmgl", "BODmgl", "CODmgl", "SSmgl", "pH", "NH3Nmgl", "RIVERSTATUS"],
    "Scientific name": [
        "Dissolved Oxygen (mg/L)",
        "Biochemical Oxygen Demand (mg/L)",
        "Chemical Oxygen Demand (mg/L)",
        "Total Suspended Solids (mg/L)",
        "pH",
        "Ammoniacal Nitrogen, NH\u2083–N (mg/L)",
        "River class/status label (0=Clean/Safe, 1=Polluted) — commonly derived from WQI class"
    ]
})
st.dataframe(gloss, use_container_width=True, hide_index=True)

st.info(
    "**Comparing with WQI** — If your dataset’s `RIVERSTATUS` was assigned from Malaysian WQI classes, "
    "the app compares the logistic prediction vs. WQI-derived labels and reports **Persistency (%)** and a confusion table. "
    "If you want the app to compute WQI itself from raw readings, provide the DOE WQI formula/weights and thresholds you use."
)

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Mode & Threshold")
    mode = st.radio("Choose how to compute:", ["Manual coefficients", "Train from data"], index=0)
    thresh = st.slider("Classification threshold (k ≥ threshold = Polluted)", 0.05, 0.95, 0.50, 0.01)
    st.caption("Columns required: " + ", ".join(FEATURES + [TARGET]))

# ---------- Helpers ----------
def ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    return df[cols]

def compute_outputs(df: pd.DataFrame, intercept: float, coefs: dict) -> pd.DataFrame:
    """
    Produce z, e^z, k and predicted class/label using the global threshold 'thresh'.
    """
    X = df[[c for c in FEATURES if c in df.columns]].astype(float)
    beta = np.array([coefs.get(col, 0.0) for col in X.columns], dtype=float)
    z = intercept + X.values @ beta
    ez = np.exp(z)
    k = ez / (1.0 + ez)
    out = df.copy()
    out["z"] = z
    out["e^z"] = ez
    out["k"] = k
    out["PredictedStatus"] = (out["k"] >= thresh).astype(int)
    out["PredictedLabel"] = np.where(out["PredictedStatus"] == 1, "Polluted", "Clean")
    return out

def format_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pretty labels for UI (downloads keep simple names).
    """
    return df.rename(columns={
        "z": "𝑧 = ln(p/(1−p))",
        "e^z": "e^𝑧",
        "k": "k = e^𝑧/(1+e^𝑧)"
    })

def download_df(df: pd.DataFrame, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download {label} (CSV)",
        data=csv,
        file_name=f"{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

def status_bar_chart(df_with_preds: pd.DataFrame, title: str = "Predicted Water Status"):
    counts = df_with_preds["PredictedLabel"].value_counts().reindex(["Clean", "Polluted"]).fillna(0).astype(int)
    st.markdown(f"##### {title}")
    st.bar_chart(counts)

# ---------- UI ----------
if mode == "Manual coefficients":
    st.subheader("Manual Coefficients")
    with st.expander("Set coefficients (β) and intercept (β₀)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            # Defaults inserted per your formula image
            beta0 = st.number_input("Intercept β₀", value=-11.326, format="%.6f")
            b_DO  = st.number_input("β_DOmgl", value=3.415, format="%.6f")
        with c2:
            b_BOD = st.number_input("β_BODmgl", value=-1.781, format="%.6f")
            b_COD = st.number_input("β_CODmgl", value=-0.271, format="%.6f")
        with c3:
            b_SS  = st.number_input("β_SSmgl", value=-0.035, format="%.6f")
            b_pH  = st.number_input("β_pH", value=0.000, format="%.6f")  # pH not used in your equation; set 0 by default
            b_NH3 = st.number_input("β_NH3Nmgl", value=5.853, format="%.6f")

        coefs = {"DOmgl": b_DO, "BODmgl": b_BOD, "CODmgl": b_COD, "SSmgl": b_SS, "pH": b_pH, "NH3Nmgl": b_NH3}

    st.markdown("#### Input options")
    tab1, tab2 = st.tabs(["Single row", "Batch upload"])

    # --- Single row ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            DO = st.number_input("DOmgl", value=5.0)
            BOD = st.number_input("BODmgl", value=2.0)
        with c2:
            COD = st.number_input("CODmgl", value=20.0)
            SS  = st.number_input("SSmgl", value=25.0)
        with c3:
            pH  = st.number_input("pH", value=7.0)
            NH3 = st.number_input("NH3Nmgl", value=0.1)

        input_df = pd.DataFrame([{"DOmgl": DO, "BODmgl": BOD, "CODmgl": COD, "SSmgl": SS, "pH": pH, "NH3Nmgl": NH3}])
        result = compute_outputs(input_df, beta0, coefs)

        # Display k + status
        prob = float(result.loc[0, "k"])
        status = "Polluted" if prob >= thresh else "Clean"
        colm1, colm2 = st.columns(2)
        with colm1:
            st.metric("Predicted probability (k)", f"{prob:.3f}")
            st.progress(min(max(prob, 0.0), 1.0))
        with colm2:
            st.metric("Predicted status", status)

        st.markdown("##### Output table")
        st.dataframe(format_display(result), use_container_width=True)
        download_df(result, "manual_single_prediction")

    # --- Batch upload ---
    with tab2:
        st.write("Upload a CSV or Excel with columns: " + ", ".join(FEATURES + [TARGET]))
        file = st.file_uploader("Upload file", type=["csv", "xlsx"])
        if file:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                try:
                    df = pd.read_excel(file, engine="openpyxl")
                except Exception as _e_xl:
                    st.error("Failed to read Excel file. Ensure it's a valid .xlsx (openpyxl engine).")
                    st.exception(_e_xl)
                    df = pd.DataFrame()

            if not df.empty:
                df = ensure_dataframe(df)
                st.dataframe(df.head(), use_container_width=True)
                result = compute_outputs(df, beta0, coefs)
                st.markdown("##### Output table")
                st.dataframe(format_display(result), use_container_width=True, height=420)
                status_bar_chart(result, "Predicted Water Status (Batch)")

                # If true labels present, show Persistency and confusion
                if TARGET in result.columns and TARGET in df.columns:
                    try:
                        cm = pd.crosstab(df[TARGET].astype(int), result["PredictedStatus"].astype(int),
                                         rownames=["True"], colnames=["Pred"], dropna=False)
                        acc = (df[TARGET].astype(int).values == result["PredictedStatus"].astype(int).values).mean() * 100.0
                        st.metric("Persistency (%)", f"{acc:.1f}%")
                        st.markdown("##### Confusion (WQI label vs Predicted)")
                        st.dataframe(cm)
                    except Exception as _e:
                        st.warning("Could not compute Persistency/Confusion. Check `RIVERSTATUS` values are 0/1.")
                        st.exception(_e)

                download_df(result, "manual_batch_prediction")

else:
    # ---------------------- Train from Data ----------------------
    st.subheader("Train from Data")
    st.markdown("- Upload a **CSV/Excel** containing the columns: " + ", ".join(FEATURES + [TARGET]) + " (target 0/1).")
    file = st.file_uploader("Upload training dataset", type=["csv", "xlsx"])
    if file:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except Exception as e_x:
                st.error("Failed to read Excel file. Please ensure it's a valid .xlsx file.")
                st.exception(e_x)
                df = pd.DataFrame()

        if not df.empty:
            df = ensure_dataframe(df)
            missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.success(f"Loaded {len(df)} rows.")
                st.dataframe(df.head(), use_container_width=True)

                # Train/validation split
                test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
                X = df[FEATURES].astype(float)
                y = df[TARGET].astype(int)

                # Optional scaling
                scale = st.checkbox("Standardize features (recommended)", value=True)
                if scale:
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                model = LogisticRegression(solver="liblinear", max_iter=1000)
                model.fit(X_train, y_train)

                # Metrics
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= thresh).astype(int)
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("##### Confusion Matrix (using chosen threshold)")
                    st.write(pd.DataFrame(confusion_matrix(y_test, y_pred),
                                          index=["True 0","True 1"], columns=["Pred 0","Pred 1"]))
                    st.markdown("##### Classification Report")
                    st.text(classification_report(y_test, y_pred, digits=3))
                with colB:
                    auc = roc_auc_score(y_test, y_prob)
                    st.metric("ROC AUC", f"{auc:.3f}")
                    fpr, tpr, thr = roc_curve(y_test, y_prob)
                    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr})
                    st.line_chart(roc_df.set_index("FPR")["TPR"])

                # Coefficients (convert to raw-space if scaled)
                coef = model.coef_.ravel().copy()
                intercept = float(model.intercept_[0])
                if scale:
                    mu = scaler.mean_
                    sigma = scaler.scale_
                    raw_betas = coef / sigma
                    raw_intercept = intercept - np.sum(coef * mu / sigma)
                    intercept, coef = raw_intercept, raw_betas

                coef_df = pd.DataFrame({"Feature": FEATURES, "Beta": coef})
                st.markdown("#### Model Coefficients (for raw, unscaled inputs)")
                st.dataframe(coef_df, use_container_width=True)
                st.write(f"**Intercept β₀**: {intercept:.6f}")

                # Inference
                st.markdown("---")
                st.markdown("### Predict / Score Data")
                tab1, tab2 = st.tabs(["Single row", "Batch upload"])

                def score_df(df_in: pd.DataFrame):
                    coef_map = dict(zip(FEATURES, coef))
                    return compute_outputs(df_in, intercept, coef_map)

                # Single row scoring
                with tab1:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        DO = st.number_input("DOmgl", value=5.0)
                        BOD = st.number_input("BODmgl", value=2.0)
                    with c2:
                        COD = st.number_input("CODmgl", value=20.0)
                        SS  = st.number_input("SSmgl", value=25.0)
                    with c3:
                        pH  = st.number_input("pH", value=7.0)
                        NH3 = st.number_input("NH3Nmgl", value=0.1)

                    inp = pd.DataFrame([{
                        "DOmgl": DO, "BODmgl": BOD, "CODmgl": COD,
                        "SSmgl": SS, "pH": pH, "NH3Nmgl": NH3
                    }])
                    res = score_df(inp)

                    prob = float(res.loc[0, "k"])
                    status = "Polluted" if prob >= thresh else "Clean"
                    colm1, colm2 = st.columns(2)
                    with colm1:
                        st.metric("Predicted probability (k)", f"{prob:.3f}")
                        st.progress(min(max(prob, 0.0), 1.0))
                    with colm2:
                        st.metric("Predicted status", status)

                    st.dataframe(format_display(res), use_container_width=True)
                    download_df(res, "trained_model_single_prediction")

                # Batch scoring
                with tab2:
                    st.write("Upload a CSV/Excel to score (must include columns: " + ", ".join(FEATURES) + ")")
                    f2 = st.file_uploader("Upload scoring file", type=["csv","xlsx"], key="scoreupload")
                    if f2:
                        # Read file safely
                        if f2.name.lower().endswith(".csv"):
                            df2 = pd.read_csv(f2)
                        else:
                            try:
                                df2 = pd.read_excel(f2, engine="openpyxl")
                            except Exception as e_x:
                                st.error("Failed to read Excel file. Please ensure it's a valid .xlsx file.")
                                st.exception(e_x)
                                df2 = pd.DataFrame()

                        if not df2.empty:
                            # Score only feature columns
                            df2 = df2[[c for c in FEATURES if c in df2.columns]].astype(float)
                            res2 = score_df(df2)

                            st.dataframe(format_display(res2), use_container_width=True, height=420)
                            status_bar_chart(res2, "Predicted Water Status (Batch)")

                            # Persistency vs RIVERSTATUS if present in original training df
                            if TARGET in df.columns and TARGET in df2.columns:
                                try:
                                    true_vals = df2[TARGET].astype(int).values
                                    pred_vals = res2["PredictedStatus"].astype(int).values
                                    persist = (true_vals == pred_vals).mean() * 100.0
                                    st.metric("Persistency (%)", f"{persist:.1f}%")
                                except Exception:
                                    pass

                            download_df(res2, "trained_model_batch_prediction")

                # Export params
                st.markdown("---")
                st.markdown("### Export Parameters")
                params = {
                    "intercept": float(intercept),
                    "betas": dict(zip(FEATURES, [float(b) for b in coef])),
                    "features_order": FEATURES,
                    "target": TARGET,
                    "threshold": float(thresh),
                }
                st.download_button(
                    "⬇️ Download model parameters (JSON)",
                    data=json.dumps(params, indent=2).encode("utf-8"),
                    file_name="logistic_parameters.json",
                    mime="application/json",
                    use_container_width=True
                )

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression (scikit-learn) • Columns: " + ", ".join(FEATURES + [TARGET]))
