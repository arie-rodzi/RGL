
# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# ---------- Project Identity ----------
PROJECT_NAME = "RiverGuard Logistic (RGL): Water Pollution Predictor"

st.set_page_config(page_title=PROJECT_NAME, layout="wide")

st.title(PROJECT_NAME)

with st.expander("📘 Quick Start — How to use this app", expanded=True):
    st.markdown(
        """
        **Inputs:** `DOmgl`, `BODmgl`, `CODmgl`, `SSmgl`, `pH`, `NH3Nmgl` (and `RIVERSTATUS` if training).  
        **Outputs:** `ln(p/(1-p))` (logit), `e^z`, and `p = e^z/(1+e^z)` where `z = β₀ + β·x`.
        
        **Two modes:**
        1. **Manual coefficients** — Key in intercept (β₀) and each β. Enter a single row or upload a file to score.
        2. **Train from data** — Upload CSV/Excel with all columns (including binary `RIVERSTATUS` 0/1). The app trains a logistic model, shows metrics, and exports raw-space coefficients for reproduction.
        
        **Graphs available:**
        - **Single row:** probability meter and predicted status.  
        - **Batch scoring:** bar chart of predicted water status (Clean vs Polluted); if true labels exist, you also get a confusion matrix.
        
        **Tip:** Use the threshold in the left sidebar to choose when `p` becomes "Polluted". Default is 0.50.
        """
    )

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown(
        """
        **RiverGuard Logistic (RGL)** is a simple decision-support tool that uses **logistic regression** 
        to estimate the probability that a river sample is **polluted (1)** versus **clean/safe (0)** 
        given common water-quality indicators.  
        
        The model computes a **logit** score `z = β₀ + Σ βᵢ xᵢ` and converts it to a probability `p` via the logistic function.
        You can supply your own coefficients (e.g., from published studies) or train them from a labelled dataset.
        
        **Important:** Logistic regression captures *linear* relationships in the log-odds. Model quality depends on data quality, 
        proper labelling of `RIVERSTATUS`, and whether the linearity assumption is reasonable. Always validate with held-out data and domain knowledge.
        """
    )

FEATURES = ["DOmgl", "BODmgl", "CODmgl", "SSmgl", "pH", "NH3Nmgl"]
TARGET = "RIVERSTATUS"

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Mode & Threshold")
    mode = st.radio("Choose how to compute:", ["Manual coefficients", "Train from data"], index=0)
    thresh = st.slider("Classification threshold (p ≥ threshold = Polluted)", 0.05, 0.95, 0.50, 0.01)
    st.caption("Column names must match exactly: " + ", ".join(FEATURES + [TARGET]))

# ---------- Helper Functions ----------
def ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    return df[cols]

def compute_outputs(df: pd.DataFrame, intercept: float, coefs: dict) -> pd.DataFrame:
    X = df[[c for c in FEATURES if c in df.columns]].astype(float)
    beta = np.array([coefs.get(col, 0.0) for col in X.columns], dtype=float)
    z = intercept + X.values @ beta
    ez = np.exp(z)
    p = ez / (1.0 + ez)
    out = df.copy()
    out["ln(p/(1-p))"] = z
    out["e^z"] = ez
    out["p=ez/(1+ez)"] = p
    out["PredictedStatus"] = (out["p=ez/(1+ez)"] >= thresh).astype(int)
    out["PredictedLabel"] = np.where(out["PredictedStatus"]==1, "Polluted", "Clean")
    return out

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
    counts = df_with_preds["PredictedLabel"].value_counts().reindex(["Clean","Polluted"]).fillna(0).astype(int)
    st.markdown(f"##### {title}")
    st.bar_chart(counts)

# ---------- UI ----------
if mode == "Manual coefficients":
    st.subheader("Manual Coefficients")
    with st.expander("Set coefficients (β) and intercept (β₀)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            beta0 = st.number_input("Intercept β₀", value=0.0, format="%.6f")
            b_DO  = st.number_input("β_DOmgl", value=0.0, format="%.6f")
        with c2:
            b_BOD = st.number_input("β_BODmgl", value=0.0, format="%.6f")
            b_COD = st.number_input("β_CODmgl", value=0.0, format="%.6f")
        with c3:
            b_SS  = st.number_input("β_SSmgl", value=0.0, format="%.6f")
            b_pH  = st.number_input("β_pH", value=0.0, format="%.6f")
            b_NH3 = st.number_input("β_NH3Nmgl", value=0.0, format="%.6f")

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

        # Display probability + status
        prob = float(result.loc[0, "p=ez/(1+ez)"])
        status = "Polluted" if prob >= thresh else "Clean"
        colm1, colm2 = st.columns(2)
        with colm1:
            st.metric("Predicted probability of pollution (p)", f"{prob:.3f}")
            st.progress(min(max(prob, 0.0), 1.0))
        with colm2:
            st.metric("Predicted status", status)

        st.markdown("##### Output table")
        st.dataframe(result, use_container_width=True)
        download_df(result, "manual_single_prediction")

    # --- Batch upload ---
    with tab2:
        st.write("Upload a CSV or Excel with columns: " + ", ".join(FEATURES + [TARGET]))
        file = st.file_uploader("Upload file", type=["csv", "xlsx"])
        if file:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            df = ensure_dataframe(df)
            st.dataframe(df.head(), use_container_width=True)
            result = compute_outputs(df, beta0, coefs)
            st.markdown("##### Output table")
            st.dataframe(result, use_container_width=True, height=420)
            status_bar_chart(result, "Predicted Water Status (Batch)")
            # If true labels present, show simple confusion vs threshold
            if TARGET in result.columns and TARGET in df.columns:
                if result[TARGET].dropna().shape[0] == result.shape[0]:
                    cm = pd.crosstab(df[TARGET].astype(int), result["PredictedStatus"].astype(int),
                                     rownames=["True"], colnames=["Pred"], dropna=False)
                    st.markdown("##### Confusion (if ground truth provided)")
                    st.dataframe(cm)
            download_df(result, "manual_batch_prediction")

else:
    st.subheader("Train from Data")
    st.markdown("- Upload a **CSV/Excel** containing the columns: " + ", ".join(FEATURES + [TARGET]) + " with target 0/1.")
    file = st.file_uploader("Upload training dataset", type=["csv", "xlsx"])
    if file:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df = ensure_dataframe(df)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            st.success(f"Loaded {len(df)} rows.")
            st.dataframe(df.head(), use_container_width=True)

            test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
            X = df[FEATURES].astype(float)
            y = df[TARGET].astype(int)

            scale = st.checkbox("Standardize features (recommended)", value=True)
            if scale:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            model = LogisticRegression(solver="liblinear", max_iter=1000)
            model.fit(X_train, y_train)

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

            # Reverse to raw-space betas if scaled
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

                inp = pd.DataFrame([{"DOmgl": DO, "BODmgl": BOD, "CODmgl": COD, "SSmgl": SS, "pH": pH, "NH3Nmgl": NH3}])
                res = score_df(inp)

                prob = float(res.loc[0, "p=ez/(1+ez)"])
                status = "Polluted" if prob >= thresh else "Clean"
                colm1, colm2 = st.columns(2)
                with colm1:
                    st.metric("Predicted probability of pollution (p)", f"{prob:.3f}")
                    st.progress(min(max(prob, 0.0), 1.0))
                with colm2:
                    st.metric("Predicted status", status)

                st.dataframe(res, use_container_width=True)
                download_df(res, "trained_model_single_prediction")

            with tab2:
                st.write("Upload a CSV/Excel to score (must include columns: " + ", ".join(FEATURES) + ")")
                f2 = st.file_uploader("Upload scoring file", type=["csv","xlsx"], key="scoreupload")
                if f2:
                    if f2.name.lower().endswith(".csv"):
                        df2 = pd.read_csv(f2)
                    else:
                        df2 = pd.read_excel(f2)
                    df2 = df2[[c for c in FEATURES if c in df2.columns]].astype(float)
                    res2 = score_df(df2)
                    st.dataframe(res2, use_container_width=True, height=420)
                    status_bar_chart(res2, "Predicted Water Status (Batch)")
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

st.markdown("---")
st.caption("Built with Streamlit • Logistic Regression (scikit-learn) • Columns: " + ", ".join(FEATURES + [TARGET]))

