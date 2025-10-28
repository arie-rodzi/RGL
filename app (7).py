# app.py
# RiverGuard — z, e^z, k Calculator (no training)
# Input: CSV/XLSX with DOmgl, BODmgl, CODmgl, SSmgl, pH, NH3Nmgl (RIVERSTATUS optional: 1=Clean, 0=Polluted)
# Output: Original data + z, e^z, k (+ Persistency vs WQI if RIVERSTATUS present), downloadable as Excel/CSV

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RGL — RiverGuard Logistic", layout="wide")
st.title("RGL — RiverGuard Logistic")

st.markdown("""
### ✅ Welcome to RiverGuard

Upload your file and the app will compute:
- **z** (logit score),
- **e^z**,
- **k = e^z/(1+e^z)** interpreted as **P(Polluted)**.

If **RIVERSTATUS** exists (**1 = Clean, 0 = Polluted**), the app will compute:
- **Persistency (%)** vs your labels, and
- a **confusion table** (using the threshold), **without adding any extra prediction column**.

Required columns: `DOmgl, BODmgl, CODmgl, SSmgl, pH, NH3Nmgl`  
Optional: `RIVERSTATUS` (1=Clean, 0=Polluted)
""")

FEATURES = ["DOmgl", "BODmgl", "CODmgl", "SSmgl", "pH", "NH3Nmgl"]
TARGET = "RIVERSTATUS"  # optional (1=Clean, 0=Polluted)

# ----- Coefficients & threshold -----
with st.sidebar:
    st.header("Coefficients (β)")
    st.caption("Defaults loaded from your formula. Edit as needed.")
    beta0 = st.number_input("β₀ (Intercept)", value=-11.326, format="%.6f")
    b_DO  = st.number_input("β_DOmgl", value=3.415, format="%.6f")
    b_BOD = st.number_input("β_BODmgl", value=-1.781, format="%.6f")
    b_COD = st.number_input("β_CODmgl", value=-0.271, format="%.6f")
    b_SS  = st.number_input("β_SSmgl", value=-0.035, format="%.6f")
    b_pH  = st.number_input("β_pH", value=0.000, format="%.6f")  # optional in equation
    b_NH3 = st.number_input("β_NH3Nmgl", value=5.853, format="%.6f")

    # Default threshold set to 0.90 as requested
    thresh = st.slider("Threshold for k = P(Polluted)", 0.05, 0.95, 0.90, 0.01)
    st.caption("Used only for Persistency/Confusion vs RIVERSTATUS. No extra prediction column is added.")

BETAS = {"DOmgl": b_DO, "BODmgl": b_BOD, "CODmgl": b_COD, "SSmgl": b_SS, "pH": b_pH, "NH3Nmgl": b_NH3}

def compute_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return original data + z, e^z, k (k is P(Polluted)). No predicted columns are added."""
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].astype(float).copy()
    beta_vec = np.array([BETAS.get(c, 0.0) for c in cols], dtype=float)

    z = beta0 + X.values @ beta_vec
    ez = np.exp(z)
    k = ez / (1.0 + ez)  # interpret as P(Polluted)

    out = df.copy()
    out["z"] = z
    out["e^z"] = ez
    out["k"] = k
    return out

def pretty_display(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty column names for UI (downloads keep simple names)."""
    return df.rename(columns={
        "z": "𝑧 = ln(p/(1−p))",
        "e^z": "e^𝑧",
        "k": "k = e^𝑧/(1+e^𝑧)"
    })

st.markdown("### Upload your data")
file = st.file_uploader("Upload CSV or Excel (.xlsx)", type=["csv", "xlsx"], accept_multiple_files=False)

if file:
    # Read input file
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        try:
            df = pd.read_excel(file, engine="openpyxl")
        except Exception as e:
            st.error("Failed to read Excel file with openpyxl. Ensure it's a valid .xlsx.")
            st.exception(e)
            df = pd.DataFrame()

    if not df.empty:
        # Compute results
        res = compute_outputs(df)

        # Preview — ONLY show (optional) RIVERSTATUS + z, e^z, k
        st.markdown("#### Preview (first 10 rows)")
        pretty = pretty_display(res.head(10))
        ordered_cols = ([TARGET] if TARGET in pretty.columns else []) + [
            "𝑧 = ln(p/(1−p))", "e^𝑧", "k = e^𝑧/(1+e^𝑧)"
        ]
        st.dataframe(pretty[ordered_cols], use_container_width=True, height=360)

        # Persistency vs RIVERSTATUS using threshold (internal classification)
        persist = None
        cm_df = None
        if TARGET in res.columns:
            try:
                true_vals = res[TARGET].astype(int).values  # 1=Clean, 0=Polluted
                # Internal classification from k without creating a column:
                # if k >= thr → Polluted → predicted RIVERSTATUS = 0; else Clean → 1
                pred_riverstatus = np.where(res["k"] >= thresh, 0, 1).astype(int)
                persist = float((true_vals == pred_riverstatus).mean() * 100.0)

                cm_df = pd.crosstab(
                    pd.Series(true_vals, name="True RIVERSTATUS (1=Clean,0=Polluted)"),
                    pd.Series(pred_riverstatus, name="Pred RIVERSTATUS (1=Clean,0=Polluted)"),
                    dropna=False
                ).astype(int)

                st.metric("Persistency (%)", f"{persist:.1f}%")
                st.markdown("##### Confusion (WQI vs Predicted from k)")
                st.dataframe(cm_df, use_container_width=True)
            except Exception:
                st.warning("Could not compute Persistency. Ensure RIVERSTATUS is coded 1=Clean, 0=Polluted.")

        # Build Excel with Results + Summary (Results has only original + z, e^z, k)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            res.to_excel(writer, index=False, sheet_name="Results")
            summary_rows = [
                ("k interpretation", "P(Polluted)"),
                ("Threshold for classification from k", thresh),
            ]
            if persist is not None:
                summary_rows.append(("Persistency (%)", round(persist, 1)))
            summary_df = pd.DataFrame(summary_rows, columns=["Item", "Value"])
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            if cm_df is not None:
                start_row = len(summary_rows) + 3
                cm_df.to_excel(writer, sheet_name="Summary", startrow=start_row)

        st.download_button(
            "⬇️ Download Excel (Results + Summary)",
            data=buffer.getvalue(),
            file_name="z_ez_k_with_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # Optional CSV (Results only)
        csv_bytes = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV (Results only)",
            data=csv_bytes,
            file_name="z_ez_k_results.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
st.caption("No model is trained. The app applies your β to compute z, e^z, and k (P(Polluted)). "
           "If RIVERSTATUS (1=Clean, 0=Polluted) is present, Persistency and Confusion are computed using k vs threshold—without adding a prediction column.")
