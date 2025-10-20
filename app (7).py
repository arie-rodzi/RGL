# app.py
# RiverGuard — z, e^z, k Calculator (no training)
# Input: CSV/XLSX with DOmgl, BODmgl, CODmgl, SSmgl, pH, NH3Nmgl (RIVERSTATUS optional)
# Output: Original data + z, e^z, k (+ Persistency vs WQI if RIVERSTATUS present), downloadable as Excel/CSV

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RiverGuard — z, e^z, k Calculator", layout="wide")
st.title("RiverGuard — z, e^z, k Calculator")

st.markdown("""
### ✅ Welcome to RiverGuard

This app helps you **calculate water pollution indicators** using your existing data.  
Just upload your file — the app will do the rest.

---

### 🔹 What You’ll Get from the App
For every row in your dataset, the app will automatically:
- Calculate a **logistic score (z)**
- Convert it to **eⁿ (exponential value)**
- Compute a **probability value (k = e^z / (1 + e^z))**

If your data also includes **RIVERSTATUS** (0 = Clean, 1 = Polluted), the app will:
- Compare its prediction with your WQI label
- Show **Persistency (%)**
- Display a **confusion table** (accuracy breakdown)

---

### 📂 What You Need to Upload
Your file must contain these columns exactly:
`DOmgl, BODmgl, CODmgl, SSmgl, pH, NH3Nmgl`

Optional column (if available):
`RIVERSTATUS`

You can upload **Excel or CSV** files.

---

### ⚙️ Customize Your Coefficients
On the left side, you can:
- Edit the logistic regression coefficients (β values)
- Adjust the prediction threshold

**Note:** Any value of `k ≥ 0.90` is automatically classified as *Polluted*.

---

### 📥 What You Can Download
After processing, you’ll get:
- An **Excel file** with your original data plus new columns (z, e^z, k)
- A **Summary sheet** (threshold, persistency, confusion table)
- Or a **CSV version** if preferred

---

### ✅ No Training Needed
This app does **not** build or fit any model — it only calculates results based on the coefficients you provide.

You're ready to start! Upload your data above.
""")


FEATURES = ["DOmgl", "BODmgl", "CODmgl", "SSmgl", "pH", "NH3Nmgl"]
TARGET = "RIVERSTATUS"  # optional (WQI-derived 0/1 label)

# ----- Coefficients & threshold -----
with st.sidebar:
    st.header("Coefficients (β)")
    st.caption("Defaults loaded from your formula. Edit as needed.")
    beta0 = st.number_input("β₀ (Intercept)", value=-11.326, format="%.6f")
    b_DO  = st.number_input("β_DOmgl", value=3.415, format="%.6f")
    b_BOD = st.number_input("β_BODmgl", value=-1.781, format="%.6f")
    b_COD = st.number_input("β_CODmgl", value=-0.271, format="%.6f")
    b_SS  = st.number_input("β_SSmgl", value=-0.035, format="%.6f")
    b_pH  = st.number_input("β_pH", value=0.000, format="%.6f")  # not used in your equation by default
    b_NH3 = st.number_input("β_NH3Nmgl", value=5.853, format="%.6f")

    thresh = st.slider("Threshold for class(k) = Polluted (optional)", 0.05, 0.95, 0.50, 0.01)
    st.caption("Used only to compute PredictedStatus and Persistency vs WQI (if RIVERSTATUS exists).")

BETAS = {"DOmgl": b_DO, "BODmgl": b_BOD, "CODmgl": b_COD, "SSmgl": b_SS, "pH": b_pH, "NH3Nmgl": b_NH3}

def compute_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute z, e^z, k (and predicted class/label if threshold given)."""
    X = df[[c for c in FEATURES if c in df.columns]].astype(float).copy()
    beta_vec = np.array([BETAS.get(c, 0.0) for c in X.columns], dtype=float)
    z = beta0 + X.values @ beta_vec
    ez = np.exp(z)
    k = ez / (1.0 + ez)
    out = df.copy()
    out["z"] = z
    out["e^z"] = ez
    out["k"] = k
    out["PredictedStatus"] = (out["k"] >= thresh).astype(int)
    out["PredictedLabel"] = np.where(out["PredictedStatus"] == 1, "Polluted", "Clean")
    return out

def pretty_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format output column headers for UI display (downloads keep simple names)."""
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
            # requires openpyxl
            df = pd.read_excel(file, engine="openpyxl")
        except Exception as e:
            st.error("Failed to read Excel file with openpyxl. Ensure it's a valid .xlsx.")
            st.exception(e)
            df = pd.DataFrame()

    if not df.empty:
        # Compute results
        res = compute_outputs(df)

        # Preview
        st.markdown("#### Preview (first 10 rows)")
        st.dataframe(pretty_display(res.head(10)), use_container_width=True, height=360)

        # Persistency vs WQI (if available)
        persist = None
        cm_df = None
        if TARGET in res.columns:
            try:
                true_vals = res[TARGET].astype(int).values
                pred_vals = res["PredictedStatus"].astype(int).values
                persist = float((true_vals == pred_vals).mean() * 100.0)
                cm_df = pd.crosstab(
                    res[TARGET].astype(int),
                    res["PredictedStatus"].astype(int),
                    rownames=["True (WQI)"],
                    colnames=["Pred (k ≥ thr)"],
                    dropna=False
                ).astype(int)
                st.metric("Persistency (%)", f"{persist:.1f}%")
                st.markdown("##### Confusion (WQI vs Predicted)")
                st.dataframe(cm_df, use_container_width=True)
            except Exception:
                st.warning("Could not compute Persistency. Ensure RIVERSTATUS is 0/1.")

        # Build Excel with Results + Summary
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            res.to_excel(writer, index=False, sheet_name="Results")
            # Summary sheet
            summary_rows = [("Threshold", thresh)]
            if persist is not None:
                summary_rows.append(("Persistency (%)", round(persist, 1)))
            summary_df = pd.DataFrame(summary_rows, columns=["Item", "Value"])
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            if cm_df is not None:
                # append confusion matrix below summary
                start_row = len(summary_rows) + 3
                cm_df.to_excel(writer, sheet_name="Summary", startrow=start_row)

        st.download_button(
            "⬇️ Download Excel (Results + Summary)",
            data=buffer.getvalue(),
            file_name="z_ez_k_with_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # Optional CSV download
        csv_bytes = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV (Results only)",
            data=csv_bytes,
            file_name="z_ez_k_results.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
st.caption("This tool does not train any model. It applies your coefficients to compute z, e^z, and k per row. "
           "If RIVERSTATUS (WQI-based) is present, it reports Persistency (%) and a confusion table.")
