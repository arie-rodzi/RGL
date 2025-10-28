# app.py
# RiverGuard — z, e^z, k Calculator (no training)
# Input: CSV/XLSX with DOmgl, BODmgl, CODmgl, SSmgl, pH, NH3Nmgl (RIVERSTATUS optional: 1=Clean, 0=Polluted)
# Output: Original data + z, e^z, k (+ Persistency/Confusion vs WQI if RIVERSTATUS present), downloadable as Excel/CSV

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RGL — RiverGuard Logistic", layout="wide")
st.title("RGL — RiverGuard Logistic")

st.markdown("""
Upload your file and the app will compute:
- **z** (logit score),
- **e^z**,
- **k = e^z/(1+e^z)** interpreted as the **model's predicted probability** (P of one class).

If **RIVERSTATUS** exists (**1 = Clean, 0 = Polluted**), the app:
- **Auto-calibrates** whether *k* behaves like **P(Polluted)** or **P(Clean)**, and
- finds a good **threshold** by sweeping 0.05→0.95 to maximize accuracy,
- then reports **Persistency (%)** and a **confusion table**.

No extra prediction column is added — **k is the only prediction value**.
""")

FEATURES = ["DOmgl", "BODmgl", "CODmgl", "SSmgl", "pH", "NH3Nmgl"]
TARGET = "RIVERSTATUS"  # optional (WQI-derived: 1=Clean, 0=Polluted)

# ----- Coefficients & user controls -----
with st.sidebar:
    st.header("Coefficients (β)")
    st.caption("Edit if needed (defaults from your formula).")
    beta0 = st.number_input("β₀ (Intercept)", value=-11.326, format="%.6f")
    b_DO  = st.number_input("β_DOmgl", value=3.415, format="%.6f")
    b_BOD = st.number_input("β_BODmgl", value=-1.781, format="%.6f")
    b_COD = st.number_input("β_CODmgl", value=-0.271, format="%.6f")
    b_SS  = st.number_input("β_SSmgl", value=-0.035, format="%.6f")
    b_pH  = st.number_input("β_pH", value=0.000, format="%.6f")
    b_NH3 = st.number_input("β_NH3Nmgl", value=5.853, format="%.6f")

    st.divider()
    k_meaning = st.radio("k meaning", ["Auto (use labels if available)", "P(Polluted)", "P(Clean)"], index=0)
    thresh = st.slider("Threshold (used only for evaluation)", 0.05, 0.95, 0.90, 0.01)
    st.caption("If RIVERSTATUS exists and 'Auto' is selected, the app will choose the best orientation & threshold.")

BETAS = {"DOmgl": b_DO, "BODmgl": b_BOD, "CODmgl": b_COD, "SSmgl": b_SS, "pH": b_pH, "NH3Nmgl": b_NH3}

# ----- Helpers -----
def compute_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return original data + z, e^z, k. (k is the ONLY prediction value shown.)"""
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].astype(float).copy()
    beta_vec = np.array([BETAS.get(c, 0.0) for c in cols], dtype=float)

    z = beta0 + X.values @ beta_vec
    ez = np.exp(z)
    k = ez / (1.0 + ez)

    out = df.copy()
    out["z"] = z
    out["e^z"] = ez
    out["k"] = k
    return out

def pretty_display(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty column names for UI (downloads keep raw names)."""
    return df.rename(columns={
        "z": "𝑧 = ln(p/(1−p))",
        "e^z": "e^𝑧",
        "k": "k = e^𝑧/(1+e^𝑧)"
    })

def classify_from_k(k_vals: np.ndarray, meaning: str, thr: float) -> np.ndarray:
    """
    Return predicted RIVERSTATUS coded 1=Clean, 0=Polluted from (k, meaning, thr).
    """
    if meaning == "P(Polluted)":
        # if k ≥ thr ⇒ Polluted ⇒ 0; else Clean ⇒ 1
        return np.where(k_vals >= thr, 0, 1).astype(int)
    else:  # "P(Clean)"
        # if k ≥ thr ⇒ Clean ⇒ 1; else Polluted ⇒ 0
        return np.where(k_vals >= thr, 1, 0).astype(int)

def best_orientation_and_threshold(true_vals: np.ndarray, k_vals: np.ndarray):
    """Grid-search thresholds & both orientations; return (best_meaning, best_thr, best_acc)."""
    meanings = ["P(Polluted)", "P(Clean)"]
    thrs = np.round(np.arange(0.05, 0.951, 0.01), 2)
    best = ("P(Polluted)", 0.50, -1.0)
    for m in meanings:
        for t in thrs:
            pred = classify_from_k(k_vals, m, t)
            acc = (pred == true_vals).mean()
            if acc > best[2]:
                best = (m, float(t), float(acc))
    return best

# ----- UI: Upload & run -----
st.markdown("### Upload your data")
file = st.file_uploader("Upload CSV or Excel (.xlsx)", type=["csv", "xlsx"], accept_multiple_files=False)

if file:
    # Read input
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
        # Compute z, e^z, k
        res = compute_outputs(df)

        # Preview — show (optional) RIVERSTATUS + z, e^z, k only
        st.markdown("#### Preview (first 10 rows)")
        pretty = pretty_display(res.head(10))
        ordered_cols = ([TARGET] if TARGET in pretty.columns else []) + [
            "𝑧 = ln(p/(1−p))", "e^𝑧", "k = e^𝑧/(1+e^𝑧)"
        ]
        st.dataframe(pretty[ordered_cols], use_container_width=True, height=360)

        # Persistency / Confusion if labels exist
        persist = None
        cm_df = None
        chosen_meaning = None
        chosen_thr = None

        if TARGET in res.columns:
            try:
                true_vals = res[TARGET].astype(int).values  # 1=Clean, 0=Polluted
                k_vals = res["k"].values

                if k_meaning.startswith("Auto"):
                    chosen_meaning, chosen_thr, best_acc = best_orientation_and_threshold(true_vals, k_vals)
                    st.info(
                        f"Auto-calibrated: k interpreted as **{chosen_meaning}** with threshold **{chosen_thr:.2f}** "
                        f"(accuracy {best_acc*100:.1f}%)."
                    )
                else:
                    chosen_meaning = "P(Polluted)" if "Polluted" in k_meaning else "P(Clean)"
                    chosen_thr = float(thresh)

                pred = classify_from_k(k_vals, chosen_meaning, chosen_thr)
                persist = float((pred == true_vals).mean() * 100.0)

                cm_df = pd.crosstab(
                    pd.Series(true_vals, name="True RIVERSTATUS (1=Clean,0=Polluted)"),
                    pd.Series(pred, name="Pred RIVERSTATUS (1=Clean,0=Polluted)"),
                    dropna=False
                ).astype(int)

                st.metric("Persistency (%)", f"{persist:.1f}%")
                st.caption(f"k interpreted as **{chosen_meaning}**, threshold = **{chosen_thr:.2f}**")
                st.markdown("##### Confusion (WQI vs Predicted from k)")
                st.dataframe(cm_df, use_container_width=True)

            except Exception:
                st.warning("Could not compute Persistency. Ensure RIVERSTATUS is coded 1=Clean, 0=Polluted.")

        # Build Excel with Results + Summary (Results: only original + z, e^z, k)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            res.to_excel(writer, index=False, sheet_name="Results")

            summary_rows = [
                ("k interpretation",
                 chosen_meaning if chosen_meaning else ("P(Polluted)" if "Polluted" in k_meaning else "P(Clean)")),
                ("Threshold for classification from k",
                 chosen_thr if chosen_thr is not None else float(thresh)),
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
st.caption(
    "No model is trained. The app applies your β to compute z, e^z, and k per row. "
    "If RIVERSTATUS (1=Clean, 0=Polluted) is present, it auto-calibrates k's meaning and the threshold to maximize accuracy."
)
