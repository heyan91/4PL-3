import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import io  # ✅ Fix for StringIO

# --- 4PL function ---
def four_param_logistic(x, a, b, c, d):
    return d + (a - d) / (1.0 + (x / c)**b)

# --- Inverse 4PL for LOD ---
def inverse_4pl(y, a, b, c, d):
    return c * ((a - d) / (y - d) - 1)**(1 / b)

st.title("🔬 4PL Curve Fitting and Limit of Detection (LOD) Calculator")

st.write("Paste your data in this format (tab or comma separated):")
st.code("Concentration\tRep1\tRep2\tRep3\tRep4\n0.01\t0.12\t0.13\t0.14\t0.13\n0.1\t0.22\t0.21\t0.23\t0.24\n...\nNC\t0.02\t0.03\t0.02\t0.01\t... (8 replicates)", language="text")

data_input = st.text_area("📋 Paste your data below:")

if data_input:
    try:
        # ✅ Use built-in io.StringIO instead of deprecated pandas.compat
        df = pd.read_csv(io.StringIO(data_input), sep=None, engine="python")
        st.success("✅ Data successfully read!")

        # Parse concentrations and replicates
        concentrations = []
        signals = []

        for i, row in df.iterrows():
            label = str(row[0]).strip().lower()
            values = row[1:].dropna().astype(float).values
            if label == 'nc':
                nc_values = values
            else:
                concentrations.append(float(label))
                signals.append(values)

        conc_array = np.array(concentrations)
        means = np.array([np.mean(s) for s in signals])
        sds = np.array([np.std(s) for s in signals])

        # Fit 4PL
        p0 = [min(means), 1, np.median(conc_array), max(means)]
        popt, _ = curve_fit(four_param_logistic, conc_array, means, p0, maxfev=10000)

        # Predict curve
        x_fit = np.logspace(np.log10(min(conc_array)*0.1), np.log10(max(conc_array)*10), 200)
        y_fit = four_param_logistic(x_fit, *popt)

        # Calculate LOD
        nc_mean = np.mean(nc_values)
        nc_std = np.std(nc_values)
        lod_signal = nc_mean + 3 * nc_std

        try:
            lod_conc = inverse_4pl(lod_signal, *popt)
        except Exception:
            lod_conc = np.nan

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.errorbar(conc_array, means, yerr=sds, fmt='o', label="Mean ± SD")
        for x, y in zip(conc_array, signals):
            ax.scatter([x]*len(y), y, alpha=0.3, color='gray')  # raw points
        ax.plot(x_fit, y_fit, label="4PL Fit", color='blue')
        ax.axhline(lod_signal, color='red', linestyle='--', label=f"LOD Signal = {lod_signal:.3f}")
        ax.axvline(lod_conc, color='green', linestyle='--', label=f"LOD Concentration = {lod_conc:.3g}")
        ax.set_xscale('log')
        ax.set_xlabel("Concentration")
        ax.set_ylabel("Signal")
        ax.set_title("4PL Curve with LOD")
        ax.legend()
        st.pyplot(fig)

        st.success(f"✅ LOD Signal = {lod_signal:.4f}")
        st.success(f"✅ Estimated LOD Concentration = {lod_conc:.4g}")

    except Exception as e:
        st.error("⚠️ Error processing data: " + str(e))

