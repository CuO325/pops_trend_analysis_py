#app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from modules.data_loader import create_timeseries
from modules.stl_module import calc_stl
from modules.classical_break import piecewise_fit, linear_regression
from modules.bayes_break import (
    run_breakpoint1_mcmc, run_breakpoint2_mcmc,
    plot_posterior_trend, plot_posterior_trend_2break
)
from modules.ti_module import run_ti_numpyro_breakpoint

st.set_page_config(layout="wide")
st.title("POPs Monitoring Trend Analysis (Modular v1.2)")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is None:
    st.info("データをアップロードしてください")
    st.stop()

import pandas as pd
df = pd.read_csv(uploaded_file)
substances = sorted(df["調査対象物質"].unique())

with st.sidebar:
    st.header("Settings")
    substance = st.selectbox("Substance", substances)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Data",
    "2. STL",
    "3. Classical Breakpoints",
    "4. Bayesian Breakpoints",
    "5. TI Model Comparison",
    "6. Multi-substance Comparison"
])

# -----------------------------
# 1. Data
# -----------------------------
with tab1:
    ts_log = create_timeseries(df, substance)
    st.session_state["ts_log"] = ts_log

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts_log.index, ts_log.values, "o-")
    ax.set_title(f"{substance} RAW log concentration")
    st.pyplot(fig)

# -----------------------------
# 2. STL
# -----------------------------
with tab2:
    ts_log = st.session_state.get("ts_log")
    if ts_log is None:
        st.warning("タブ1でデータを読み込んでください")
        st.stop()

    res = calc_stl(ts_log)
    trend = res.trend.dropna()
    st.session_state["trend"] = trend

    fig = res.plot()
    fig.set_size_inches(12, 8)
    st.pyplot(fig)

# -----------------------------
# 3. Classical Breakpoints
# -----------------------------
with tab3:
    trend = st.session_state.get("trend")
    if trend is None:
        st.warning("タブ2でSTLを実行してください")
        st.stop()

    n_segments = st.slider("Number of segments", 2, 6, 3)
    pwlf_model, break_dates, slopes = piecewise_fit(trend, n_segments)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(trend.index, trend.values, "o-", alpha=0.6)

    x_hat = np.linspace(0, len(trend)-1, 300)
    y_hat = pwlf_model.predict(x_hat)
    start_date = trend.index[0]
    date_hat = [start_date + pd.DateOffset(months=int(i)) for i in x_hat]

    ax.plot(date_hat, y_hat, "r-", linewidth=3)
    st.pyplot(fig)

    st.write("Break dates:", break_dates)

# -----------------------------
# 4. Bayesian Breakpoints
# -----------------------------
with tab4:
    trend = st.session_state.get("trend")
    if trend is None:
        st.warning("タブ2でSTLを実行してください")
        st.stop()

    x = np.arange(len(trend))
    y = trend.values

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run 1-break MCMC"):
            mcmc1 = run_breakpoint1_mcmc(x, y)
            st.session_state["bayes_1"] = mcmc1.get_samples()

    with col2:
        if st.button("Run 2-break MCMC"):
            mcmc2 = run_breakpoint2_mcmc(x, y)
            st.session_state["bayes_2"] = mcmc2.get_samples()

    if "bayes_1" in st.session_state:
        st.write("### 1-break Posterior Fit")
        fig = plot_posterior_trend(x, y, st.session_state["bayes_1"])
        st.pyplot(fig)

    if "bayes_2" in st.session_state:
        st.write("### 2-break Posterior Fit")
        fig = plot_posterior_trend_2break(x, y, st.session_state["bayes_2"])
        st.pyplot(fig)

# -----------------------------
# 5. TI Model Comparison
# -----------------------------
with tab5:
    trend = st.session_state.get("trend")
    if trend is None:
        st.warning("タブ2でSTLを実行してください")
        st.stop()

    x = np.arange(len(trend))
    y = trend.values

    if st.button("Run TI"):
        logZ1, beta1, E1 = run_ti_numpyro_breakpoint(x, y, model="1break")
        logZ2, beta2, E2 = run_ti_numpyro_breakpoint(x, y, model="2break")

        st.session_state["ti"] = {
            "logZ1": logZ1,
            "logZ2": logZ2,
            "logBF": logZ2 - logZ1,
            "beta": beta1,
            "E1": E1,
            "E2": E2
        }

    if "ti" in st.session_state:
        ti = st.session_state["ti"]
        st.write("logZ(1-break):", ti["logZ1"])
        st.write("logZ(2-break):", ti["logZ2"])
        st.write("logBF:", ti["logBF"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ti["beta"], ti["E1"], label="1-break")
        ax.plot(ti["beta"], ti["E2"], label="2-break")
        ax.legend()
        st.pyplot(fig)

# -----------------------------
# 6. Multi-substance Comparison
# -----------------------------
with tab6:
    selected = st.multiselect("Select substances", substances, default=[substance])

    fig, ax = plt.subplots(figsize=(12, 6))
    for sub in selected:
        ts = create_timeseries(df, sub)
        res = calc_stl(ts)
        trend_sub = res.trend.dropna()
        ax.plot(trend_sub.index, trend_sub.values, label=sub)

    ax.legend()
    st.pyplot(fig)
