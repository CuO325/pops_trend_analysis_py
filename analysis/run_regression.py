import streamlit as st
import pandas as pd
from analysis.config import BayesConfig, build_beta_grid
from analysis.run_analysis import run_ols_analysis, run_mle_and_aic, run_fft_bayes

st.title("Bayesian Trend Analysis GUI")

# --- ① ファイル入力 ---
uploaded = st.file_uploader("データファイルを選択 (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("データプレビュー", df.head())

    # --- ② MCMC パラメータ設定 ---
    st.sidebar.header("MCMC 設定")

    n_walkers = st.sidebar.number_input("n_walkers", 4, 256, 32)
    n_steps = st.sidebar.number_input("n_steps", 100, 50000, 3000)
    burn_frac = st.sidebar.slider("burn_frac", 0.1, 0.9, 0.5)
    bw_factor = st.sidebar.number_input("bw_factor", 0.1, 5.0, 1.0)
    grid_size_lookup = st.sidebar.selectbox("grid_size_lookup", [1024, 2048, 4096])
    mode_loo = st.sidebar.checkbox("mode_loo (LOO)", value=True)

    beta_schedule = st.sidebar.selectbox(
        "βスケジュール",
        ["default", "coarse", "fine"]
    )

    cfg = BayesConfig(
        mode_loo=mode_loo,
        grid_size_lookup=grid_size_lookup,
        bw_factor=bw_factor,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_frac=burn_frac,
        beta_schedule=beta_schedule
    )

    st.sidebar.write("βグリッド:", build_beta_grid(cfg))

    # --- ③ 実行ボタン ---
    if st.button("OLS 実行"):
        st.subheader("OLS 結果")
        ols = run_ols_analysis(df)
        st.write(ols)

    if st.button("MLE + AIC 実行"):
        st.subheader("MLE 結果")
        mle = run_mle_and_aic(df)
        st.write(mle)

    if st.button("FFT-KDE ベイズ 実行"):
        st.subheader("FFT-KDE ベイズ結果")
        bayes = run_fft_bayes(df, cfg)
        st.write(bayes)
