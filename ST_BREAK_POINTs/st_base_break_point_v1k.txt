#streamlit run st_bayes_break_point_v1k.py
#2026/5/24 Hiroaki Shiraishi

### AI Assistance
#This repository was developed with assistance from Microsoft Copilot and Google Gemini.  
#All final code, design decisions, and validations were made by the author.

#MIT Licence
import os
import streamlit as st
from datetime import datetime, date
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from modules.data_loader import create_timeseries
from modules.stl_module import calc_k_Shape, calc_stl, plot_stl_components, create_raw_plot, create_mode_plot, fig_to_png_bytes, plot_multi_substances
from modules.classical_break import piecewise_fit, linear_regression,  calc_moving_slope,  run_classical_break_analysis, plot_classical_break
from modules.bayes_break import (
    run_breakpoint1_mcmc, run_breakpoint2_mcmc,
    plot_posterior_trend, plot_posterior_trend_2break
)
from modules.ti_module import run_ti_numpyro_breakpoint
from modules.utils import add_regulation_lines, add_regulation_lines_date
from modules.report_generator import  generate_html_report_full
import statsmodels.api as sm
#import datetime

st.set_page_config(layout="wide")
st.title("POPs Monitoring Break Point Regression version 1.0k")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is None:
    st.info("Please upload your data in CSV format. The CSV should have columns: '調査対象物質', '年月', '濃度'.")
    st.stop()

import pandas as pd
df = pd.read_csv(uploaded_file)
substances = sorted(df["調査対象物質"].unique())

#print(df[df["調査対象物質"].str.contains("HCH", na=False)])

#for chem in df["調査対象物質"].unique():
#    if "HCH" in chem:
#        print(chem, [hex(ord(c)) for c in chem])
#df["調査対象物質"] = df["調査対象物質"].str.strip()

#-------------------
# サイドバー
#-------------------

with st.sidebar:
    st.header("Settings")
    substance = st.selectbox("Substance", substances)
    yaxis_mode = "STL"
    yaxis_mode = st.radio("Y-axis display other than basian",[ "STL", "RAW",  "Normalized", "Moving slope" ] )

    st.write(f"display yaxis mode = {yaxis_mode}")    
    window = st.slider("Sliding window", 6, 36, 12)
    pops_year = st.checkbox("plot pops regulation year")
    # ★ 年月日でフィルタ開始日を指定
    #import datetime
    filter_start_date = st.date_input(
        "Filter data from",
        value=date(2010, 4, 1)
    )




# 💡 物質固有のセッションキーを一元定義
key_df_sub = f"df_sub_{substance}"
key_trend = f"trend_{substance}"
key_seasonal = f"seasonal_{substance}"
key_resid = f"resid_{substance}"
key_pwlf_res = f"pwlf_res_{substance}"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Data",
    "2. STL",
    "3. Classical Breakpoints",
    "4. Bayesian Breakpoints",
    "5. TI (Thermodynamic Integration)",
    "6. Batch"
])

# -----------------------------------------------------------------
# タブ1: データ画面 (RAW表示 & mode切替グラフ)
# -----------------------------------------------------------------

with tab1:
    # データの読み込みと前処理 古いデータのフィルタリングを追加
    ts_log = create_timeseries(df, substance)
    ts_log = ts_log[ts_log.index >= pd.Timestamp(filter_start_date)]


    st.session_state[key_df_sub] = ts_log

    st.subheader("Time Series Plots")

    # --- RAW グラフ ---
    fig_raw = create_raw_plot(
        ts_log,
        substance,
        pops_year=pops_year,
        add_regulation_lines=add_regulation_lines
    )
    st.pyplot(fig_raw)

    st.download_button(
        label="🖼️ RAWグラフをダウンロード",
        data=fig_to_png_bytes(fig_raw),
        file_name=f"{substance}_raw_plot.png",
        mime="image/png",
        key="download_raw_plot_btn"
    )

    # --- mode 切替グラフ ---
    if yaxis_mode != "RAW":
        st.markdown("---")

        fig_mode = create_mode_plot(
            ts_log,
            substance,
            yaxis_mode,
            window,
            pops_year=pops_year,
            add_regulation_lines=add_regulation_lines,
            calc_stl=calc_stl,
            calc_moving_slope=calc_moving_slope
        )
        if yaxis_mode == "STL":
            docs_link = "時系列分解（STL: Seasonal and Trend decomposition using Loess）についての論文: http://bit.ly/stl1990"
            st.markdown("### 時系列分解（STL: Seasonal and Trend decomposition using Loess）\nSTLは、時系列データをトレンド、季節性、残差の3つの成分に分解する手法です。これにより、データの長期的な傾向や周期的なパターンを明らかにできます。\n\n"
                        f" 📚 詳細は論文(Cleveland, et.al. 1990)を参照: {docs_link}")
        elif yaxis_mode == "Normalized":
            st.markdown("### Normalized Trend\n各物質のトレンドを0-1の範囲に正規化して表示しています。")
        elif yaxis_mode == "Moving slope":
            st.markdown("### Moving Slope\n移動傾きは、指定したウィンドウサイズでトレンドの傾きを計算し、時間とともにどのように変化しているかを示します。")    

        st.write(f"### {yaxis_mode} Plot" )
        st.pyplot(fig_mode)

        st.download_button(
            label=f"🖼️ {yaxis_mode} グラフをダウンロード",
            data=fig_to_png_bytes(fig_mode),
            file_name=f"{substance}_{yaxis_mode}_plot.png",
            mime="image/png",
            key=f"download_{yaxis_mode}_plot_btn"
        )

    st.session_state[f"fig_raw_{substance}"] = fig_raw


# ------------------------------------
# Tab 2: STL (物質別対応化)
# ------------------------------------

with tab2:
    st.header(f"STL Decomposition: {substance}")
    target_series = st.session_state.get(key_df_sub)
    if target_series is None:
        st.warning("タブ1でデータを確認してください")
        st.stop()

    with st.spinner("Decomposing..."):
        try:
            res = calc_stl(target_series)
            trend = res.trend.dropna()
        except Exception as e:
            st.warning(f"{substance} の STL 計算でエラー発生：{e}")
            st.stop()

        #res = calc_stl(target_series)
        st.session_state[key_trend] = res.trend
        st.session_state[key_seasonal] = res.seasonal
        st.session_state[key_resid] = res.resid
        st.success(f"STL Done for {substance}!")

    if key_trend in st.session_state:
        trend = st.session_state[key_trend]
        seasonal = st.session_state[key_seasonal]
        resid = st.session_state[key_resid]

        fig_stl = plot_stl_components(
            target_series, trend, seasonal, resid, substance
        )

        st.pyplot(fig_stl)
        st.session_state[f"fig_stl_{substance}"] = fig_stl


# ------------------------------------
# Tab 3: Classical Breakpoints (Y軸管理・物質別完全対応版)
# ------------------------------------
#from modules.classical_break_analysis import run_classical_break_analysis

with tab3:
    st.subheader("Classical Breakpoint Analysis (pwlf + OLS)")

    trend_global = st.session_state.get(f"trend_{substance}")
    ts_log = st.session_state.get(f"df_sub_{substance}")


    if trend_global is None:
        st.warning("タブ2でSTLを実行してください")
        st.stop()

    n_segments = st.slider("Number of segments", 2, 6, 3, key=f"slider_{substance}")


    current_conditions = {
        "n_segments": n_segments,
        "yaxis_mode": yaxis_mode
    }

    key_pwlf_res = f"pwlf_res_{substance}"
    key_calc_conditions = f"pwlf_cond_{substance}"

    if st.button("Run Classical Piecewise Fit", key=f"btn_tab3_{substance}"):
        fig, df_table, pwlf_model = run_classical_break_analysis(
            trend_global,
            ts_log,
            yaxis_mode,
            n_segments,
            piecewise_fit,
            calc_moving_slope,
            pops_year,
            add_regulation_lines,
            substance
        )
        st.session_state[key_pwlf_res] = (fig, df_table)
        st.session_state[key_calc_conditions] = current_conditions
        st.success(f"{n_segments-1}-break Classical Fit Done for {substance}!")

    # --- 条件一致チェック ---
    if key_pwlf_res in st.session_state:
        saved_conditions = st.session_state.get(key_calc_conditions, {})
        if saved_conditions != current_conditions:
            st.info("⚠️ 設定が変更されました。再計算してください。")
            st.stop()

        classical_fig, classical_table = st.session_state[key_pwlf_res]

        st.pyplot(classical_fig)
        st.session_state[f"classical_fig_{substance}"] = classical_fig

        st.subheader("Interval Regression Statistics")
        st.dataframe(classical_table, use_container_width=True)
        st.session_state[f"classical_table_{substance}"] = classical_table


# ------------------------------------
# Tab 4: Bayesian Breakpoints
# ------------------------------------
with tab4:
    #trend = st.session_state.get("trend")
    trend = st.session_state.get(f"trend_{substance}")  # 👈 こう書き換える    
    if trend is None:
        st.warning("タブ2でSTLを実行してください")
        st.stop()

    # X軸は連番、y軸はトレンド値を取得
    x = np.arange(len(trend))
    y = trend.values

    # 物質固有のサンプルの部屋（キー）を定義
    key_samples1 = f"samples1_{substance}"
    key_samples2 = f"samples2_{substance}"

    if st.button("Run MCMC", key=f"btn_mcmc_{substance}"): # 👈 keyを追加して安全に
        # 計算効率と安定化のためにXを標準化（規格化）
        x_mean, x_std = x.mean(), x.std()
        x_stdized = (x - x_mean) / x_std

        with st.spinner("Running 1-break..."):
            mcmc1 = run_breakpoint1_mcmc(x_stdized, y)
            st.session_state[key_samples1] = mcmc1.get_samples()

        with st.spinner("Running 2-break..."):
            mcmc2 = run_breakpoint2_mcmc(x_stdized, y)
            st.session_state[key_samples2] = mcmc2.get_samples()

        st.success(f"MCMC Done for {substance}!")

    # 現在選択中の物質の計算結果がセッションにある場合のみ描画
    if key_samples1 in st.session_state and key_samples2 in st.session_state:
        s1 = st.session_state[key_samples1]
        s2 = st.session_state[key_samples2]
        s1 = st.session_state[key_samples1]
        s2 = st.session_state[key_samples2]

        # 共通の補助計算（標準化からの復元用）
        x_mean, x_std = x.mean(), x.std()
        x_stdized = (x - x_mean) / x_std
        tick_indices = np.linspace(0, len(trend) - 1, 8, dtype=int)
        tick_labels = [trend.index[i].strftime('%Y-%m') for i in tick_indices]

        # ==================================================
        # 【重要】mu_mean_1, mu_mean_2 の定義と計算
        # ==================================================
        # 1-break モデルの全サンプルにおける μ の平均 (E[μ]) を計算
        mus_1 = []
        for a, b1_val, b2_val, t in zip(s1["alpha"], s1["beta1"], s1["beta2"], s1["tau"]):
            mus_1.append(a + b1_val * x_stdized + b2_val * np.maximum(0, x_stdized - t))
        mu_mean_1 = np.mean(np.array(mus_1), axis=0)  # shape: (n_time,)

        # 2-break モデルの全サンプルにおける μ の平均 (E[μ]) を計算
        mus_2 = []
        for a, b1_val, b2_val, b3_val, t1_val, t2_val in zip(s1["alpha"], s1["beta1"], s2["beta2"], s2["beta3"], s2["tau1"], s2["tau2"]):
            # ※ alpha, beta1 は s2 のものを使うため修正
            pass
        
        # 上記ループを安全に書き直した形式が以下です：
        mus_2 = []
        for a, b1_val, b2_val, b3_val, t1_val, t2_val in zip(s2["alpha"], s2["beta1"], s2["beta2"], s2["beta3"], s2["tau1"], s2["tau2"]):
            mus_2.append(a + b1_val * x_stdized + b2_val * np.maximum(0, x_stdized - t1_val) + b3_val * np.maximum(0, x_stdized - t2_val))
        mu_mean_2 = np.mean(np.array(mus_2), axis=0)  # shape: (n_time,)



        # 共通の補助計算（標準化からの復元用）
        x_mean, x_std = x.mean(), x.std()
        x_stdized = (x - x_mean) / x_std
        tick_indices = np.linspace(0, len(trend) - 1, 8, dtype=int)
        tick_labels = [trend.index[i].strftime('%Y-%m') for i in tick_indices]

        # ==================================================
        # ① 上段：不確実性を見る左右2分割プロット
        # ==================================================
        st.write("### ⏱️ Mean of Posterior Fitted Values  E[μ]:")
        st.caption("MCMCサンプリングによって、各サンプルで計算した μ の平均。Posterior Predictive Mean(事後予測平均)　と95% CI (confidence interval)　")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 1-break Posterior Trend (Uncertainty)")
            fig1 = plot_posterior_trend(x_stdized, y, s1)
            ax1 = fig1.gca()
            ax1.set_xticks((tick_indices - x_mean) / x_std)
            ax1.set_xticklabels(tick_labels, rotation=45)
            st.pyplot(fig1)

        with col2:
            st.write("### 2-break Posterior Trend (Uncertainty)")
            fig2 = plot_posterior_trend_2break(x_stdized, y, s2)
            ax2 = fig2.gca()
            ax2.set_xticks((tick_indices - x_mean) / x_std)
            ax2.set_xticklabels(tick_labels, rotation=45)
            st.pyplot(fig2)


        # ==================================================
        # 💡 構造変化点（τ）の確率密度分布（ヒストグラム）
        # ==================================================
        #st.write("### ⏱️ Posterior Distributions of Breakpoints (τ)")
        #st.caption("MCMCサンプリングによって得られた、構造変化点（τ）がその時期に存在する確率の高さ（分布）を示しています。")
        st.write("### ⏱️ Estimated Breakpoint Distributions — $p(\tau | y)$")
        st.caption("MCMCサンプリングによって得られた、構造変化点（$\tau$）の事後確率密度分布です。山が立っている時期ほど、そのタイミングでトレンドが変化した可能性が高いことを示しています。")

        fig_hist, (ax_h1, ax_h2) = plt.subplots(1, 2, figsize=(12, 3.5))

        # 生データの日付軸換算用関数
        def idx_to_date(idx_val):
            i = int(np.clip(round(idx_val), 0, len(trend) - 1))
            return trend.index[i]

        # 共通のX軸目盛り設定関数
        def setup_hist_xaxis(ax_target):
            tick_idxs = np.linspace(0, len(trend) - 1, 6, dtype=int)
            tick_lbls = [trend.index[i].strftime('%Y-%m') for i in tick_idxs]
            ax_target.set_xticks(tick_idxs)
            ax_target.set_xticklabels(tick_lbls, rotation=35, fontsize=9)
            ax_target.set_xlim(0, len(trend) - 1)
            ax_target.grid(True, linestyle="--", alpha=0.4)

        # --- 左側: 1-break Model の τ 分布 ---
        tau1_samples = s1["tau"] * x_std + x_mean
        ax_h1.hist(tau1_samples, bins=30, color="green", alpha=0.6, density=True, edgecolor="darkgreen")
        ax_h1.axvline(tau1_samples.mean(), color="darkgreen", linestyle="-", lw=2, label="Mean τ")
        ax_h1.set_title("1-break: τ1 Distribution", fontsize=10, fontweight="bold")
        setup_hist_xaxis(ax_h1)
        ax_h1.legend(fontsize=8)

        # --- 右側: 2-break Model の τ₁, τ₂ 分布 ---
        tau2_1_samples = s2["tau1"] * x_std + x_mean
        tau2_2_samples = s2["tau2"] * x_std + x_mean
        ax_h2.hist(tau2_1_samples, bins=30, color="red", alpha=0.5, density=True, edgecolor="darkred", label="τ₁ (1st Break)")
        ax_h2.hist(tau2_2_samples, bins=30, color="orange", alpha=0.5, density=True, edgecolor="darkorange", label="τ₂ (2nd Break)")
        ax_h2.axvline(tau2_1_samples.mean(), color="darkred", linestyle="-", lw=1.5)
        ax_h2.axvline(tau2_2_samples.mean(), color="darkorange", linestyle="-", lw=1.5)
        ax_h2.set_title("2-break: τ1 & τ2 Distributions", fontsize=10, fontweight="bold")
        setup_hist_xaxis(ax_h2)
        ax_h2.legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig_hist)
        st.session_state[f"fig_hist_{substance}"] = fig_hist

        # --- Breakpoint Windows (95% CI) を計算 ---
        tau1_samples = s1["tau"] * x_std + x_mean
        ci1 = np.percentile(tau1_samples, [2.5, 97.5])

        tau2_1_samples = s2["tau1"] * x_std + x_mean
        tau2_2_samples = s2["tau2"] * x_std + x_mean
        ci2_1 = np.percentile(tau2_1_samples, [2.5, 97.5])
        ci2_2 = np.percentile(tau2_2_samples, [2.5, 97.5])

        st.write(f"1-break Model τ 95% CI: {idx_to_date(ci1[0])} 〜 {idx_to_date(ci1[1])}")
        st.write(f"2-break Model τ₁ 95% CI: {idx_to_date(ci2_1[0])} 〜 {idx_to_date(ci2_1[1])}")
        st.write(f"2-break Model τ₂ 95% CI: {idx_to_date(ci2_2[0])} 〜 {idx_to_date(ci2_2[1])}")

        # ==================================================
        # ② 中段：モデル比較＆国際規制線イベントの合体プロット
        # ==================================================
        st.markdown("---")
        #st.write("### 📊 Bayesian Model μ(E[θ])（代表線）")
        #st.caption("Fitted Curve at Posterior Mean Parameters（μ(E[θ]) ）後方平均パラメータにおける適合曲線。1-breakと2-breakモデルの代表形状を示すPosterior Mean Parameter Fit 後方平均パラメータ適合。")
        st.write("### 📊 Posterior Mean Parameter Fit — μ(E[θ])")
        st.caption("事後分布の平均パラメータ（α, β, τ）を用いて計算した代表的なフィット曲線。不確実性は含まれず、1-break と 2-break の構造的な違いを比較するための線。")
        # フィット形状の計算 (MCMC結果の平均値から)
        # =================================
            # 1-break のフィット (標準化スケールで計算)
        # =================================
        alpha1 = s1["alpha"].mean()
        b11 = s1["beta1"].mean()
        b21 = s1["beta2"].mean()
        tau1_std_1 = s1["tau"].mean()
        fit1 = alpha1 + b11 * x_stdized + b21 * np.maximum(0, x_stdized - tau1_std_1)
        tau1_1_idx = tau1_std_1 * x_std + x_mean


        # =================================
        # 2-break のフィット (標準化スケールで計算) 【💡ここを修正！】
        # =================================
        alpha2 = s2["alpha"].mean()
        b12 = s2["beta1"].mean()
        b22 = s2["beta2"].mean()
        b32 = s2["beta3"].mean()
        tau1_std_2 = s2["tau1"].mean()
        tau2_std_2 = s2["tau2"].mean()
        # 💡 修正：大元の breakpoint2_model の定義 (beta2 - beta1), (beta3 - beta2) と完全に一致させる
        fit2 = (
            alpha2
            + b12 * x_stdized
            + (b22 - b12) * np.maximum(0, x_stdized - tau1_std_2)
            + (b32 - b22) * np.maximum(0, x_stdized - tau2_std_2)
        )

        tau1_2_idx = tau1_std_2 * x_std + x_mean
        tau2_2_idx = tau2_std_2 * x_std + x_mean

        # グラフ作成
        fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
        ax_comp.plot(x, y, "o", ms=4, alpha=0.4, color="gray", label="STL Trend Data")
        ax_comp.plot(x, fit1, "-", lw=2.5, color="green", label="1-break Mean Fit")
        ax_comp.plot(x, fit2, "-", lw=2.5, color="red", label="2-break Mean Fit")

        if pops_year:
            add_regulation_lines_date(ax_comp,trend)

        def get_date_str(idx_val):
            i = int(np.clip(round(idx_val), 0, len(trend) - 1))
            return trend.index[i].strftime('%Y-%m')

        # 構造変化点 (τ) の縦線プロット
        ax_comp.axvline(tau1_1_idx, color="green", linestyle=":", linewidth=1.5, label=f"τ1 (1-break) = {get_date_str(tau1_1_idx)}")
        ax_comp.axvline(tau1_2_idx, color="red", linestyle="--", linewidth=1.5, label=f"τ1 (2-break) = {get_date_str(tau1_2_idx)}")
        ax_comp.axvline(tau2_2_idx, color="red", linestyle="-.", linewidth=1.5, label=f"τ2 (2-break) = {get_date_str(tau2_2_idx)}")


        ax_comp.set_xticks(tick_indices)
        ax_comp.set_xticklabels(tick_labels, rotation=45)
        ax_comp.set_title(f"Bayesian Structural Break Comparison: {substance}", fontsize=12, fontweight="bold")
        ax_comp.set_xlabel("Month/Year")
        ax_comp.grid(True, linestyle="--", alpha=0.4)
        ax_comp.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        # ============================
        # 1-break の傾きラベル
        # ============================
        # 区間1（x < tau1_1_idx）
        x_mid1 = tau1_1_idx / 2
        y_mid1 = alpha1 + b11 * ((x_mid1 - x_mean) / x_std)
        ax_comp.text(x_mid1, y_mid1, f"β1={b11:.3f}", color="green")

        # 区間2（x >= tau1_1_idx）
        x_mid2 = (tau1_1_idx + x.max()) / 2
        y_mid2 = alpha1 + b11 * ((x_mid2 - x_mean) / x_std) + b21 * max(0, (x_mid2 - tau1_1_idx) / x_std)
        ax_comp.text(x_mid2, y_mid2, f"β2={b21:.3f}", color="green")

        # ============================
        # 2-break の傾きラベル
        # ============================

        # 区間1（x < tau1_2_idx）
        x2_mid1 = tau1_2_idx / 2
        y2_mid1 = alpha2 + b12 * ((x2_mid1 - x_mean) / x_std)
        ax_comp.text(x2_mid1, y2_mid1, f"β1={b12:.3f}", color="red")

        # 区間2（tau1 <= x < tau2）
        x2_mid2 = (tau1_2_idx + tau2_2_idx) / 2
        y2_mid2 = (
            alpha2
            + b12 * ((x2_mid2 - x_mean) / x_std)
            + (b22 - b12) * max(0, (x2_mid2 - tau1_2_idx) / x_std)
        )
        ax_comp.text(x2_mid2, y2_mid2, f"β2={b22:.3f}", color="red")

        # 区間3（x >= tau2）
        x2_mid3 = (tau2_2_idx + x.max()) / 2
        y2_mid3 = (
            alpha2
            + b12 * ((x2_mid3 - x_mean) / x_std)
            + (b22 - b12) * max(0, (x2_mid3 - tau1_2_idx) / x_std)
            + (b32 - b22) * max(0, (x2_mid3 - tau2_2_idx) / x_std)
        )
        ax_comp.text(x2_mid3, y2_mid3, f"β3={b32:.3f}", color="red")



        st.pyplot(fig_comp, bbox_inches='tight')
        st.session_state[f"fig_comp_{substance}"] = fig_comp

        
        st.write("### ⏱️ Estimated Parameters (Posterior Mean)")
        # 標準化インデックス → 元のインデックスへ復元
        x_mean_original_index = int(round(x_mean))

        # 範囲外チェック（安全）
        if 0 <= x_mean_original_index < len(trend):
            x_mean_date = trend.index[x_mean_original_index].strftime("%Y-%m")
        else:
            x_mean_date = "N/A"

        
        one_break_params = f"α: {alpha1:.3f}, β1: {b11:.3f}, β2: {b21:.3f}, τ: {get_date_str(tau1_1_idx)}"
        two_break_params = f"α: {alpha2:.3f}, β1: {b12:.3f}, β2: {b22:.3f}, β3: {b32:.3f}, τ1: {get_date_str(tau1_2_idx)}, τ2: {get_date_str(tau2_2_idx)}"


        #st.write(f"**1-break Model Parameters:**")
        #st.info(one_break_params)   
        #st.write(f"**2-break Model Parameters:**")
        #st.info(two_break_params)


        # 高解像度PNGダウンロードボタン
        import io
        buf_bayes = io.BytesIO()
        # 💡  外側に出した凡例(legend)を明示的に含めて保存する
        leg = ax_comp.get_legend()        
        fig_comp.savefig(buf_bayes, format="png", dpi=300, bbox_inches='tight')
        st.download_button(
            label="🖼️ ベイズ統合比較グラフを高解像度PNG(300dpi)としてダウンロード",
            data=buf_bayes.getvalue(),
            file_name=f"{substance}_bayesian_integration_plot.png",
            mime="image/png",
            key=f"dl_btn_{substance}"
        )
        # ==================================================
        # 🎨 Posterior Sample Fits（50 draws）
        # ==================================================

        st.write("### 🎨 Posterior Sample Fits（50 draws）")
        st.caption(
            "事後分布からランダムに抽出した 50 本のフィット線。"
            "モデルが取りうる多様なフィット形状を可視化し、"
            "特に構造変化点 τ のブレ（posterior の不安定さ）を観察する。"
        )

        col1, col2 = st.columns(2)

        # -------------------------
        # 1-break model
        # -------------------------
        with col1:
            fig_sf1, ax_sf = plt.subplots(figsize=(6, 5))

            # 元データ
            ax_sf.plot(x, y, "o", ms=4, alpha=0.4, color="gray", label="STL Trend Data")

            # 50 サンプルをランダムに選ぶ
            n_draws = 50
            idx = np.random.choice(len(s1["alpha"]), size=n_draws, replace=False)

            for i in idx:
                a = s1["alpha"][i]
                b1 = s1["beta1"][i]
                b2 = s1["beta2"][i]
                t1 = s1["tau"][i]

                h1 = np.maximum(0, x_stdized - t1)
                mu_i = a + b1 * x_stdized + b2 * h1

                ax_sf.plot(x, mu_i, color="blue", alpha=0.15)

            ax_sf.set_title("Posterior Sample Fits (1-break model)", fontsize=12)
            ax_sf.set_xticks(tick_indices)
            ax_sf.set_xticklabels(tick_labels, rotation=45)
            ax_sf.grid(True, linestyle="--", alpha=0.3)

            st.pyplot(fig_sf1)

        # -------------------------
        # 2-break model
        # -------------------------
        with col2:
            fig_sf2, ax_sf = plt.subplots(figsize=(6, 5))

            # 元データ
            ax_sf.plot(x, y, "o", ms=4, alpha=0.4, color="gray", label="STL Trend Data")

            # 50 サンプルをランダムに選ぶ
            n_draws = 50
            idx = np.random.choice(len(s2["alpha"]), size=n_draws, replace=False)

            for i in idx:
                a = s2["alpha"][i]
                b1 = s2["beta1"][i]
                b2 = s2["beta2"][i]
                b3 = s2["beta3"][i]
                t1 = s2["tau1"][i]
                t2 = s2["tau2"][i]

                h1 = np.maximum(0, x_stdized - t1)
                h2 = np.maximum(0, x_stdized - t2)
                mu_i = a + b1 * x_stdized + b2 * h1 + b3 * h2

                ax_sf.plot(x, mu_i, color="blue", alpha=0.15)

            ax_sf.set_title("Posterior Sample Fits (2-break model)", fontsize=12)
            ax_sf.set_xticks(tick_indices)
            ax_sf.set_xticklabels(tick_labels, rotation=45)
            ax_sf.grid(True, linestyle="--", alpha=0.3)

            st.pyplot(fig_sf2)

        """----------------------------------------------------------"""
        # ==================================================
        # ③ 下段：変化点の95%信用区間（時間軸の評価）
        # ==================================================
        st.markdown("---")
        st.write("### ⏱️ Estimated Breakpoint Windows (95% Credible Intervals)")
        st.caption("ベイズモデルが推定した、構造変化が起こった時期の95%不確実性の幅（日付換算）です。")

        def get_date_str_from_idx(idx_val):
            i = int(np.clip(round(idx_val), 0, len(trend) - 1))
            return trend.index[i].strftime('%Y-%m')

        col_inf1, col_inf2 = st.columns(2)
        with col_inf1:
            st.write("**【1-break Model】**")
            tau1_samples_raw = s1["tau"] * x_std + x_mean
            tau1_low = np.percentile(tau1_samples_raw, 2.5)
            tau1_high = np.percentile(tau1_samples_raw, 97.5)
            st.info(
                f"**推定変化点 (τ₁):** {get_date_str_from_idx(tau1_samples_raw.mean())}\n\n"
                f" **95% 信用区間:** `{get_date_str_from_idx(tau1_low)}` ～ `{get_date_str_from_idx(tau1_high)}`"
            )

        with col_inf2:
            st.write("**【2-break Model】**")
            tau2_1_samples_raw = s2["tau1"] * x_std + x_mean
            tau2_2_samples_raw = s2["tau2"] * x_std + x_mean
            tau2_1_low = np.percentile(tau2_1_samples_raw, 2.5)
            tau2_1_high = np.percentile(tau2_1_samples_raw, 97.5)
            tau2_2_low = np.percentile(tau2_2_samples_raw, 2.5)
            tau2_2_high = np.percentile(tau2_2_samples_raw, 97.5)
            st.error(
                f"**第1変化点 (τ₁):** {get_date_str_from_idx(tau2_1_samples_raw.mean())}\n"
                f"   **95% 信用区間:** `{get_date_str_from_idx(tau2_1_low)}` ～ `{get_date_str_from_idx(tau2_1_high)}`\n\n"
                f"**第2変化点 (τ₂):** {get_date_str_from_idx(tau2_2_samples_raw.mean())}\n"
                f"   **95% 信用区間:** `{get_date_str_from_idx(tau2_2_low)}` ～ `{get_date_str_from_idx(tau2_2_high)}`"
            )

        # --- 1-break ---
        tau1_mean_str = get_date_str_from_idx(tau1_samples_raw.mean())
        tau1_low_str  = get_date_str_from_idx(tau1_low)
        tau1_high_str = get_date_str_from_idx(tau1_high)

        # --- 2-break ---
        tau2_1_mean_str = get_date_str_from_idx(tau2_1_samples_raw.mean())
        tau2_1_low_str  = get_date_str_from_idx(tau2_1_low)
        tau2_1_high_str = get_date_str_from_idx(tau2_1_high)

        tau2_2_mean_str = get_date_str_from_idx(tau2_2_samples_raw.mean())
        tau2_2_low_str  = get_date_str_from_idx(tau2_2_low)
        tau2_2_high_str = get_date_str_from_idx(tau2_2_high)


        # ==================================================
        # ④ 下段：WAICによるモデル確率（モデル選択の定量化）
        # ==================================================
        st.markdown("---")

        st.write("### ⚖️ Model Weights based on WAIC")
        st.caption("MCMCの対数尤度からWAICを算出し、各モデルがデータを説明できている確率（尤もらしさの比率）を簡易評価します。")

        # 安全なインラインWAIC計算関数
        def calc_simple_waic(samples_dict, y_data, model_type):
            alpha = samples_dict["alpha"]
            b1 = samples_dict["beta1"]
            mus = []
            if model_type == "1break":
                b2 = samples_dict["beta2"]
                tau = samples_dict["tau"]
                for a, beta1_val, beta2_val, t in zip(alpha, b1, b2, tau):
                    mus.append(a + beta1_val * x_stdized + beta2_val * np.maximum(0, x_stdized - t))
            else:
                b2 = samples_dict["beta2"]
                b3 = samples_dict["beta3"]
                t1 = samples_dict["tau1"]
                t2 = samples_dict["tau2"]
                for a, beta1_val, beta2_val, beta3_val, tau1_val, tau2_val in zip(alpha, b1, b2, b3, t1, t2):
                    mus.append(a + beta1_val * x_stdized + beta2_val * np.maximum(0, x_stdized - tau1_val) + beta3_val * np.maximum(0, x_stdized - tau2_val))
            
            mus = np.array(mus)
            sigma = samples_dict["sigma"][:, None]
            log_lik = -0.5 * (((y_data - mus) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
            
            lppd = np.sum(np.log(np.mean(np.exp(log_lik), axis=0)))
            p_waic = np.sum(np.var(log_lik, axis=0))
            return -2 * lppd + 2 * p_waic

        # 引数を修正し、安全にWAICを計算
        waic1 = calc_simple_waic(s1, y, "1break")
        waic2 = calc_simple_waic(s2, y, "2break")

        # ウェイトの計算
        min_waic = min(waic1, waic2)
        d1 = -0.5 * (waic1 - min_waic)
        d2 = -0.5 * (waic2 - min_waic)
        w1 = np.exp(d1) / (np.exp(d1) + np.exp(d2))
        w2 = np.exp(d2) / (np.exp(d1) + np.exp(d2))

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.metric(label="🟢 1-break Model Probability", value=f"{w1 * 100:.1f} %", delta=f"WAIC: {waic1:.2f}", delta_color="inverse")
        with col_w2:
            st.metric(label="🔴 2-break Model Probability", value=f"{w2 * 100:.1f} %", delta=f"WAIC: {waic2:.2f}", delta_color="inverse")
        waic_verdict = "" # 判定結果を格納する変数
        # WAIC単体での判定（ベースラインの出力）
        if abs(waic1 - waic2) < 2:
            waic_verdict = "1-break_preferred"
            waic_txt = "* **WAIC判定:** 両モデルの差は僅かです。よりシンプルな「1-breakモデル」の採用をお勧めします。\n"
            waic_html = f"<b>WAIC判定:</b> 両モデルの差は僅かです。よりシンプルな「1-breakモデル」の採用をお勧めします<br>"
            st.warning(f"⚖️ {waic_txt}")
        elif waic2 < waic1:
            if (waic1 - waic2) > 5:
                waic_verdict = "2-break_strongly_preferred" 
            else:
                waic_verdict = "2-break_preferred"
            waic_txt = f"*  **WAIC判定:** 構造変化点が2箇所ある **「2-breakモデル」** が、{w2*100:.1f}% の確率で優位と判定されました。"
            waic_html = f"<b>WAIC判定:</b> 構造変化点が2箇所ある <b>「2-breakモデル」</b> が、{w2*100:.1f}% の確率で優位と判定されました<br>"
            st.success(f"🏆 {waic_txt} ※最終採用は下部の総合評価を参照）。")   #st.success
        else:
            waic_verdict = "1-break_preferred"
            waic_txt = f"* **WAIC判定:** 構造変化点が1箇所のみの **「1-breakモデル」** が、{w1*100:.1f}% の確率でより適切と判定されました。"
            waic_html = f"<b>WAIC判定:</b> 構造変化点が1箇所のみの <b>「1-breakモデル」</b> が、{w1*100:.1f}% の確率でより適切と判定されました<br>"
            st.success(f"🏆  {waic_txt} ")   #st.success   

        # --- HTML テーブル 基本的に別途モジュールで作成するが、簡単な部分を本体で作成しておく---
    
        breakpoint_table_html = f"""

        <h2>⚖️ WAIC-based Model Comparison</h2>
        <p>WAIC(Widely Applicable Information Criterion)はモデルの説明力を定量的に評価する指標ですが、モデル選択の唯一の基準ではありません。特に、構造変化点の推定の安定性や、実務的な解釈のしやすさも重要な要素となります。</p>
        <p> <b>{waic_verdict}</b><br> {waic_html}</p>

        <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse;">
            <thead>
                <tr>
                <th>Model</th>
                <th>Weight w</th>
                <th>WAIC</th>
                <th>Params (standardized: {x_mean_date})</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                <td>1-break model</td>
                <td>{w1*100:.1f} %</td>
                <td>{waic1:.2f}</td>
                <td>{one_break_params}</td>
                </tr>
                <tr>
                <td>2-break model</td>
                <td>{w2*100:.1f} %</td>
                <td>{waic2:.2f}</td>
                <td>{two_break_params}</td>
                </tr>
            </tbody>
            </table>


        <h2>⏱️ Estimated Breakpoint Windows (95% Credible Intervals)</h2>
        <p>Bayesian model–estimated timing of structural changes (converted to calendar dates).</p>

        <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse;">
            <tr style="background-color:#f0f0f0;">
                <th>Model</th>
                <th>Breakpoint</th>
                <th>Mean</th>
                <th>95% CI (Low)</th>
                <th>95% CI (High)</th>
            </tr>

            <tr>
                <td rowspan="1">1-break</td>
                <td>τ₁</td>
                <td>{tau1_mean_str}</td>
                <td>{tau1_low_str}</td>
                <td>{tau1_high_str}</td>
            </tr>

            <tr>
                <td rowspan="2">2-break</td>
                <td>τ₁</td>
                <td>{tau2_1_mean_str}</td>
                <td>{tau2_1_low_str}</td>
                <td>{tau2_1_high_str}</td>
            </tr>
            <tr>
                <td>τ₂</td>
                <td>{tau2_2_mean_str}</td>
                <td>{tau2_2_low_str}</td>
                <td>{tau2_2_high_str}</td>
            </tr>
        </table>
        """
        #-------------------------------------------------------------------------------------
        st.markdown("---")
        st.write("### ⏱️ Estimated Parameters (Posterior Mean)")
        st.info(
        f"*X軸の基準点: 時系列データのインデックス({x_mean:.3f}) → 元の時系列では {x_mean_date}** をX=0の基準点として、モデルのパラメータを解釈してください。*\n\n"
        "以下のパラメータは、MCMCサンプリングから得られた**事後分布の平均値**です "
        "これらの数値は、データとモデルに基づいて最も可能性の高い値を表していますが、特に分岐点（τ）については不確実性があることを忘れないでください。結果を解釈する際は、単なる平均値だけでなく、事後分布の全体形状や信用区間の幅（不確実性）も必ず視野に入れてください。"
    )
        st.components.v1.html(breakpoint_table_html, height=600, scrolling=True )
        #st.markdown("<h2>📉 練習　breakpoint_table_html</h2>", unsafe_allow_html=True)
        #st.markdown(breakpoint_table_html, unsafe_allow_html=True)


        st.session_state[f"bp_table_{substance}"] = breakpoint_table_html


        #----------------------------------------------------------------------------------------------------
        # ==================================================
        # ⑥ 最終結論（ロジックの統合アウトプット）
        # ==================================================
        
        st.markdown("### 📌 データ構造とモデルの堅牢性、適合度に関する注意点")
        conclusion_container = st.empty()
        # ① RMSE_param：データ vs 代表線 μ(E[θ]) のズレ（小さいほど適合）RMSE(y, μ(E[θ]))　代表線の適合度（ズレることが多い）
        rmse_param_1 = np.sqrt(np.mean((y - fit1)**2))
        rmse_param_2 = np.sqrt(np.mean((y - fit2)**2))

        # ② Nonlinearity Gap：事後予測平均 E[μ] vs 代表線 μ(E[θ]) のズレ　RMSE(E[μ], μ(E[θ]))　posterior の非線形性・複雑さ

        gap1 = np.sqrt(np.mean((mu_mean_1 - fit1)**2))
        gap2 = np.sqrt(np.mean((mu_mean_2 - fit2)**2))

        # 観測値 vs 事後予測平均 の適合度の比較　RMSE(y, E[μ])　モデルの平均予測力（最も信頼できる）
        rmse_pred_1 = np.sqrt(np.mean((y - mu_mean_1)**2))
        rmse_pred_2 = np.sqrt(np.mean((y - mu_mean_2)**2))

        if rmse_pred_1>rmse_pred_2:
            pred_penalty1 = " (2-break is better)"
            pred_penalty2 = ""
        else:
            pred_penalty1 = ""
            pred_penalty2 = " (1-break is better)"              

        # ③ Separation Score：2-break における τ₂ − τ₁ の分離度・安定度
        # サンプリング期間全体（x_stdized のレンジ）に対して、十分な間隔（例: 5% 以上）が空いているか
        delta_threshold = 0.05
        sep_prob = np.mean((s2["tau2"] - s2["tau1"]) > delta_threshold)

        def calc_relative_score(val1, val2, eps=1e-9):
                    vals = np.array([val1, val2], dtype=float)
                    vmin, vmax = vals.min(), vals.max()
                    if abs(vmax - vmin) < eps:
                        return 0.0, 0.0
                    s1 = (val1 - vmin) / (vmax - vmin)
                    s2 = (val2 - vmin) / (vmax - vmin)
                    return s1, s2
        # TI結果の取得とペナルティ加算
        ti_result = st.session_state.get(f"ti_{substance}")
        ti_text = ""    #TIの結果を表示するための変数を初期化
        
        robustness_score_1= 0   #ペナルティスコアは後で加算するため、ここでは初期値を0に設定
        robustness_score_2= 0



        # 各メトリクスにおけるペナルティスコア (0.0 = 最優秀, 1.0 = 最悪)
        fit_s1, fit_s2 = calc_relative_score(rmse_param_1, rmse_param_2)
        gap_s1, gap_s2 = calc_relative_score(gap1, gap2)
        
        # ③ CI 幅（変化点の不確実性）の計算
        width1 = (tau1_high - tau1_low).days if hasattr(tau1_high, 'days') else (tau1_high - tau1_low)
        width2_1 = (tau2_1_high - tau2_1_low).days if hasattr(tau2_1_high, 'days') else (tau2_1_high - tau2_1_low)
        width2_2 = (tau2_2_high - tau2_2_low).days if hasattr(tau2_2_high, 'days') else (tau2_2_high - tau2_2_low)
        width2 = 0.5 * (width2_1 + width2_2)
        width_s1, width_s2 = calc_relative_score(width1, width2)

        # ④ 変化点不整合ペナルティ
        if sep_prob > 0.98:
            tau_penalty_2 = 0 
        elif sep_prob >0.90:
            tau_penalty_2 = 0.5        
        else:
            tau_penalty_2 = 1

        tau_penalty_1 = 0.0



        #4 TI
        if ti_result is not None :
            max_tv_ratio = ti_result['max_tv_ratio']
            conditions = ti_result['conditions']
            tv_ratio_1 = ti_result['tv_ratio_1']
            tv_ratio_2 = ti_result['tv_ratio_2']
        else:
            max_tv_ratio = 0    #暫定

        if ti_result is not None and max_tv_ratio <1.05:
            ti_favors = "2-break" if ti_result['logBF'] > 0 else "1-break"
            # ペナルティ設計：支持されなかった側にペナルティ 1.0 を加算
            if ti_result['logBF'] > 0: 
                ti_penalty_1 = 1.0
                ti_penalty_2 = 0.0
            if ti_result['logBF'] > 1.1: 
                ti_penalty_1 = 2.0
                ti_penalty_2 = 0.0
            if ti_result['logBF'] > 3: 
                ti_penalty_1 = 3.0
                ti_penalty_2 = 0.0
            if ti_result['logBF'] > 5: 
                ti_penalty_1 = 4.0
                ti_penalty_2 = 0.0
            else: 
                ti_penalty_1 = 0.0
                ti_penalty_2 = 1.0

            robustness_score_1 = fit_s1 + gap_s1 + width_s1 + tau_penalty_1
            robustness_score_2 = fit_s2 + gap_s2 + width_s2 + tau_penalty_2
            total_score_1 = robustness_score_1 + ti_penalty_1 
            total_score_2 = robustness_score_2 + ti_penalty_2  
            
            ti_text = (
                f"\n\n**【熱力学的積分（TI）による結果】**\n"  #
                f" **MCMC条件**: {conditions}\n\n 収束状況: tv_ratio1={tv_ratio_1:.2f},  tv_ratio2={tv_ratio_2:.2f}, logZ1={ti_result['logZ1']:.2f}, logZ2={ti_result['logZ2']:.2f}\n\n" #
                f"別タブで実行されたTI（厳密な周辺尤度計算）では、対数ベイズファクター $\\log BF = {ti_result['logBF']:.2f}$ となり、 **「{ti_favors} モデル」** が支持されています。\n\n"  
                f"* 補足: Jeffreys' Scaleをを形式的にスコア(0-4)として、堅牢性ペナルティに加算した総合ペナルティは 1-break ({total_score_1:.2f})、2-break({total_score_2:.2f}) となります。特に、Posterior Mean Parameter Fit — μ(E[θ])のグラフや他の堅牢性ペナルティの性質に留意して最終判断すること。"
            )
        elif  ti_result is not None:
            tv_status_1 = "Failed"
            tv_status_2 = "Failed"

            if tv_ratio_1<1.05:
                tv_status_1 = "OK"
            elif tv_ratio_1<1.10:
                tv_status_1 = "Marginal"

            if tv_ratio_2<1.05:
                tv_status_2 = "OK"
            elif tv_ratio_2<1.10:
                tv_status_2 = "Marginal"

            ti_text = (
                f"\n\n*(注: 「5. TI」タブで熱力学的積分を実行しましたが、{conditions}で\ntv_ratio1={tv_ratio_1:.2f} {tv_status_1}および tv_ratio2={tv_ratio_2:.2f} {tv_status_2}であり、logZ1={ti_result['logZ1']:.2f}, logZ2={ti_result['logZ2']:.2f}は、モデル比較に活用できない可能性があります。"
                "成功した場合は、厳密な周辺尤度に基づく検証結果がここに追加されます。⚠️"
                "モデル自体がデータに適合していない場合はTIを実行してもモデルの比較には活用できませんので、計算コストの高い再計算は慎重に判断してください。)*\n\n"
               )
            ti_text_e = (
                f"\n\nNote:\n\nTI **failed at {conditions}** on the **”5. TI“ tab**, tv1:{tv_ratio_1:.2f} and tv2:{tv_ratio_2:.2f}.\n\n "
                "If successful, the verification results based on the exact marginal likelihood will be added here.\n\n"
                "⚠️If the model itself does not fit the data, running TI will not be useful for model comparison, "
                "so please exercise caution.\n\n"
                )
            robustness_score_1 = fit_s1 + gap_s1 + width_s1 + tau_penalty_1
            robustness_score_2 = fit_s2 + gap_s2 + width_s2 + tau_penalty_2
            total_score_1 = robustness_score_1  #暫定
            total_score_2 = robustness_score_2  #暫定　後続で使用しているため           


        else:
            ti_text = (
                f" **TI Note**:科学的に変化傾向を検知することを重視する場合はTI基準を優先するが、モデルの適合状況を**堅牢性ペナルティ**で判断しモデルの頑健性を優先する場合、ペナルティが低いモデルの採用が推奨される。"
                "\n\n **「5. TI」タブ**で熱力学的積分を実行すると、正確な周辺尤度に基づく検証結果がここに追加されます。"
                "\n\n 先に進む前に、**代表曲線の適合度**および**非線形ギャップ**に対するペナルティを確認してください。"
                "\n\n ⚠️モデル自体がデータに適合していない場合、TIの結果をモデル比較に適用すべきではありません。\n\n"
                )
              
            robustness_score_1 = fit_s1 + gap_s1 + width_s1 + tau_penalty_1
            robustness_score_2 = fit_s2 + gap_s2 + width_s2 + tau_penalty_2
            total_score_1 = robustness_score_1  #暫定
            total_score_2 = robustness_score_2  #暫定　後続で使用しているため







        # 表示用データの整理（生値とペナルティスコアを明記して誤解を防ぐ）
        if ti_result == None:
            report_data = {
                "評価指標 (Metrics)": [
                    "① 代表線適合度 (RMSE) ↓",
                    "② 非線形ギャップ (Gap) ↓",
                    "③ 変化点の不確実性 (CI幅) ↓",
                    f"④  τ₂ − τ₁ が　全期間ｘ{delta_threshold}である確率（分離度)↓",

                    "⑤  予測性能（RMSE_pred） ↓↑",
                    "📊 堅牢性ペナルティスコア (Model Robustness) ↓"
                ],
                "1-break Model": [
                    f"{rmse_param_1:.4f} (pt: {fit_s1:.2f})",
                    f"{gap1:.4f} (pt: {gap_s1:.2f})",
                    f"{width1:.1f}日 (pt: {width_s1:.2f})",
                    f"非該当 (pt: {tau_penalty_1:.2f})",   
                    f"{rmse_pred_1:.4f} {pred_penalty1}" ,          
                    f"{robustness_score_1:.2f}"
                ],
                "2-break Model": [
                    f"{rmse_param_2:.4f} (pt: {fit_s2:.2f})",
                    f"{gap2:.4f} (pt: {gap_s2:.2f})",
                    f"{width2:.1f}日 (pt: {width_s2:.2f})",
                    f"{sep_prob:.4f} (pt: {tau_penalty_2:.2f})",
                    f"{rmse_pred_2:.4f}  {pred_penalty2}" ,                         
                    f"{robustness_score_2:.2f}"
                ]
            }
        else:
            if max_tv_ratio <1.05:
                total_score_1 = robustness_score_1 + ti_penalty_1   # 付けたしのためロジックの整理が不足しているかも
                total_score_2 = robustness_score_2 + ti_penalty_2
            else:
                total_score_1 = robustness_score_1
                total_score_2 = robustness_score_2
                ti_penalty_1 = 0 #"計算不適"
                ti_penalty_2 = 0 #"計算不適"

            report_data = {
                "評価指標 (Metrics)": [
                    "① 代表線適合度 (RMSE) ↓",
                    "② 非線形ギャップ (Gap) ↓",
                    "③ 変化点の不確実性 (CI幅) ↓",
                    f"④  τ₂ − τ₁ が　全期間ｘ{delta_threshold}である確率（分離度)↓",

                    "⑤  予測性能（RMSE_pred） ↓↑",
                    "📊 **(堅牢性ペナルティ)** ↓",
                    "形式的TIペナルティスコア(Jeffreys' Scaleに対応 0 - 4) ↓",
                    "📊 総合ペナルティスコア  **(堅牢性ペナルティ)** ↓"
                ],
                "1-break Model": [
                    f"{rmse_param_1:.4f} (pt: {fit_s1:.2f})",
                    f"{gap1:.4f} (pt: {gap_s1:.2f})",
                    f"{width1:.1f}日 (pt: {width_s1:.2f})",
                    f"非該当 (pt: {tau_penalty_1:.2f})",   
                    f"{rmse_pred_1:.4f} {pred_penalty1}" , 
                    f" **({robustness_score_1})**",
                    f"{ti_penalty_1:.2f}" if max_tv_ratio <1.05 else "利用不適",           
                    f"{total_score_1:.2f}　**({robustness_score_1})**"
                ],
                "2-break Model": [
                    f"{rmse_param_2:.4f} (pt: {fit_s2:.2f})",
                    f"{gap2:.4f} (pt: {gap_s2:.2f})",
                    f"{width2:.1f}日 (pt: {width_s2:.2f})",
                    f"{sep_prob:.4f} (pt: {tau_penalty_2:.2f})",
                    f"{rmse_pred_2:.4f} {pred_penalty2}" ,    
                    f" **({robustness_score_2})**",                 
                    f"{ti_penalty_2:.2f}"if max_tv_ratio <1.05 else "利用不適",  
                    f"{total_score_2:.2f}　**({robustness_score_2})**"
                ]
            }   


        df_report = pd.DataFrame(report_data)   

        import re

        def extract_pt(value):
            # 文字列から "pt: 数値" の形式でポイントを抽出する関数
            match = re.search(r"pt:\s*([0-9.]+)", value)
            if match:
                return float(match.group(1))
            return 0.0
        def highlight_nonzero_pt(val):
            # セルの値からポイントを抽出し、ポイントが0でない場合にスタイルを適用する関数
            pt = extract_pt(val)
            if pt != 0:
                return "background-color: #fff3b0; font-weight: bold;"
            return ""
        
        # stylerを使ってポイントが0でないセルをハイライトするスタイルを適用
        

        styled = (
            df_report.style
            .applymap(highlight_nonzero_pt)
            .set_table_styles([
                {"selector": "table", "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid black"),
                ]},
                {"selector": "th", "props": [
                    ("border", "1px solid black"),
                    ("padding", "6px"),
                ]},
                {"selector": "td", "props": [
                    ("border", "1px solid black"),
                    ("padding", "6px"),
                ]},
            ])
            .set_properties(**{"border": "none"})  # ← ここで Styler デフォルトの border を殺す
        )

        #styled = df_report.style.applymap(highlight_nonzero_pt)
        #テーブル本体のHTMLを生成
        html_table = styled.to_html()

        # CSSスタイルを抽出して、テーブルの前に挿入するために分割
        html_style = styled.to_html().split("<style")[1]
        html_style = "<style" + html_style.split("</style>")[0] + "</style>"
        # Streamlit 表示

        
        html_table = "<h2>📊 多角的なモデル堅牢性の評価（ポイント付き）</h2>" + html_table
        #st.markdown("### 📊 多角的なモデル堅牢性の評価（ポイント付き）")
        st.markdown(html_table, unsafe_allow_html=True) #罫線は1重で表示される。ポイントのハイライトも効いている
        #st.components.v1.html(html_table, height=600, scrolling=True ) #罫線は２重で表示される。ポイントのハイライトも効いている


        #breakpoint_table_html = html_style+breakpoint_table_html
        #st.components.v1.html(breakpoint_table_html, height=600, scrolling=True )   #罫線は1重で表示される。
        # 見出しを少し強調してテーブル表示
        st.markdown("---")
        #breakpoint_table_html2 = html_style+breakpoint_table_html+html_table    
        #st.components.v1.html(breakpoint_table_html2, height=600, scrolling=True ) #breakpoint_table_html2の罫線は1重、html_tableの罫線は2重。
        #st.write("### 🛡️ 多角的なモデル堅牢性の評価")
        #st.table(report_data)
        # 2. report_generator モジュールへ送るためのHTMLの組み立て（CSSスタイル + WAICテーブル + 堅牢性テーブル）
        breakpoint_table_html = (
            html_style +                 # ← CSS を最初に置く
            breakpoint_table_html +      # ← WAIC テーブル（手書き HTML）
            html_table                   # ← 堅牢性テーブル（Styler）
        )
        # 3. Streamlit で表示
        #st.components.v1.html(final_html, height=800, scrolling=True) #breakpoint_table_html2の罫線は1重、html_tableの罫線は2重。で変化なし

        st.session_state[f"report_data_{substance}"] = report_data

        # --------------------------------------------------
        # 判定ロジックの修正（ペナルティが小さい方を推奨する）
        # --------------------------------------------------
        base_txt = "\n\n科学的に変化傾向を検知することを重視する場合はTI基準を優先するが、実務的な解釈の明瞭性やモデルの頑健性を優先する場合は、**堅牢性ペナルティの低いモデルの採用がありうる** \n\n"

        rmse_warning=f"* **代表線の適合不全**: 代表線フィット曲線とデータの乖離（RMSE: {rmse_param_2:.4f}）が大きく、過学習の可能性があります。"  if rmse_param_2>1 else f"* **代表線の適合不全**: 代表線フィット曲線とデータの乖離（RMSE: {rmse_param_2:.4f}）が 1-break modelより大きい。"
        gap_warning= f"* **非線形ギャップ**: 事後予測平均 E[μ] と代表線 μ(E[θ]) の乖離(Gap: {gap2:.4f}）が大きく、posterior が単純な代表値で表現できないほど複雑化している兆候があります。"  if gap2>1 else f"* **大きな非線形ギャップ**: 事後予測平均 E[μ] と代表線 μ(E[θ]) の乖離(Gap: {gap2:.4f}）が 1-break modelより大きい。"
        width_warning= f"* **変化点の信頼性**: 信用区間の幅 {width2:.1f}月 (ペナルティ:{width_s2:.2f})に課題があります。"
        sep_warning=f"* **変化点の分離度**: 分離安定性 {sep_prob:.4f}（ペナルティ: {tau_penalty_2:.2f}）に課題があります。"
        overfitting_warning = (
            "* **予測性能**: 代表線フィット曲線とデータの乖離が大きいが、予測性能がよい。**過学習の可能性** があります。"
            if (rmse_param_2 > rmse_param_1 + 1)       # 代表線の適合が悪化
            and (rmse_pred_2 <= rmse_pred_1 + 0.5)     # 予測性能は悪化していない
            and (gap2 > gap1 + 1)                      # posterior が複雑化
            else ""
        )

        
        rmse_txt = rmse_warning if fit_s2 >  fit_s1 else ""
        gap_txt = gap_warning if gap_s2 > gap_s1 else ""
        width_txt = width_warning if width2 > width1 else ""
        sep_txt = sep_warning if tau_penalty_2 > tau_penalty_1  else ""

        recommendation_text = base_txt  + rmse_txt + gap_txt +  width_txt + sep_txt+"\n"
        recommendation_html = base_txt +"<br>" + rmse_txt+"<br>" + gap_txt+"<br>" +  width_txt+"<br>" + sep_txt+"<br>"

        tau_width_penalty2 = tau_penalty_2 + width_s2

        if robustness_score_1  < robustness_score_2 and gap_s2 > 1:
            text=(    "💡 **総合評価に基づく推奨: 「1-break モデル」の採用検討**\n\n"
                "**【判断のロジック】**\n\n"

                #f"{base_txt}\n\n"    #                
                #f"形式的な情報量基準（WAIC）では {waic_verdict} 2-break モデルが圧倒的（確率 {w2*100:.1f}%）に支持されています。 "
                #f"{waic_txt} \n\n"    #

                f"WAICでは考慮できていない以下のモデルの多角的な堅牢性評価におけるペナルティスコアには、2-break側 でより多くのリスク（ペナルティ）{robustness_score_2}が 確認されました。\n\n"   #
                f"{rmse_txt}  \n"  #
                f"{gap_txt}  \n"   #
                f"{width_txt}  \n"  #
                f"{sep_txt}  \n"   #
                f"{overfitting_warning} \n"    #
                #f"{ti_text}  \n"                
                )
            #html_text = text.replace("test", "<br>")
            conclusion_container.info(f"{text}")
            st.info(f"{ti_text}")
        # パターン2: total_scoreで、2-break の方が優秀（または同等以上）な場合
        elif gap_s2 < 1  and robustness_score_2 < robustness_score_1:

            text = (
                "🏆 **総合評価に基づく推奨: 「2-break モデル」の採用**\n\n"    #
                "**【判断のロジック】**\n"    #

                #f"{base_txt}\n\n"    #
                #f"統計的情報量基準（WAIC）:{waic_txt} \n\n"    #

                f"2-break モデル（スコア: {fit_s2 + gap_s2 + width_s2 + tau_penalty_2:.2f}）が 1-break モデル（スコア: {fit_s1 + gap_s1 + width_s1 + tau_penalty_1:.2f}）を上回る（ペナルティが小さい）結果となっています。\n"
                f"代表線の適合性、非線形ギャップの小ささ、変化点の信用区間幅や、分離安定性に問題がないならば、構造変化が2段階で起きていると解釈することは妥当です。 \n"    #


                f"{rmse_txt}  \n"  #
                f"{gap_txt}  \n"   #
                f"{width_txt}  \n"  #
                f"{sep_txt}  \n"   #
                f"{overfitting_warning} \n"    #
                #f"{ti_text}  \n"                
                )

            conclusion_container.success(f"{text}")
            st.info(f"{ti_text}")

        # パターン3: スコアが近似、あるいは 1-break が完全に優位な場合
        else:
            text = ( 
                #f"{base_txt}\n\n"   #
                #f" 統計的情報量基準（WAIC）:{waic_txt} \n\n"   #

                "⚠️ **総合評価に基づく推奨: 要慎重判断（シンプルなモデルを推奨）**\n\n"
                "**【判断のロジック】**\n"
                f"総合ペナルティスコアは 1-break ({total_score_1:.2f}) に対し 2-break ({total_score_2:.2f}) である。\n"
                "非線形ギャップが大きく2-breakには過学習の兆候がある。過剰なパラメータ化を避けるのであれば、統計モデルの原則（オッカムの剃刀）に従い、よりシンプルで頑健な **1-break モデル** をベースに更なる考察を進めることを推奨します。\n"    #

                f"{rmse_txt}  \n"  #
                f"{gap_txt}  \n"   #
                f"{width_txt}  \n"  #
                f"{sep_txt}  \n"   #
                f"{overfitting_warning} \n"    #
                #f"{ti_text}"
                )        

            conclusion_container.warning(f"{text}" )            
            st.info(f"{ti_text}")

        st.session_state[f"ti_text_{substance}"] = ti_text






        # ==================================================
        # 💾 追加：HTMLレポート保存ボタン
        # ==================================================
        st.markdown("---")
        st.write("### 📂 レポートの出力")
        st.caption("現在の解析結果（判定見解と堅牢性評価テーブル）をHTMLファイルとしてサーバーの `output/` ディレクトリに保存します。")


        if st.button("📄 HTMLとして保存", key=f"btn_save_html2_{substance}"):

            # --- img_classical = "<p>Classical analysis was not performed for this substance.</p>"
            classical_fig = st.session_state.get(f"classical_fig_{substance}")
            classical_table = st.session_state.get(f"classical_table_{substance}")

            if classical_fig is None or classical_table is None:
                if ts_log is None:
                    ts_log = create_timeseries(df, substance) # ここで再度作成しているのは、セッションに保存されていない場合に備えてです。create_timeseries関数内でフィルタリングなどの処理が必要な場合は、ここで対応してください。
                classical_fig, classical_table, pwlf_model = run_classical_break_analysis(
                    trend,
                    ts_log,
                    yaxis_mode,
                    n_segments,
                    piecewise_fit,
                    calc_moving_slope,
                    pops_year,
                    add_regulation_lines,
                    substance
                )

                st.session_state[f"classical_fig_{substance}"] = classical_fig
                st.session_state[f"classical_table_{substance}"] = classical_table
            file_path = generate_html_report_full(
                            substance,
                            fig_raw,
                            fig_stl,
                            fig1,
                            fig2,
                            fig_hist,
                            fig_comp,
                            classical_fig if "classical_fig" in locals() else None,
                            classical_table if "classical_table" in locals() else None,
                            ti_result,
                            ti_text,
                            robustness_score_1,
                            robustness_score_2,
                            gap_s2,
                            w2,
                            rmse_param_2,
                            gap2,
                            width_s2,
                            tau_penalty_2,
                            total_score_1,
                            total_score_2,
                            report_data,
                            breakpoint_table_html,
                        )

            st.success(f"✅ レポートが {file_path} に保存されました。")

        st.session_state[f"fig_stl_{substance}"] = fig_stl
        st.session_state[f"fig1_{substance}"] = fig1
        st.session_state[f"fig2_{substance}"] = fig2

        st.session_state[f"robustness_score_1_{substance}"] = robustness_score_1
        st.session_state[f"robustness_score_2_{substance}"] = robustness_score_2
        st.session_state[f"gap_s2_{substance}"] = gap_s2
        st.session_state[f"gap2_{substance}"] = gap2
        st.session_state[f"width_s2_{substance}"] = width_s2 
        st.session_state[f"w2_{substance}"] = w2  #waicの2-break支持率
        st.session_state[f"rmse_param_2_{substance}"] = rmse_param_2
        st.session_state[f"tau_penalty_2_{substance}"] = tau_penalty_2
        st.session_state[f"total_score_1_{substance}"] = total_score_1
        st.session_state[f"total_score_2_{substance}"] = total_score_2

# ------------------------------------
# Tab 5: TI (Thermodynamic Integration)
# ------------------------------------
with tab5:
    import streamlit as st
    import numpy as np
    def calculate_tv_ratio(y):
        # 実際の総移動距離（隣り合う点との差分の絶対値の合計）
        total_variation = np.sum(np.abs(np.diff(y)))
        # 始点と終点の純粋な差分（理想的な最小移動距離）
        net_change = np.abs(y[-1] - y[0])
        
        # ゼロ除算を防ぐ安全策を入れつつ比率を計算
        return total_variation / net_change if net_change != 0 else np.nan

    st.sidebar.header("🛠️ TI / MCMC condition")

    n_beta = st.sidebar.slider("βステップ数", min_value=3, max_value=150, value=20, step=1)
    beta_schedule = st.sidebar.selectbox("βの配置", ["Power-law (推奨)", "Linear (equidistant)"])
    #beta_schedule = st.sidebar.selectbox("βの配置", ["Power-law (推奨)", "Linear (equidistant)", "Quadratic", "Sigmoid"])
    num_warmup = st.sidebar.number_input("Warmup (Tune)", min_value=500, max_value=5000, value=2000, step=500)  #1000 
    num_samples = st.sidebar.number_input("Samples (Draws)",  min_value=500, max_value=10000, value=4000, step=500) #3000 でもおおむね良いが、HCBDは収束しない。

    trend = st.session_state.get(f"trend_{substance}")    
    #trend = st.session_state.get("trend")
    if trend is None:
        st.warning("タブ2でSTLを実行してください")
        st.stop()

    x = np.arange(len(trend))
    y = trend.values

    # 💡 物質固有のTI結果の部屋（キー）を定義
    key_ti = f"ti_{substance}"

    if st.button("Run TI", key=f"btn_ti_{substance}"): # 👈 keyを追加して安全に
        # 計算を劇的に安定化させるためにXを標準化
        x_mean, x_std = x.mean(), x.std()
        x_stdized = (x - x_mean) / x_std
    # 画面にプレースホルダー（空の枠）を作成
        progress_text = st.empty()
        progress_container = st.empty()
    # --- 実行ボタンが押された時の処理 ---
    #if st.button("Run TI"):

        conditions = {
            "n_beta": n_beta,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "beta_schedule": beta_schedule,
        }

        
        with st.spinner("TI計算を実行中..."):
            
            # 1-break の計算（ユーザーの選択条件を動的に渡す）elogl1->E1
            progress_text.markdown("### 🟢 1-break モデルの計算中")
            logZ1, beta1, E1 = run_ti_numpyro_breakpoint(
                x, y, model="1break", 
                n_beta=n_beta, 
                num_warmup=num_warmup, 
                num_samples=num_samples,
                beta_schedule=beta_schedule,
                progress_container=progress_container # ← 空の枠をそのまま渡す
            )
            # 1. プロットに使われている「平均期待対数尤度」の配列をれ作成
            y1 = np.array([np.mean(e) for e in E1])
            tv_ratio_1 = calculate_tv_ratio(y1) # パス1の変動比率を計算
            st.info(f"1-break モデル：TI計算の変動比率は、{tv_ratio_1}")

            # 2-break の計算
            progress_text.markdown("### 🔴 2-break モデルの計算中")
            logZ2, beta2, E2 = run_ti_numpyro_breakpoint(
                x, y, model="2break", 
                n_beta=n_beta, 
                num_warmup=num_warmup, 
                num_samples=num_samples,
                beta_schedule=beta_schedule,
                progress_container=progress_container # ← 同じ枠を使い回す
            )
        
        # あとは logZ1, logZ2 の結果を画面に表示するだけ！
        """
        with st.spinner("Running 1-break TI..."):
            logZ1, beta1, E1 = run_ti_numpyro_breakpoint(x_stdized, y, model="1break")
        with st.spinner("Running 2-break TI..."):
            logZ2, beta2, E2 = run_ti_numpyro_breakpoint(x_stdized, y, model="2break")
        """
        # 1. プロットに使われている「平均期待対数尤度」の配列をそれぞれ作成
        #y1 = np.array([np.mean(e) for e in E1])
        y2 = np.array([np.mean(e) for e in E2])

        # 各パスの変動比率を計算
        #tv_ratio_1 = calculate_tv_ratio(y1)
        tv_ratio_2 = calculate_tv_ratio(y2)
        #st.info(f"変動比率は、{tv_ratio_2}")
        max_tv_ratio = max(tv_ratio_1,tv_ratio_2)

        # 物質別の部屋に計算結果を安全にキャストして保存
        st.session_state[key_ti] = {
            "logZ1": float(np.ravel(logZ1)[0]),
            "logZ2": float(np.ravel(logZ2)[0]),
            "logBF": float(np.ravel(logZ2)[0]) - float(np.ravel(logZ1)[0]),
            "beta": beta1,
            "E1": E1,
            "E2": E2,
            "tv_ratio_1": tv_ratio_1,
            "tv_ratio_2": tv_ratio_2,
            "max_tv_ratio":max_tv_ratio,
            "conditions": conditions,   # 👈 ここに追加
        }
        # 計算が終わったらバーを消去してスッキリさせる
        progress_text.empty()
        progress_container.empty()
        #st.success("TI計算が完了しました！")
        st.success("TI Done!")
        st.rerun()  # 💡 必ずこのボタン処理ブロックの最後（インデントを合わせる）で実行します
    # 現在選択中の物質のTI結果がセッションにある場合のみ画面に出力

    if key_ti in st.session_state:
        ti = st.session_state[key_ti]
        
        st.write("### 📐 Model Evidence Comparison")
        st.caption("熱力学的積分（TI）によって算出された、各モデルの対数周辺尤度（logZ）の厳密な比較結果です。")

        col_ti1, col_ti2, col_ti3 = st.columns(3)
        with col_ti1:
            st.metric(label="🟢 1-break 対数周辺尤度 (logZ1)", value=f"{ti['logZ1']:.4f}")
        with col_ti2:
            st.metric(label="🔴 2-break 対数周辺尤度 (logZ2)", value=f"{ti['logZ2']:.4f}")
        with col_ti3:
            st.metric(label="⚖️ 対数ベイズファクター (logBF)", value=f"{ti['logBF']:.4f}", delta="2-break model" if ti['logBF'] > 0 else "1-break model")


        # グリッドプロットの描画
        st.markdown("---")
        st.write("#### 📈 Thermodynamic Integration Curve (β vs Expected Log-Likelihood)")
        st.caption("逆温度 β の変化に伴う期待対数尤度の推移です。この曲線の下側の面積の差が、モデルの優劣（周辺尤度の差）になります。")

        fig_ti, ax_ti = plt.subplots(figsize=(10, 4.5))
        ax_ti.plot(ti["beta"], [np.mean(e) for e in ti["E1"]], "o-", color="green", lw=1.5, label="1-break Path")
        ax_ti.plot(ti["beta"], [np.mean(e) for e in ti["E2"]], "s-", color="red", lw=1.5, label="2-break Path")
        ax_ti.set_xlabel("Inverse Temperature (Beta)")
        ax_ti.set_ylabel("Expected Log-Likelihood")
        ax_ti.grid(True, linestyle="--", alpha=0.4)
        
        # 凡例がはみ出さないよう外側に綺麗に配置
        ax_ti.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        
        # 画面にプロット
        st.pyplot(fig_ti, bbox_inches='tight')

        #-------------------------------------------------------------------------------------------------
        # 1. プロットに使われている「平均期待対数尤度」の配列をそれぞれ作成
        y1 = np.array([np.mean(e) for e in ti["E1"]])
        y2 = np.array([np.mean(e) for e in ti["E2"]])
        """
        def calculate_tv_ratio(y):
            # 実際の総移動距離（隣り合う点との差分の絶対値の合計）
            total_variation = np.sum(np.abs(np.diff(y)))
            # 始点と終点の純粋な差分（理想的な最小移動距離）
            net_change = np.abs(y[-1] - y[0])
            
            # ゼロ除算を防ぐ安全策を入れつつ比率を計算
            return total_variation / net_change if net_change != 0 else np.nan
        """
        # 各パスの変動比率を計算
        tv_ratio_1 = calculate_tv_ratio(y1)
        tv_ratio_2 = calculate_tv_ratio(y2)

        # --- 出力方法の例 ---
        # ① ターミナルやログに出く場合
        print(f"【TI品質指標】1-break Path 総変動比率: {tv_ratio_1:.2f}倍 (1.0に近いほど正常)")
        print(f"【TI品質指標】2-break Path 総変動比率: {tv_ratio_2:.2f}倍 (1.0に近いほど正常)")

        # ② Streamlit画面上にアラートやテキストで表示したい場合（グラフの上下などに追加）
        #import streamlit as st

        st.write(f"⚠️ **TI計算の安定性チェック (Total Variation Ratio):**\n"
                f"- 1-break Path: **{tv_ratio_1:.2f}倍**\n"
                f"- 2-break Path: **{tv_ratio_2:.2f}倍**\n\n"
                f"*※理論上は1.0に近い滑らかな曲線になります。数値が著しく大きい場合は、サンプリング不足による振動（破綻）をしています。")
        
                #\nMCMCのサンプル数（draws や tune）を増やす:
                #\n$\beta$ のステップ数（scheduling）を増やす、または配置を最適化する:
                #\nバーンイン（Burn-in / Warmup）期間を延ばす*")
        if n_beta <20:
            st.error(f"⚠️ **デバッグモードです。n_beta = {n_beta} が少なすぎるため、TI計算の安定性チェック は無効です**\n")

        max_tv_ratio = max(tv_ratio_1, tv_ratio_2)

        if max_tv_ratio >= 1.10:
            st.error(
                f"🚨 **警告: 再計算を強く推奨します（Total Variation Ratio: {max_tv_ratio:.2f}）**\n\n"
                f"{tv_ratio_1:.2f}倍と{tv_ratio_2:.2f}倍の変動比率は、理想的な1.0から大きく乖離しており、"
                "TI曲線に激しい振動が検出されました。サンプリング不足の可能性が高く、"
                "算出された **logZ および logBF 自体がまだ信用できない状態** です。\n\n"
                "【対策】MCMCのサンプル数（draws/tune）を増やすか、βのステップ数を増やしてください。"
            )
        elif max_tv_ratio >= 1.05:
            st.warning(
                f"⚠️ **注意: 再計算を推奨します（Total Variation Ratio: {max_tv_ratio:.2f}）**\n\n"
                f"{tv_ratio_1:.2f}倍と{tv_ratio_2:.2f}倍の変動比率は、理想的な1.0からやや乖離しています。サンプリング不足の可能性があり、"
                "TI曲線にわずかなガタつき（ノイズ）が見られます。結果の信頼性を高めるため、"
                "可能であればサンプリング数を増やして再計算することをおすすめします。"
            )
        else:
            st.success(
                f"✅ **チェック合格（Total Variation Ratio: {max_tv_ratio:.2f}）**\n\n"
                f"{tv_ratio_1:.2f}倍と{tv_ratio_2:.2f}倍の変動比率は、理想的な1.0に非常に近く、"
                "許容範囲内です（極めて軽微なサンプリングノイズのみ）。計算結果は安定しています。"
            )
        #-------------------------------------------------------------------------------------------------
        # 1. まず各モデルの滑らかさを完全に独立して評価する
        is_1break_valid = (tv_ratio_1 <= 1.05)
        is_2break_valid = (tv_ratio_2 <= 1.05)

        # 2. 判定ロジック
        if is_1break_valid and is_2break_valid:
            tv_txt = "✅ 両モデルとも計算は完全安定。BFをそのまま信頼してOKです。"

        elif is_1break_valid and not is_2break_valid:
            tv_txt = f"⚠️ 2-break側にサンプリングのノイズ（TV Ratio: {tv_ratio_2:.2f}）があるが、1-break（logBF）は信頼可能です。ノイズの原因が『モデルが複雑すぎるための彷徨い』であると解釈できる場合、TI法によるモデルの比較の試み自体が意味を成しませんので、この時点での中止を推奨します。さらに継続する場合には、βステップ数にするTune/Draw数を増加することを勧めます。この指標（tv_ratio）はβステップ数に依存します。"
           
        
        elif not is_1break_valid and is_2break_valid:
            tv_txt = "⚠️ 1-break側の計算が不安定です。設定を見直してください。"

        else:
            tv_txt ="❌ 両モデルとも計算が不安定です。βステップ数に対してTune/Draw数が不足しています。この指標はβステップ数に依存します。"
                #--------------------------------------------------------------------------------------------
        st.warning(f" {tv_txt}")
                    
        # 💡 おすすめ追加1: ジェフリーズの基準に基づく「証拠の強さ」の自動言語化判定
        st.markdown("---")
        st.write("#### 📝 Statistical Evidence Assessment (Jeffreys' Scale)")
        
        abs_bf = abs(ti['logBF'])

        if max_tv_ratio <1.05:
            if abs_bf < 1.0:
                evidence_str = "両モデルの差は無視できるレベル（Barely worth mentioning）です。"
                alert_func = st.warning
            elif 1.0 <= abs_bf < 2.5:
                evidence_str = "差はわずかに認められます（Substantial Evidence）が、決定的な差ではありません。"
                alert_func = st.info
            elif 2.5 <= abs_bf < 5.0:
                evidence_str = "統計的に強い証拠（Strong Evidence）が確認されました。"
                alert_func = st.success
            else:
                evidence_str = "統計的に極めて決定的な、圧倒的な証拠（Decisive / Very Strong Evidence）です。"
                alert_func = st.success

            if ti['logBF'] > 0:
                alert_func(f"🏆 **判定:** 構造変化点が2箇所ある **「2-breakモデル」** を支持する {evidence_str}")
            else:
                alert_func(f"🏆 **判定:** 構造変化点が1箇所のみの **「1-breakモデル」** を支持する {evidence_str}")
        else:
            evidence_str = "**判定不可:**算出された **logZ および logBF 自体がまだ信用できない状態** です "
            alert_func = st.error  #errorとする
            alert_func(f"🏆 **判定不能** {evidence_str}")

        #----------------------------------------------------------------------------------------------------------------

        # 💡 おすすめ追加2: 高解像度ダウンロード機能（凡例はみ出しガード付き）
        import io
        buf_ti = io.BytesIO()
        leg_ti = ax_ti.get_legend()
        fig_ti.savefig(buf_ti, format="png", dpi=300, bbox_extra_artists=(leg_ti,), bbox_inches='tight')
        
        st.download_button(
            label="🖼️ TIグリッドグラフを高解像度PNG(300dpi)としてダウンロード",
            data=buf_ti.getvalue(),
            file_name=f"{substance}_thermodynamic_integration_route.png",
            mime="image/png",
            key=f"dl_ti_btn_{substance}"
        )


# -----------------------------
# 6. Multi-substance Comparison
# -----------------------------

with tab6:
    st.header("Draw Substances Group Plot")
#from modules.multi_plot import plot_multi_substances

    selected =[]  # ここで選択された物質のリストを初期化
    group_chem = st.radio("group",[ "none", "PCB", "DDT",  "HCH", "Chlordane", "PCN", "HBCD" ], index=1, key="group_chem_radio")

    if group_chem == "PCB":
        selected = ["PCB-Mono", "PCB-Di", "PCB-Tri", "PCB-Tetra", "PCB-Penta", "PCB-Hexa", "PCB-Hepta", "PCB-Octa", "PCB-Nona", "PCB-Deca"]
    elif group_chem == "DDT":
        selected = ["p,p'-DDT", "p,p'-DDE", "p,p'-DDD", "o,p'-DDT", "o,p'-DDE", "o,p'-DDD"]
    elif group_chem == "HCH":
        selected = ["HCH-α", "HCH-β", "HCH-γ", "HCH-δ"]  # ← strip() 済み前提
    elif group_chem == "Chlordane":
        selected = ["Chlordane-cis", "Chlordane-trans", "Oxychlordane", "Nonachlor-cis", "Nonachlor-trans", "Heptachlor", "Heptachlor epoxide-cis", "Heptachlor epoxide-trans"]
    elif group_chem == "PCN":
        selected = ["PCN-Total", "PCN-Mono", "PCN-Di", "PCN-Tri", "PCN-Tetra", "PCN-Penta", "PCN-Hexa", "PCN-Hepta", "PCN-Octa"]
    elif group_chem == "HBCD":
        selected =["Hexabromocyclododecane_1,2,5,6,9,10-α", "hexabromocyclododecane_1,2,5,6,9,10-β", "hexabromocyclododecane_1,2,5,6,9,10-γ", "hexabromocyclododecane_1,2,5,6,9,10-δ", "hexabromocyclododecane_1,2,5,6,9,10-ε", ]


    #selected += st.multiselect("Select substances", substances, default=[]) 
    options = ["All"] + substances
    selected += st.multiselect( "add substances", options, key="batch_export_select", default=[substance] )

    # All が含まれていたら全物質に置き換え
    if "All" in selected:
        selected = substances
    # キー名の可読性を上げるため変数に格納
    session_key = f"fig_group_{selected}"

    # 1. まだセッション状態にキーが存在しない場合のみ、None で初期化（毎回リセットしない）
    if session_key not in st.session_state:
        st.session_state[session_key] = None


    redraw = False  # 描画フラグの初期化

    # 2. ボタンが押された時の処理
    if st.button("draw group", key="draw_group"):
        fig_group = plot_multi_substances(
            df=df,
            substances=substances,
            selected=selected,
            yaxis_mode=yaxis_mode,
            window=window,
            pops_year=pops_year,
            add_regulation_lines=add_regulation_lines,
            create_timeseries=create_timeseries,
            calc_stl=calc_stl,
            filter_start_date=filter_start_date
        )
        
        # セッション状態に保存
        st.session_state[session_key] = fig_group
        redraw = True  # 描画フラグを立てる

    # 3. グラフが保存されている（Noneではない）場合は常に描画する
    if st.session_state[session_key] is not None:
        if redraw:
            st.write(f"Generated plot for selected group: {', '.join(selected)}")
        else:
            st.write(f"Displaying previously generated plot for selected group: {', '.join(selected)}")
            redraw = False  # 描画フラグをリセット
        # 保存されているオブジェクト（またはリスト）を描画
        st.pyplot(st.session_state[session_key])

    st.header("Batch HTML Export")
    if st.button("Run Batch Export"):

        for sub in selected:

            st.write(f" Processing {sub} ...")

            # -----------------------------------------------------------
            # 1) ts_log（RAW 時系列）を取得 or 新規生成
            # -----------------------------------------------------------
            ts_log = st.session_state.get(f"df_sub_{sub}")
            if ts_log is None:
                ts_log = create_timeseries(df, sub)

                #ts_log = ts_log[ts_log.index >= pd.Timestamp(filter_start_date)]   # 2010年4月以降に絞り込みnadoの処理が必要な場合は、create_timeseries関数内で対応するか、ここでフィルタリングしてください。
                st.session_state[f"df_sub_{sub}"] = ts_log

            # -----------------------------------------------------------
            # 2) STL（trend / seasonal / resid）を取得 or 新規生成
            # -----------------------------------------------------------
            trend = st.session_state.get(f"trend_{sub}")
            seasonal = st.session_state.get(f"seasonal_{sub}")
            resid = st.session_state.get(f"resid_{sub}")
            
            if trend is None or seasonal is None or resid is None:
                if ts_log is None:                    
                    ts_log = create_timeseries(df, sub) # ここで再度作成しているのは、セッションに保存されていない場合に備えてです。create_timeseries関数内でフィルタリングなどの処理が必要な場合は、ここで対応してください。

                try:
                    res = calc_stl(ts_log)
                    trend = res.trend.dropna()
                except Exception as e:
                    st.warning(f"{sub} の STL 計算でエラー発生：{e}")
                    continue
                #res = calc_stl(ts_log)
                trend = res.trend
                seasonal = res.seasonal
                resid = res.resid

                st.session_state[f"trend_{sub}"] = trend
                st.session_state[f"seasonal_{sub}"] = seasonal
                st.session_state[f"resid_{sub}"] = resid

            # -----------------------------------------------------------
            # 3) Classical Breakpoint（pwlf）を取得 or 新規生成
            # -----------------------------------------------------------
            classical_fig = st.session_state.get(f"classical_fig_{sub}")
            classical_table = st.session_state.get(f"classical_table_{sub}")

            if classical_fig is None or classical_table is None:
                if ts_log is None:
                    ts_log = create_timeseries(df, sub) # ここで再度作成しているのは、セッションに保存されていない場合に備えてです。create_timeseries関数内でフィルタリングなどの処理が必要な場合は、ここで対応してください。
                classical_fig, classical_table, pwlf_model = run_classical_break_analysis(
                    trend,
                    ts_log,
                    yaxis_mode,
                    n_segments,
                    piecewise_fit,
                    calc_moving_slope,
                    pops_year,
                    add_regulation_lines,
                    sub
                )

                st.session_state[f"classical_fig_{sub}"] = classical_fig
                st.session_state[f"classical_table_{sub}"] = classical_table

            # -----------------------------------------------------------
            # 4) 図（fig_raw / fig_stl）を取得 or 新規生成
            # -----------------------------------------------------------
            fig_raw = st.session_state.get(f"fig_raw_{sub}")
            if fig_raw is None:
                fig_raw = create_raw_plot(
                    ts_log,
                    sub,
                    pops_year=pops_year,
                    add_regulation_lines=add_regulation_lines
                )
                st.session_state[f"fig_raw_{sub}"] = fig_raw

            fig_stl = st.session_state.get(f"fig_stl_{sub}")
            if fig_stl is None:
                fig_stl = plot_stl_components(
                    ts_log,
                    trend,
                    seasonal,
                    resid,
                    sub
                )
                st.session_state[f"fig_stl_{sub}"] = fig_stl

            # -----------------------------------------------------------
            # 5) Bayesian / TI / Recommendation などの結果を取得（存在しない場合は None）
            # -----------------------------------------------------------
            fig1 = st.session_state.get(f"fig1_{sub}")
            fig2 = st.session_state.get(f"fig2_{sub}")
            fig_hist = st.session_state.get(f"fig_hist_{sub}")
            fig_comp = st.session_state.get(f"fig_comp_{sub}")
            ti_result = st.session_state.get(f"ti_{sub}")
            ti_text = st.session_state.get(f"ti_text_{sub}", "")
            recommendation_title = st.session_state.get(f"rec_title_{sub}")
            recommendation_text = st.session_state.get(f"rec_text_{sub}")
            report_data = st.session_state.get(f"report_data_{sub}", {})
            breakpoint_table_html = st.session_state.get(f"bp_table_{sub}")

            # スコア類
            robustness_score_1 = st.session_state.get(f"robustness_score_1_{sub}")
            robustness_score_2 = st.session_state.get(f"robustness_score_2_{sub}")
            gap_s2 = st.session_state.get(f"gap_s2_{sub}")
            gap2 = st.session_state.get(f"gap2_{sub}")
            w2 = st.session_state.get(f"w2_{sub}")
            rmse_param_2 = st.session_state.get(f"rmse_param_2_{sub}")
            width_s2 = st.session_state.get(f"width_s2_{sub}")
            tau_penalty_2 = st.session_state.get(f"tau_penalty_2_{sub}")
            total_score_1 = st.session_state.get(f"total_score_1_{sub}")
            total_score_2 = st.session_state.get(f"total_score_2_{sub}")

            # -----------------------------------------------------------
            # 6) HTML レポート生成
            # -----------------------------------------------------------
            generate_html_report_full(
                sub,
                fig_raw,
                fig_stl,
                fig1,
                fig2,
                fig_hist,
                fig_comp,
                classical_fig,
                classical_table,
                ti_result,
                ti_text,
                robustness_score_1,
                robustness_score_2,
                gap_s2,
                w2,
                rmse_param_2,
                gap2,
                width_s2,
                tau_penalty_2,
                total_score_1,
                total_score_2,
                report_data,
                breakpoint_table_html,
            )

        st.success("Batch export completed")
    #----------------------------------------------------------- 
    
 
    import os
    import streamlit as st
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # --- (既存の k-Shape プロットコード) ---
    # ここに、calc_k_Shape 関数を呼び出して k-Shape クラスタリングを実行し、結果のプロットを生成するコードが入ります。
    st.header("k-Shape Clustering Results")
    n_clusters = st.number_input("number of clusters k-Shape", min_value=2, max_value=10, value=4, step=1, key="n_clusters_kshape")
    
    if st.button("Run k-Shape with HTML Export"):
        with st.spinner("Running k-Shape..."):
            result_ks, html_output = calc_k_Shape(
                df,
                substances,
                selected,
                yaxis_mode,
                window,
                pops_year,
                add_regulation_lines,
                create_timeseries,
                calc_stl,
                calc_moving_slope,
                n_clusters=n_clusters,
                filter_start_date=filter_start_date
            )

            st.success("k-Shape 完了！")


            # HTML 保存ボタン
            st.download_button(
                label="k-Shape 結果をダウンロード",
                data=html_output,
                file_name=f"{selected[0]}-clusters{n_clusters}-kshape_result.html",
                mime="text/html"
            )
