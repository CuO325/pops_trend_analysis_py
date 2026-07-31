# streamlit run ti_fft_kde_regression_v1e.py

import os
import streamlit as st
import pandas as pd
from analysis.config import BayesConfig, build_beta_grid
from analysis.run_analysis3 import run_ols_analysis, run_mle_and_aic, run_fft_bayes, generate_html_report, run_ti

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import re
import datetime 
import corner
import seaborn as sns


""" -- Known issue: If the substance name is the same, the session is overwritten, resulting in unintended output.
 This issue can be addressed by changing the names of the chemicals, such as "PCB_Water" and "PCB_Air". """

def clean_name(name):
    safe_name = re.sub(r'[\\/:"*?<>|,\'.]', '_', name)
    return safe_name

def clean_value_user(val):
    if isinstance(val, str):
        val = val.replace('tr(', '').replace(')', '')
        val = val.replace('---', '')
        val = val.replace('nd','0')
    try:
        if val == '': return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan

# --- ページ設定 ---
st.set_page_config(
    page_title="Bayesian Trend Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 物質ごとの結果格納辞書を事前に初期化
for key in ["ols_results", "mle_results", "bayes_results", "ti_results"]:
    if key not in st.session_state:
        st.session_state[key] = {}


# ==============================================================================
# 📁 左側サイドバー（元の Tab 1: Data Entry + 元の Tab 3: MCMC Settings）
# ==============================================================================
with st.sidebar:
    st.header("📁 Data Entry")
    uploaded = st.file_uploader("Select a CSV file", type=["csv"])

    if uploaded:
        df_all = pd.read_csv(uploaded)
        file_name = uploaded.name if hasattr(uploaded, 'name') else "uploaded_data.csv"
        file_name = file_name.rsplit('.', 1)[0]  # 拡張子を除去
        
        st.caption(f"**Loaded File:** {uploaded.name}")
        
        # substance名を安全化
        df_all["safe_substance"] = df_all["substance"].apply(clean_name)
        df_all["safe_f_substance"] = file_name + "_" + df_all["safe_substance"]
        
        df_all["concentration"] = df_all["concentration"].apply(clean_value_user)
        df_all["detection_limit"] = df_all["detection_limit"].apply(clean_value_user)

        is_nondetect = df_all["concentration"] <= df_all["detection_limit"]
        substitute_value = df_all["detection_limit"] / np.sqrt(2)
        df_all["concentration_sub"] = df_all["concentration"].copy()
        df_all.loc[is_nondetect, "concentration_sub"] = substitute_value[is_nondetect]
        df_all["log_conc"] = np.log10(df_all["concentration_sub"])
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["log_conc", "detection_limit", "concentration"]
        )

        # 変換されたデータを保存
        output_path = f"data/{file_name}_conv.csv"
        df_all.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        # 他のタブでも使えるように Session State へ格納
        st.session_state["df_all"] = df_all
        st.session_state["substances"] = df_all['substance'].unique()
        st.session_state["file_name"] = file_name

    st.markdown("---")

    # 旧 Tab 3 の設定項目
    st.header("⚙️ MCMC settings (config.py)")
    n_walkers = st.number_input("n_walkers", 4, 256, 32)
    n_steps = st.number_input("n_steps", 100, 50000, 3000)
    burn_frac = st.slider("burn_frac", 0.1, 0.9, 0.5)
    bw_factor = st.number_input("bw_factor", 0.1, 5.0, 1.0)
    grid_size_lookup = st.selectbox("grid_size_lookup", [1024, 2048, 4096])
    mode_loo = st.checkbox("mode_loo (LOO)", value=True)
    beta_schedule = st.selectbox("Beta Schedule", [ "test", "recommended", "geometric", "default", "coarse", "fine", " ultra_fine"])

    cfg = BayesConfig(
        mode_loo=mode_loo,
        grid_size_lookup=grid_size_lookup,
        bw_factor=bw_factor,
        n_walkers=n_walkers,
        n_steps=n_steps,
        burn_frac=burn_frac,
        beta_schedule=beta_schedule
    )
    st.session_state["cfg"] = cfg
    st.success("Settings saved")


# ==============================================================================
# 📊 右側メイン画面（元の Tab 2 と Tab 4 のみで構成、プレビューもここに配置）
# ==============================================================================
st.title("Bayesian Trend Analysis GUI")

# メイン画面のタブを2つ（またはプレビュー用を含めて3つ）にする
main_tab1, main_tab2, main_tab3 = st.tabs(["📁 Data Preview", "📊 Run Analysis", "📈 Plot"])

# -----------------------------
# 📁 Main Tab 1: Data Preview (サイドバーで読み込んだデータの確認)
# -----------------------------
with main_tab1:
    st.header("Data Preview")
    if "df_all" not in st.session_state:
        st.info("Please select and load a CSV file from the sidebar.")
    else:
        df_all = st.session_state["df_all"]
        file_name = st.session_state["file_name"]
        substances = st.session_state["substances"]
        
        st.write(f"**File name:** {file_name}")
        st.write(f"**Target Substances:** {list(substances)}")
        st.write("**Column Names:**", df_all.columns.tolist())
        st.write(f"**Number of lines:** {len(df_all)}")
        
        st.subheader("Preview Table")
        st.dataframe(df_all.head())
        
        st.subheader("Missing Values")
        st.write(df_all.isna().sum())


# -----------------------------
# 📊 Main Tab 2: Run Analysis
# -----------------------------
with main_tab2:
    if st.button("Reset session"):
        for key in ["ols_results", "mle_results", "bayes_results", "ti_results"]:
            st.session_state[key] = {}
        st.rerun()

    st.header("Run Analysis")

    if "df_all" not in st.session_state:
        st.warning("Please load the data first.")
    else:
        df_all = st.session_state["df_all"]
        substances = st.session_state["substances"]
        file_name = st.session_state["file_name"]

        # --- 1. OLS 単独実行 ---
        if st.button("Ordinary Least Squares (OLS)"):
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"\n==================== {substance} ====================")

                df = df_all[df_all['substance'] == substance].copy()
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['year'])

                if len(df) < 24 or len(df["log_conc"].unique()) < 3:
                    st.write("Skip due to insufficient data or too many NDs.")
                    continue

                st.subheader("OLS Results")
                ols_result = run_ols_analysis(df)
                
                ols_df = pd.DataFrame([ols_result]).drop(columns=[c for c in ["summary_html", "df"] if c in ols_result])
                st.write(ols_df)
                st.session_state["ols_results"][safe_f_substance] = ols_result

        # --- 2. MLE 単独実行 ---
        if st.button("MLE + AIC"):
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"\n==================== {substance} ====================")
                
                df = df_all[df_all['substance'] == substance].copy()
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['year'])

                if len(df) < 24 or len(df["log_conc"].unique()) < 3:
                    st.write("Skip due to insufficient data or too many NDs.")
                    continue

                st.subheader("MLE Results (Based on the MOE Method)")
                mle_result = run_mle_and_aic(df)
                st.write(mle_result)
                st.session_state["mle_results"][safe_f_substance] = mle_result

        # --- 3. Bayes 単独実行 ---
        if st.button("FFT-KDE Bayesian"):
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"\n==================== {substance} ====================")
                
                df = df_all[df_all['substance'] == substance].copy()
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['year'])

                if len(df) < 24 or len(df["log_conc"].unique()) < 3:
                    st.write("Skip due to insufficient data or too many NDs.")
                    continue

                st.subheader("FFT-KDE Bayesian Results")
                cfg = st.session_state["cfg"]
                result_bayes = run_fft_bayes(df, cfg)
                st.write(result_bayes)
                st.session_state["bayes_results"][safe_f_substance] = result_bayes

        # --- 4. 3手法同時実行 + HTML出力 ---
        if st.button("OLS, MLE and FFT-KDE with HTML output"):
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"\n==================== {substance} ====================")

                df = df_all[df_all['substance'] == substance].copy()
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['year'])

                if len(df) < 24 or len(df["log_conc"].unique()) < 3:
                    st.warning(f"{substance}: Skip due to insufficient data.")
                    continue

                st.subheader("OLS 結果")
                ols_result = run_ols_analysis(df)
                st.session_state["ols_results"][safe_f_substance] = ols_result

                st.subheader("MLE Results (Based on the MOE Method)")
                mle_result = run_mle_and_aic(df)
                st.session_state["mle_results"][safe_f_substance] = mle_result

                st.subheader("FFT-KDE ベイズ結果")
                cfg = st.session_state["cfg"]
                bayes_result = run_fft_bayes(df, cfg)
                st.session_state["bayes_results"][safe_f_substance] = bayes_result

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("outputs/plots_regression", exist_ok=True)    
                plot_path = f"plots_regression/{safe_f_substance}_regression_{timestamp}.png"
                out_path = f"outputs/{safe_f_substance}_report_{timestamp}.html"

                path = generate_html_report(df, ols_result, mle_result, bayes_result, None, plot_path, out_path, safe_f_substance)
                st.success(f"{substance}: The HTML report has been saved: {path}")

        # --- 5. TI を含む全手法実行 ---
        options = ["All"] + list(substances)
        selected_substances = st.multiselect("Select the chemical to run thermodynamic integral (TI)", options, default=["All"], key="ti_run_button_out")

        if st.button("Perform OLS, MLE, Bayesian and TI ", key="ti_run_button"):
            start_time = datetime.datetime.now()
            st.write(start_time.strftime("%Y-%m-%d %H:%M:%S") + " The TI script has started. Please wait...")
            cfg = st.session_state["cfg"]

            target_list = substances if "All" in selected_substances else selected_substances

            for substance in target_list:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"\n==================== {substance} ====================")

                df = df_all[df_all['substance'] == substance].copy()
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                df = df.dropna(subset=['year'])

                if len(df) < 24 or len(df["log_conc"].unique()) < 3:
                    st.warning(f"{substance}: Skip due to insufficient data.")
                    continue

                ols_result = run_ols_analysis(df)
                st.session_state["ols_results"][safe_f_substance] = ols_result

                mle_result = run_mle_and_aic(df)
                st.session_state["mle_results"][safe_f_substance] = mle_result

                bayes_result = run_fft_bayes(df, cfg)
                st.session_state["bayes_results"][safe_f_substance] = bayes_result

                progress_box = st.empty()
                table_box = st.empty()
                progress_bar = st.progress(0)
                ti_result = run_ti(df, cfg, safe_f_substance, progress_box=progress_box, table_box=table_box, progress_bar=progress_bar)
                st.session_state["ti_results"][safe_f_substance] = ti_result

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = f"plots_regression/{safe_f_substance}_regression_{timestamp}.png"
                out_path = f"outputs/{file_name}_{clean_name(substance)}_loo_{str(cfg.mode_loo)}_{cfg.beta_schedule}_report_{timestamp}.html"
                
                path = generate_html_report(df, ols_result, mle_result, bayes_result, ti_result, plot_path, out_path, safe_f_substance)
                st.success(f"{substance}: The HTML report has been saved: {path}")

            end_time = datetime.datetime.now()
            st.success(f"{(end_time - start_time).total_seconds()/60:.2f} min for n={len(target_list)} substance TI execution")

        # --- 6. HTML再生成 ---
        html_options = set()
        for d in [st.session_state["ols_results"], st.session_state["mle_results"], st.session_state["bayes_results"], st.session_state["ti_results"]]:
            for k, v in d.items():
                if "substance" in v:
                    html_options.add(v["substance"])
        
        html_options = ["All"] + sorted(list(html_options))
        selected_html_substances = st.multiselect("Select the material for creating HTML", html_options, default=["All"], key="html_run_button_out")
        
        if st.button("Generate HTML for the selected substance", key="html_run_button"):          
            target_list = substances if "All" in selected_html_substances else selected_html_substances

            for substance in target_list:
                safe_f_substance = file_name + "_" + clean_name(substance)
                
                ols_result = st.session_state["ols_results"].get(safe_f_substance)
                mle_result = st.session_state["mle_results"].get(safe_f_substance)
                bayes_result = st.session_state["bayes_results"].get(safe_f_substance)
                ti_res = st.session_state["ti_results"].get(safe_f_substance)

                if not any([ols_result, mle_result, bayes_result]):
                    st.warning(f"No analysis results found for {substance}. Please run analysis first.")
                    continue

                df = df_all[df_all['substance'] == substance].copy()
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
                df = df.dropna(subset=['year'])

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = f"plots_regression/{safe_f_substance}_regression_{timestamp}.png" 
                out_path = f"outputs/{safe_f_substance}_after_report_{timestamp}.html"

                path = generate_html_report(df, ols_result, mle_result, bayes_result, ti_res, plot_path, out_path, safe_f_substance)           
                st.success(f"HTML generated for {substance}: {path}")


# -----------------------------
# 📈 Main Tab 3: Plots
# -----------------------------
with main_tab3:
    st.header("Bayesian posterior distribution (slope a)")

    if "bayes_results" not in st.session_state or len(st.session_state["bayes_results"]) == 0:
        st.info("Please run FFT-KDE Bayes first.")
    else:
        bayes_keys = list(st.session_state["bayes_results"].keys())
        display_options = {st.session_state["bayes_results"][k]["substance"]: k for k in bayes_keys}

        selected_substance_name = st.selectbox(
            "Select substance",
            ["-- Select substance --"] + list(display_options.keys())
        )

        if selected_substance_name != "-- Select substance --":
            target_key = display_options[selected_substance_name]
            result = st.session_state["bayes_results"][target_key]
            samples = result["samples"]

            st.subheader("Bayes settings used for this analysis")
            cfg = result.get("config", None)

            if cfg is None:
                st.info("No configuration was saved for this run.")
            else:
                st.write(cfg)
                st.write(vars(cfg))

            # ---- Corner plot ----
            figure = corner.corner(samples, labels=["a", "b"])
            st.pyplot(figure)
            plt.close(figure)

            # ---- Posterior of slope a ----
            a_samps = samples[:, 0]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(a_samps, bins=40, density=True, alpha=0.7, color="skyblue")
            ax.axvline(result["a_mean"], color="red", linestyle="--", label="mean")
            ax.axvline(result["ci_a"][0], color="green", linestyle=":", label="2.5%")
            ax.axvline(result["ci_a"][1], color="green", linestyle=":", label="97.5%")
            ax.set_title(f"Posterior of slope a ({selected_substance_name})")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            # ---- Joint posterior ----
            st.header("Joint posterior (a vs b)")
            a_mean = result["a_mean"]
            b_mean = result["b_mean"]

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, label="samples")

            sns.kdeplot(
                x=samples[:, 0],
                y=samples[:, 1],
                fill=True,
                cmap="Blues",
                alpha=0.4,
                ax=ax
            )

            ax.scatter(a_mean, b_mean, color="red", s=50, label="posterior mean")
            ax.set_xlabel("slope a")
            ax.set_ylabel("intercept b")
            ax.set_title(f"Joint posterior samples ({selected_substance_name})")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)