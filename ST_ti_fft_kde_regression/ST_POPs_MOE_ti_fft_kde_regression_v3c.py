# streamlit run ti_fft_kde_regression_v1e.py より、UIボタンの形状, figのHTML埋め込み処理の追加に伴う変更

# UI for Bayesian Trend Analysis with Thermodynamic Integral (TI) using Streamlit
# streamlit run ST_POPs_MOE_ti_fft_kde_regression_v3c.py


import os
import streamlit as st
import pandas as pd
from analysis.config import BayesConfig, build_beta_grid
from analysis.run_analysis3 import run_ols_analysis, run_mle_and_aic, run_fft_bayes, generate_html_report, run_ti , plot_regression_results2

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import re
import datetime 
import corner
import seaborn as sns
# ---- primaryボタンを緑色にするCSSスタイル定義 ----
st.markdown(
    """
    <style>
    /* type="primary" のボタン背景を赤色に変更 */
    div.stButton > button[kind="primary"] {
        background-color: #ff4b4b !important; /* 赤色 */
        color: white !important;               /* 文字色 */
        border-color: #ff4b4b !important;
    }
    
    /* マウスホバー時の色 */
    div.stButton > button[kind="primary"]:hover {
        background-color: #e03e3e !important;
        border-color: #e03e3e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---- secondaryボタンを緑色にするCSSスタイル定義 ----
st.markdown(
    """
    <style>
    /* type="secondary" のボタンを緑色にカスタマイズ */
    div.stButton > button[kind="secondary"] {
        background-color: #28a745 !important; /* ボタンの背景色（緑） */
        color: white !important;               /* 文字色 */
        border-color: #28a745 !important;       /* 枠線の色 */
    }
    /* ホバー（カーソルを乗せたとき）の色 */
    div.stButton > button[kind="secondary"]:hover {
        background-color: #218838 !important;
        border-color: #1e7e34 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- tertiaryボタンを青色にするCSSスタイル定義 ---
st.markdown("""
    <style>
    /* type="tertiary" のボタンを青色にカスタマイズ */
    div.stButton > button[kind="tertiary"] {
        background-color: #007bff !important; /* ボタンの背景色（青） */
        color: white !important;               /* 文字色 */
        border-color: #007bff !important;       /* 枠線の色 */
    }
    /* ホバー（カーソルを乗せたとき）の色 */
    div.stButton > button[kind="tertiary"]:hover {
        background-color: #0069d9 !important;
        border-color: #0056b3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
#-- Known issue: If the substance name is the same, the session is overwritten, resulting in unintended output.
# This issue can be addressed by changing the names of the chemicals, such as "PCB_Water" and "PCB_Air". 

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
# キーが存在しない場合のみ初期化
if "substances" not in st.session_state:
    st.session_state["substances"] = []

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
    else:
        st.info("Please upload a CSV file to proceed.")

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
#import numpy as np

def make_bayes_light(result_bayes):
    keys_to_keep = ["safe_f_substance", "substance", "analysis_type", "a_mean", "b_mean", "ci_a","ci_b","Rhat","p_neg_slope","p_pos_slope","steps_completed","stopped_early","mcmc_summary","config"]
    light = {k: result_bayes[k] for k in keys_to_keep if k in result_bayes}
    return light


def to_serializable(obj):
    # NumPy 配列
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # NumPy 数値
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    # dict
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]

    # dataclass / クラス → __dict__ に変換
    if hasattr(obj, "__dict__"):
        return {k: to_serializable(v) for k, v in obj.__dict__.items()}

    # その他はそのまま返す（str, int, float など）
    return obj


def save_result_to_file(result, safe_f_substance, method):
    os.makedirs("outputs/results", exist_ok=True)
    out_path = f"outputs/results/{safe_f_substance}_{method}.json"

    serializable_result = to_serializable(result)

    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)

    return out_path



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

        st.caption(f"ℹ️ 対象データセット: **{len(substances)} 物質** の全データを順次解析します。")

        # --- Section 1: 個別手法の実行（画面表示のみ / HTML出力なし） ---
        st.subheader("1. 全物質: 個別手法の実行 (Json保存 / HTMLなし)")
        st.caption("全物質について指定の解析手法のみを実行し、jsonファイルとして保存します。HTMLには4で保存できます")

        # 1. カラムを作成してボタンだけを横並びに配置
        col1, col2, col3 = st.columns(3)

        # 押されたアクションを記録する変数を初期化
        action = None

        with col1:
            if st.button("📊 OLS (最小二乗法)", use_container_width=True):
                action = "ols"

        with col2:
            if st.button("📈 MLE + AIC", use_container_width=True):
                action = "mle"

        with col3:
            if st.button("🧬 FFT-KDE Bayesian", use_container_width=True):
                action = "bayes"


        # 2. カラムブロック（with構文）を出た「外側」で表示処理を実行する
        if action == "ols":
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"==================== {substance} ====================")
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
                save_result_to_file(ols_result, safe_f_substance, "ols")

        elif action == "mle":
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"==================== {substance} ====================")
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
                save_result_to_file(mle_result, safe_f_substance, "mle")

        elif action == "bayes":
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"==================== {substance} ====================")
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
                #save_result_to_file(result_bayes, safe_f_substance, "bayes")   #samplesを含めすべて保存する場合。
                save_result_to_file(make_bayes_light(result_bayes), safe_f_substance, "bayes")  #samplesを保存しない軽量版


        # --- Section 2: 全手法の一括実行 & レポート生成（HTML出力あり） ---
        st.subheader("2. 全物質: 全手法の一括解析 & HTMLレポート生成")
        st.caption("💡 全物質に対して OLS / MLE / ベイズ を一括実行し、解析結果を **HTMLファイル** として自動保存します。jsonファイルは作成しません。") #st.info

        if st.button("🚀 全手法を一括実行 ＋ HTMLレポート保存", type="tertiary", use_container_width=True):
            for substance in substances:
                safe_f_substance = file_name + "_" + clean_name(substance)
                st.write(f"\n==================== {substance} ====================")

                df = df_all[df_all['substance'] == substance].copy()
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
                df = df.dropna(subset=['year'])

                if len(df) < 24 or len(df["log_conc"].unique()) < 3:
                    st.warning(f"{substance}: Skip due to insufficient data.")
                    continue
                st.caption("OLS is running...") 
                #st.subheader("OLS ")
                ols_result = run_ols_analysis(df)
                st.session_state["ols_results"][safe_f_substance] = ols_result

                st.caption("MLE is running...")
                #st.subheader("MLE (Based on the MOE Method)")
                mle_result = run_mle_and_aic(df)
                st.session_state["mle_results"][safe_f_substance] = mle_result

                st.caption("FFT-KDE ベイズ is running...")
                cfg = st.session_state["cfg"]
                bayes_result = run_fft_bayes(df, cfg)
                st.session_state["bayes_results"][safe_f_substance] = bayes_result

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                os.makedirs("outputs/plots_regression", exist_ok=True)    
                plot_path = f"plots_regression/{safe_f_substance}_regression_{timestamp}.png"
                out_path = f"outputs/{safe_f_substance}_report_{timestamp}.html"

                fig, plot_path = plot_regression_results2(df, substance, file_name, safe_f_substance, ols_result, mle_result, bayes_result)
                fig.savefig(f"outputs/{plot_path}", dpi=150, bbox_inches="tight")

                path = generate_html_report(df, ols_result, mle_result, bayes_result, None, plot_path, out_path, safe_f_substance)
                st.success(f"📄 {substance}: HTMLレポートを保存しました: {out_path}")
                #表示の調整
                abs_path = os.path.abspath(path)

                if os.path.exists(path):

                    # Streamlit 内で表示
                    with open(path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800, scrolling=True)

                else:
                    st.error(f"ファイルが見つかりません: {path}")

    # --- 5. TI を含む全手法実行 ---
    st.subheader("3. 個別物質: Thermodynamic Integral (TI) & 全手法の一括解析 & HTMLレポート生成")
    st.caption("💡 選択された物質に対して Thermodynamic Integral (TI) を含む全手法を一括実行し、解析結果を **HTMLファイル** として自動保存します。")
    st.info("ℹ️  Beta Scheduleをtestに設定し　$\hat{R}$ (R-hat) の発散・不収束、$E[\log L]$（対数尤度の期待値）の対数爆発・異常値、$Var[\log L]$（尤度分散）の極端な巨大化の有無の確認し、破綻していれば、別の評価指標（AICやBayesian MCMC単体）にフォールバックすることを推奨します。")
    if st.session_state["substances"] is None or len(st.session_state["substances"]) == 0:
        st.warning("Please load the data first.")
    else:
        options = ["All"] + list(st.session_state["substances"])
        selected_substances = st.multiselect("Select the chemical to run thermodynamic integral (TI)", options, default=["All"], key="ti_run_button_out")

    if st.button("TIを含めた全行程（ OLS, MLE, Bayesian and TI ）の実行 ＋ HTMLレポート保存", key="ti_run_button",   type="primary", use_container_width=True):
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

            fig, plot_path = plot_regression_results2(df, substance, file_name, safe_f_substance, ols_result, mle_result, bayes_result)
            fig.savefig(f"outputs/{plot_path}", dpi=150, bbox_inches="tight")

            path = generate_html_report(df, ols_result, mle_result, bayes_result, ti_result, plot_path, out_path, safe_f_substance)
            st.success(f"{substance}: The HTML report has been saved: {path}")

        end_time = datetime.datetime.now()
        st.success(f"{(end_time - start_time).total_seconds()/60:.2f} min for n={len(target_list)} substance TI execution")

    # --- 6. HTML再生成 ---
    st.subheader("4. 個別物質: HTMLレポートの生成")
    st.caption("💡 既に実行された OLS / MLE / ベイズ  / TI の解析結果を **HTMLファイル** として保存します。")
    html_options = set()
    for d in [st.session_state["ols_results"], st.session_state["mle_results"], st.session_state["bayes_results"], st.session_state["ti_results"]]:
        for k, v in d.items():
            if "substance" in v:
                html_options.add(v["substance"])
    
    html_options = ["All"] + sorted(list(html_options))
    selected_html_substances = st.multiselect("Select the material for creating HTML", html_options, default=["All"], key="html_run_button_out")
    
    if st.button("Generate HTML for the selected substance", key="html_run_button", type="secondary", use_container_width=True):          
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

            fig, plot_path = plot_regression_results2(df, substance, file_name, safe_f_substance, ols_result, mle_result, bayes_result)
            fig.savefig(f"outputs/{plot_path}", dpi=150, bbox_inches="tight")

            path = generate_html_report(df, ols_result, mle_result, bayes_result, ti_res, plot_path, out_path, safe_f_substance)           
            st.success(f"HTML generated for {substance}: {path}")

            # ─── 自動でブラウザで開く処理 ───
            import webbrowser
            # 相対パスを絶対パスに変換して開く
            abs_path = os.path.abspath(path)
            webbrowser.open(f"file://{abs_path}")

# -----------------------------
# 📈 Main Tab 3: Plots
# -----------------------------
import io

# --- Streamlit UI メイン処理 ---
with main_tab3:
    st.header("Bayesian posterior distribution (slope a)")

    if (
        "bayes_results" not in st.session_state
        or len(st.session_state["bayes_results"]) == 0
    ):
        st.info("Please run FFT-KDE Bayes first.")
    else:
        bayes_keys = list(st.session_state["bayes_results"].keys())
        display_options = {
            st.session_state["bayes_results"][k]["substance"]: k
            for k in bayes_keys
        }


        selected_substance_name = st.selectbox(
        "Select substance tab3",
        ["-- Select substance --"] + list(display_options.keys()),
        key="select_substance_tab3",  # ← 他のselectboxと重複しないキー（任意の文字列）を指定
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

            # --------------------------------------------------
            # 1. Corner plot
            # --------------------------------------------------
            figure = corner.corner(samples, labels=["a", "b"])
            st.pyplot(figure)

            buf_corner = io.BytesIO()
            figure.savefig(buf_corner, format="png", bbox_inches="tight")
            st.download_button(
                label="Download Corner Plot (PNG)",
                data=buf_corner.getvalue(),
                file_name=f"corner_plot_{selected_substance_name}.png",
                mime="image/png",
            )
            plt.close(figure)

            # --------------------------------------------------
            # 2. Posterior of slope a
            # --------------------------------------------------
            a_samps = samples[:, 0]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(a_samps, bins=40, density=True, alpha=0.7, color="skyblue")
            ax.axvline(
                result["a_mean"], color="red", linestyle="--", label="mean"
            )
            ax.axvline(
                result["ci_a"][0], color="green", linestyle=":", label="2.5%"
            )
            ax.axvline(
                result["ci_a"][1], color="green", linestyle=":", label="97.5%"
            )
            ax.set_title(f"Posterior of slope a ({selected_substance_name})")
            ax.legend()
            st.pyplot(fig)

            buf_a = io.BytesIO()
            fig.savefig(buf_a, format="png", bbox_inches="tight")
            st.download_button(
                label="Download Slope a Plot (PNG)",
                data=buf_a.getvalue(),
                file_name=f"posterior_slope_a_{selected_substance_name}.png",
                mime="image/png",
            )
            plt.close(fig)

            # --------------------------------------------------
            # 3. Joint posterior
            # --------------------------------------------------
            st.header("Joint posterior (a vs b)")
            a_mean = result["a_mean"]
            b_mean = result["b_mean"]

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(
                samples[:, 0], samples[:, 1], s=5, alpha=0.3, label="samples"
            )

            sns.kdeplot(
                x=samples[:, 0],
                y=samples[:, 1],
                fill=True,
                cmap="Blues",
                alpha=0.4,
                ax=ax,
            )

            ax.scatter(
                a_mean, b_mean, color="red", s=50, label="posterior mean"
            )
            ax.set_xlabel("slope a")
            ax.set_ylabel("intercept b")
            ax.set_title(
                f"Joint posterior samples ({selected_substance_name})"
            )
            ax.legend()

            st.pyplot(fig)

            buf_joint = io.BytesIO()
            fig.savefig(buf_joint, format="png", bbox_inches="tight")
            st.download_button(
                label="Download Joint Posterior Plot (PNG)",
                data=buf_joint.getvalue(),
                file_name=f"joint_posterior_{selected_substance_name}.png",
                mime="image/png",
            )
            plt.close(fig)

            # --------------------------------------------------
            # 4. 回帰直線比較プロット (新規追加・整合部分)
            # --------------------------------------------------
            st.header("Regression Fits Comparison")

            # データとパラメータの準備
            df_all = st.session_state.get("df_all")
            if df_all is not None:
                df_substance = df_all[
                    df_all["substance"] == selected_substance_name
                ].copy()

                # 安定したファイル名・安全な物質名の生成
                file_name = st.session_state.get("file_name", "dataset")
                safe_f_substance = re.sub(
                    r'[\\/*?:"<>|]', "_", selected_substance_name
                )

                # OLS/MLE 参照 (result 辞書に含まれていなければ st.session_state 等から取得)
                ols_res = result.get("ols_result") or st.session_state.get(
                    "ols_results", {}
                ).get(target_key)
                mle_res = result.get("mle_result") or st.session_state.get(
                    "mle_results", {}
                ).get(target_key)

                # 回帰プロット関数の呼び出し
                fig_reg, plot_path = plot_regression_results2(
                    df=df_substance,
                    substance=selected_substance_name,
                    file_name=file_name,
                    safe_f_substance=safe_f_substance,
                    ols_result=ols_res,
                    mle_result=mle_res,
                    bayes_result=result,  # ここに result を渡す
                )

                st.pyplot(fig_reg)

                # ダウンロードボタンの設定
                buf_reg = io.BytesIO()
                fig_reg.savefig(
                    buf_reg, format="png", dpi=150, bbox_inches="tight"
                )
                st.download_button(
                    label="Download Regression Plot (PNG)",
                    data=buf_reg.getvalue(),
                    file_name=os.path.basename(plot_path),
                    mime="image/png",
                )
                plt.close(fig_reg)