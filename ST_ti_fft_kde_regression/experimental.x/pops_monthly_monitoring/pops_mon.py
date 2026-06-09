import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ページの設定
st.set_page_config(page_title="POPs月次モニタリング＆警鐘システム", layout="wide")
st.title("🧪 POPs月次モニタリング＆警鐘システム")
st.markdown("SARIMAモデルを用いたPCB濃度の時系列解析と、予測区間に基づく異常検知を行います。")

# サイドバー：設定エリア
st.sidebar.header("⚙️ 設定・パラメータ")

# 1. データの読み込み
uploaded_file = st.sidebar.file_uploader("PCBデータのCSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    # データの読み込みと初期クリーンアップ
    df = pd.read_csv(uploaded_file)
    df = df.replace('---', np.nan)

    concentration_cols = ['1日目', '2日目', '3日目']
    for col in concentration_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Concentration'] = df[concentration_cols].mean(axis=1)
    df.rename(columns={'年': 'year', '月': 'month'}, inplace=True)
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    ts_data = df[['Date', 'Concentration']].copy()
    ts_data = ts_data.set_index('Date').asfreq('MS')

    # サイドバーで解析開始年を選択できるようにする
    min_year = ts_data.index.min().year
    max_year = ts_data.index.max().year
    start_year = st.sidebar.slider("解析開始年", min_year, max_year, 2010)
    
    # データの絞り込みと対数変換
    ts_data_clean = ts_data[ts_data.index >= f"{start_year}-01-01"].copy()
    
    # 欠損値処理
    if ts_data_clean['Concentration'].isnull().sum() > 0:
        ts_data_clean['Concentration'] = ts_data_clean['Concentration'].ffill()
        ts_data_clean.dropna(inplace=True)
        
    ts_data_clean['Log_Concentration'] = np.log(ts_data_clean['Concentration'])
    ts_final = ts_data_clean['Log_Concentration']

    # SARIMAパラメータの入力設定
    st.sidebar.markdown("---")
    st.sidebar.subheader("SARIMAモデルの次数設定")
    p = st.sidebar.number_input("p (AR次数)", min_value=0, max_value=5, value=0)
    d = st.sidebar.number_input("d (階数)", min_value=0, max_value=2, value=1)
    q = st.sidebar.number_input("q (MA次数)", min_value=0, max_value=5, value=1)
    
    P = st.sidebar.number_input("P (季節性AR次数)", min_value=0, max_value=5, value=1)
    D = st.sidebar.number_input("D (季節性階数)", min_value=0, max_value=2, value=1)
    Q = st.sidebar.number_input("Q (季節性MA次数)", min_value=0, max_value=5, value=1)
    
    forecast_steps = st.sidebar.slider("予測期間 (ヶ月)", 1, 36, 12)

    # 💡 画面をタブに分割して整理
    tab1, tab2, tab3 = st.tabs(["📊 データ確認・前処理", "📈 時系列分解・定常性確認", "🚨 モデル予測・警鐘シミュレーション"])

    # --- TAB 1: データ確認 ---
    with tab1:
        st.subheader("📋 データの基本情報")
        col1, col2, col3 = st.columns(3)
        col1.metric("解析期間", f"{ts_data_clean.index.min().strftime('%Y/%m')} 〜 {ts_data_clean.index.max().strftime('%Y/%m')}")
        col2.metric("データポイント総数", f"{len(ts_data_clean)} 点")
        col3.metric("補完後の欠損値数", f"{ts_data_clean['Log_Concentration'].isnull().sum()} 点")
        
        st.markdown("### データの先頭15行")
        st.dataframe(ts_data_clean.head(15), use_container_width=True)

        st.markdown("### 総PCB濃度（対数変換後）の経年変化")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(ts_data_clean.index, ts_data_clean['Log_Concentration'], label='対数濃度', color='tab:blue')
        ax1.set_title('総PCB濃度（対数変換後）の経年変化')
        ax1.set_xlabel('年')
        ax1.set_ylabel('ln(濃度)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)

    # --- TAB 2: 分解と検定 ---
    with tab2:
        st.subheader("🔍 トレンド・季節性の分解")
        decomposition = seasonal_decompose(ts_final, model='additive', period=12) 
        fig2 = decomposition.plot()
        fig2.set_size_inches(12, 8)
        st.pyplot(fig2)
        
        st.subheader("🧐 定常性の検証 (ADF検定)")
        result = adfuller(ts_final.dropna())
        
        col_adf1, col_adf2 = st.columns(2)
        col_adf1.write(f"**ADF統計量:** {result[0]:.4f}")
        col_adf2.write(f"**p値:** {result[1]:.4f}")
        
        if result[1] > 0.05:
            st.warning(f"⚠️ p値が0.05より大きいため、データは非定常である可能性が高いです。(推奨: 階数 d=1)")
        else:
            st.success(f"✅ p値が0.05以下であるため、データは定常であると見なせます。(推奨: 階数 d=0)")

        st.markdown("### 差分適用後のACF / PACF")
        ts_diff1 = ts_final.diff(periods=1).dropna()
        ts_diff12 = ts_diff1.diff(periods=12).dropna()
        
        fig3, axes = plt.subplots(2, 1, figsize=(10, 6))
        sm.graphics.tsa.plot_acf(ts_diff12, lags=30, ax=axes[0])
        axes[0].set_title('自己相関関数 (ACF)')
        sm.graphics.tsa.plot_pacf(ts_diff12, lags=30, ax=axes[1])
        axes[1].set_title('偏自己相関関数 (PACF)')
        plt.tight_layout()
        st.pyplot(fig3)

    # --- TAB 3: 予測と警鐘 ---
    with tab3:
        st.subheader("🤖 SARIMAモデルによる予測と警鐘基準")
        
        order = (p, d, q)
        seasonal_order = (P, D, Q, 12)
        
        with st.spinner("モデルをフィッティング中..."):
            try:
                model = SARIMAX(ts_final, 
                                order=order, 
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                
                # AICの表示
                st.info(f"✨ 選択されたモデル: SARIMA{order} x {seasonal_order[:-1]}_12  |  **AIC: {results.aic:.4f}**")
                
                # 予測計算
                forecast = results.get_forecast(steps=forecast_steps)
                mean_forecast_log = forecast.predicted_mean
                confidence_intervals_log = forecast.conf_int(alpha=0.05)
                
                mean_forecast_conc = np.exp(mean_forecast_log)
                lower_bound_conc = np.exp(confidence_intervals_log['lower Log_Concentration'])
                upper_bound_conc = np.exp(confidence_intervals_log['upper Log_Concentration'])
                ts_actual_conc = np.exp(ts_final)
                
                # メインプロット
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                ax4.plot(ts_actual_conc.tail(36), label='実測濃度 (pg/m3)', color='black', marker='o')
                ax4.plot(mean_forecast_conc, label='予測平均濃度', color='red', linestyle='-')
                ax4.fill_between(mean_forecast_conc.index, 
                                 lower_bound_conc, 
                                 upper_bound_conc, 
                                 color='pink', alpha=0.4, label='95%予測区間')
                ax4.plot(upper_bound_conc, linestyle='--', color='red', label='警鐘基準 (95%上限)')
                ax4.set_title('PCB濃度予測と警鐘基準')
                ax4.set_xlabel('年')
                ax4.set_ylabel('PCB濃度 (pg/m3)')
                ax4.legend()
                ax4.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig4)
                
                # 警鐘基準のアラート表示
                st.error("🚨 **警鐘基準の定義:** 次回の実測値が、予測区間の上限（赤の点線）を上回った場合に、警鐘（異常増加）と判断します。")
                
                # 残差の診断プロットをアコーディオンに隠す（画面スッキリ化）
                with st.expander("📊 モデル残差の診断プロットを確認する"):
                    fig5 = results.plot_diagnostics(figsize=(10, 8))
                    st.pyplot(fig5)
                    
            except Exception as e:
                st.error(f"モデルの構築中にエラーが発生しました。次数パラメータを調整してください。 エラー内容: {e}")

else:
    st.info("👈 左側のサイドバーからPCBデータのCSVファイルをアップロードしてください。")