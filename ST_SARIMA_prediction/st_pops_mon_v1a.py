import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
# 自動設定のため
from pmdarima.arima import auto_arima

# ページの設定
st.set_page_config(page_title="POPs月次モニタリング＆警鐘システム", layout="wide")
st.title("🧪 POPs月次モニタリング＆警鐘システム")
st.markdown("SARIMAモデルを用いたPCB濃度の時系列解析と、予測区間に基づく異常検知を行います。")

st.sidebar.header("⚙️ 設定・パラメータ")

# 1. データの読み込み
uploaded_file = st.sidebar.file_uploader("PCBデータのCSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    # データの読み込みと初期クリーンアップ
    df = pd.read_csv(uploaded_file)
    df = df.replace('---', np.nan)
    
    # 💡 修正1: .unique の後ろに () が抜けていたのを修正
    substances = df['調査対象物質'].unique()
    
    # 💡 修正2: 最初に物質を選択させる（セレクトボックスを上に移動）
    substance = st.selectbox("調査対象物質を選択してください", substances)
    
    # 💡 修正3: 選択された物質だけでデータを絞り込む
    df_sub = df[df['調査対象物質'] == substance].copy()
    
    # プレビュー表示（絞り込んだデータ）
    st.write(f"表示中の物質: {substance}")
    st.write(df_sub) 

    concentration_cols = ['1日目', '2日目', '3日目']
    for col in concentration_cols:
        if col in df_sub.columns:
            df_sub[col] = pd.to_numeric(df_sub[col], errors='coerce')

    df_sub['Concentration'] = df_sub[concentration_cols].mean(axis=1)
    df_sub.rename(columns={'年': 'year', '月': 'month'}, inplace=True)
    
    # yearとmonthからDateを作成
    df_sub['Date'] = pd.to_datetime(df_sub[['year', 'month']].assign(day=1))
    
    # 💡 修正4: 物質を絞り込んだ後の df_sub を使うことで、日付の重複を解消
    ts_data = df_sub[['Date', 'Concentration']].copy()
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("SARIMAモデルの次数設定")

    # Session State の初期化（自動計算された値を保持する箱を作る）
    if 'p_val' not in st.session_state: st.session_state.p_val = 0
    if 'd_val' not in st.session_state: st.session_state.d_val = 1
    if 'q_val' not in st.session_state: st.session_state.q_val = 1
    if 'P_val' not in st.session_state: st.session_state.P_val = 1
    if 'D_val' not in st.session_state: st.session_state.D_val = 1
    if 'Q_val' not in st.session_state: st.session_state.Q_val = 1

    # 🤖 自動計算ボタンの設置
    if st.sidebar.button("🪄 パラメータの自動設定", type="primary"):
        with st.spinner("最適な次数を探索中（10〜20秒かかります）..."):
            try:
                # auto_arimaの実行 (季節周期12ヶ月、ステップ探索で高速化)
                auto_model = auto_arima(
                    ts_final, 
                    seasonal=True, m=12,
                    start_p=0, max_p=3, start_q=0, max_q=3,
                    start_P=0, max_P=2, start_Q=0, max_Q=2,
                    d=0, D=1, # 階数を固定してあげると劇的に計算が安定します（推奨）
                    maxiter=50,       # 反復回数を増やして収束しやすくする
                    method='nm',      # 収束しにくいデータに強いネルダーミード法を指定
                    trace=False, 
                    error_action='ignore', 
                    suppress_warnings=True, 
                    stepwise=True
)
                #d=None, D=None, # 階数（d, D）も自動判定
                # 計算された結果を Session State に保存（書き換え）
                print(auto_model)
                best_order = auto_model.order
                print(best_order)
                best_seasonal_order = auto_model.seasonal_order
                print(best_seasonal_order)
                st.session_state.p_val = best_order[0]
                st.session_state.d_val = best_order[1]
                st.session_state.q_val = best_order[2]
                st.session_state.P_val = best_seasonal_order[0]
                st.session_state.D_val = best_seasonal_order[1]
                st.session_state.Q_val = best_seasonal_order[2]
                
                st.sidebar.success("✅ 最適なパラメータを見つけました！")
                # 💡【重要】これを追加して画面を強制的にリフレッシュし、UIに値を反映させる
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"自動計算中にエラーが発生しました: {e}")

    # 入力フォーム（value に st.session_state の値を指定するのがポイント）
    p = st.sidebar.number_input("p (AR次数)", min_value=0, max_value=5, value=st.session_state.p_val, key="p_input") #1
    d = st.sidebar.number_input("d (階数)", min_value=0, max_value=2, value=st.session_state.d_val, key="d_input")  #0
    q = st.sidebar.number_input("q (MA次数)", min_value=0, max_value=5, value=st.session_state.q_val, key="q_input")    #1
    
    P = st.sidebar.number_input("P (季節性AR次数)", min_value=0, max_value=5, value=st.session_state.P_val, key="P_input")  #1
    D = st.sidebar.number_input("D (季節性階数)", min_value=0, max_value=2, value=st.session_state.D_val, key="D_input")    #1
    Q = st.sidebar.number_input("Q (季節性MA次数)", min_value=0, max_value=5, value=st.session_state.Q_val, key="Q_input")  #1
    
    # ユーザーが手動で数値をいじった場合も値を同期させる
    st.session_state.p_val = p
    st.session_state.d_val = d
    st.session_state.q_val = q
    st.session_state.P_val = P
    st.session_state.D_val = D
    st.session_state.Q_val = Q

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
        
        st.markdown("データの先頭5行")
        st.dataframe(ts_data_clean.head(5), use_container_width=True)

        st.markdown(f"### {substance}（自然対数変換後）の経年変化")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(ts_data_clean.index, ts_data_clean['Log_Concentration'], label='対数濃度', color='tab:blue')
        ax1.set_title(f'{substance}濃度（自然対数変換後）の経年変化')
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
                ax4.set_title(f'{substance}濃度予測と警鐘基準')
                ax4.set_xlabel('年')
                ax4.set_ylabel(f'{substance}濃度 (pg/m3)')
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