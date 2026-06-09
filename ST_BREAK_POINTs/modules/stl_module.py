#stl_module.py
from statsmodels.tsa.seasonal import STL

def calc_stl(ts_log):
    if ts_log is None or len(ts_log) == 0:
        raise ValueError("STL に渡された時系列データが空です。化合物名（{substance}）の不一致、フィルタ条件などを確認してください。")

    stl = STL(ts_log, period=12, robust=True)
    return stl.fit()

# stl_plotter.py
import matplotlib.pyplot as plt

def plot_stl_components(target_series, trend, seasonal, resid, substance):
    """STL 分解結果を 4 段プロットとして返す関数"""

    fig, axes = plt.subplots(
        4, 1,
        figsize=(10, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.2, 1.5]}
    )

    # --- 1段目：Observed（線 + 点）
    axes[0].plot(target_series, color="blue", label="Observed")
    axes[0].plot(target_series, "o", ms=3, alpha=0.6, color="blue")
    axes[0].legend(loc="upper left")

    # --- 2段目：Trend
    axes[1].plot(trend, label="Trend", color="orange")
    axes[1].legend(loc="upper left")

    # --- 3段目：Seasonal（線 + 点）
    axes[2].plot(seasonal, label="Seasonal", color="green")
    axes[2].plot(seasonal, "o", ms=3, alpha=0.6, color="green")
    axes[2].legend(loc="upper left")

    # --- 4段目：Residual（点 + 0ライン + σ帯）
    sigma = resid.std()

    axes[3].axhspan(-2*sigma, 2*sigma, color="gray", alpha=0.08, label="±2σ")
    axes[3].axhspan(-sigma, sigma, color="gray", alpha=0.15, label="±1σ")
    axes[3].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    axes[3].plot(resid, "o", ms=3, alpha=0.7, color="red", label="Residual")
    axes[3].legend(loc="upper left")

    plt.tight_layout()

    return fig

# modules/plot_raw_mode.py
import matplotlib.pyplot as plt
import io

def create_raw_plot(ts_log, substance, pops_year=False, add_regulation_lines=None):
    """RAW グラフを作成して fig を返す"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts_log.index, ts_log.values, "o-", markersize=4, alpha=0.7, label="RAW log")

    if pops_year and add_regulation_lines is not None:
        add_regulation_lines(ax)

    ax.set_title(f"{substance} RAW log concentration")
    ax.set_xlabel("Month/Year")
    ax.set_ylabel("log concentration")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def create_mode_plot(ts_log, substance, yaxis_mode, window, pops_year=False,
                     add_regulation_lines=None, calc_stl=None, calc_moving_slope=None):
    """STL / Normalized / Moving slope の切替グラフを作成して fig を返す"""

    fig, ax = plt.subplots(figsize=(12, 4))

    # --- STL 計算 ---
    res = calc_stl(ts_log)
    trend = res.trend.dropna()

    if yaxis_mode == "STL":
        ax.plot(trend.index, trend.values, "o-", color="orange", markersize=4)
        title = f"{substance} STL Trend"
        ylabel = "STL Trend"

    elif yaxis_mode == "Normalized":
        norm = trend - trend.iloc[0]
        ax.plot(norm.index, norm.values, "o-", color="green", markersize=4)
        title = f"{substance} Normalized STL"
        ylabel = "Δ log concentration (Initial Base)"

    elif yaxis_mode == "Moving slope":
        if calc_moving_slope is None:
            raise ValueError("calc_moving_slope 関数が指定されていません")

        ms = calc_moving_slope(trend, window=window)
        ax.plot(ms.index, ms.values, "o-", color="purple", markersize=4)
        title = f"{substance} Moving slope"
        ylabel = "slope / month"

    # --- 規制イベント線 ---
    if pops_year and add_regulation_lines is not None:
        add_regulation_lines(ax)

    ax.set_title(title)
    ax.set_xlabel("Month/Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


def fig_to_png_bytes(fig):
    """fig を PNG バイト列に変換"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    return buf.getvalue()

# modules/multi_plot.py
import matplotlib.pyplot as plt
from modules.classical_break import calc_moving_slope

def plot_multi_substances(
    df,
    substances,
    selected,
    yaxis_mode,
    window,
    pops_year,
    add_regulation_lines,
    create_timeseries,
    calc_stl,
    filter_start_date=None
):
    """
    複数物質をまとめてプロットする関数。group_mapは本体に移す
    # -------------------------
    # グループ選択
    # -------------------------
    group_map = {
        "PCB": ["PCB-Total", "PCB-Mono", "PCB-Di", "PCB-Tri", "PCB-Tetra",
                "PCB-Penta", "PCB-Hexa", "PCB-Hepta", "PCB-Octa", "PCB-Nona", "PCB-Decac"],
        "DDT": ["p,p'-DDT", "p,p'-DDE", "p,p'-DDD", "o,p'-DDT", "o,p'-DDE", "o,p'-DDD"],
    }

    selected = []
    if group_chem in group_map:
        selected = group_map[group_chem]
  """   
  
    # -------------------------
    # プロット準備
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    title = f"Multiple Substances - {yaxis_mode}"
    ax.set_title(title)
    
    for sub in selected:

        # --- 時系列生成 ---
        ts = create_timeseries(df, sub)

        # --- STL が必要な場合 ---
        if yaxis_mode in ["STL", "Normalized", "Moving slope"]:
            try:
                res = calc_stl(ts)
                trend = res.trend.dropna()
            except Exception as e:
                print(f"[WARNING] {sub} の STL 計算でエラー発生：{e}")
                continue

        # -------------------------
        # モード別プロット
        # -------------------------
        if yaxis_mode == "RAW":
            ax.plot(ts.index, ts.values, label=sub)

        elif yaxis_mode == "STL":
            ax.plot(trend.index, trend.values, label=sub)


        elif yaxis_mode == "Normalized":
            if filter_start_date is not None:
                # フィルタ開始日以降の最初の値を基準にする
                base_value = trend.loc[trend.index >= pd.Timestamp(filter_start_date)].iloc[0]
            else:
                # 従来通り
                base_value = trend.iloc[0]

            processed_ts = trend - base_value

        #elif yaxis_mode == "Normalized":
        #    norm = trend - trend.iloc[0]
            ax.plot(processed_ts.index, processed_ts.values, label=sub)

        elif yaxis_mode == "Moving slope":
            ms = calc_moving_slope(trend, window=window)
            ax.plot(ms.index, ms.values, label=sub)

    # -------------------------
    # 装飾
    # -------------------------
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.6)

    if pops_year:
        add_regulation_lines(ax)

    ax.legend()

    return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import io
import base64

#from modules.utils import add_regulation_lines
#from modules.stl_module import calc_stl, calc_moving_slope

def calc_k_Shape(
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
    n_clusters=3,
    filter_start_date=None
):

    ts_dict = {}

    for sub in selected:
        ts = create_timeseries(df, sub)
        if ts.index.duplicated().any():
            st.warning(f"{sub} に重複した日付がありました。継続のため、最初の値のみ使用します。データ確認を推奨します。")
            ts = ts[~ts.index.duplicated(keep="first")]

        # ★ index の重複を削除（最初の値を採用）
        #ts_log = ts_log[~ts_log.index.duplicated(keep="first")]

        # ★ フィルタ適用
        if filter_start_date is not None:
            ts = ts[ts.index >= pd.Timestamp(filter_start_date)]

        # --- モード別のデータ処理 ---
        if yaxis_mode == "RAW":
            # RAWデータの場合はそのまま使用（あとでTimeSeriesScalerMeanVarianceで標準化されます）
            processed_ts = ts.dropna()
            
        elif yaxis_mode in ["STL", "Normalized", "Moving slope"]:
            try:
                res = calc_stl(ts)
                trend = res.trend.dropna()
            except Exception as e:
                st.warning(f"{sub} の STL 計算でエラー発生：{e}")
                continue

            if yaxis_mode == "STL":
                processed_ts = trend

            elif yaxis_mode == "Normalized":    #基準年の変更　統一
                if filter_start_date is not None:
                    # フィルタ開始日以降の最初の値を基準にする
                    base_value = trend.loc[trend.index >= pd.Timestamp(filter_start_date)].iloc[0]
                else:
                    # 従来通り
                    base_value = trend.iloc[0]

                processed_ts = trend - base_value
            #lif yaxis_mode == "Normalized":
            #    processed_ts = trend - trend.iloc[0]

            elif yaxis_mode == "Moving slope":
                processed_ts = calc_moving_slope(trend, window=window).dropna()

        # 物質名をキーにしてデータを保存
        ts_dict[sub] = processed_ts
        


    if not ts_dict:
        st.error("有効な時系列データがありません。")
        return

    # 2. すべての物質を一つの DataFrame に結合（行＝物質、列＝時間軸）
    # ※各物質のデータポイント数（長さ）が揃っている必要があります
    cluster_df = pd.DataFrame(ts_dict).T 

    st.dataframe(cluster_df, use_container_width=True)


    # 欠損値があるとk-Shapeが動かないため、念のため線形補間＆残ったNaNを削除
    cluster_df = cluster_df.interpolate(axis=1).dropna()

    #st.write("### 各物質の有効データ数（欠損値を除く）")
    #st.write(cluster_df.notnull().sum())    

  

    if cluster_df.empty:
        st.error("物質間でデータ期間が一致しないか、データが不足しています。")
        return
    # グループ選択されたデータフレーム（df_groupなど）の有効データ数を表示する例

    # 3. k-Shape用のデータ整形と学習
    data = cluster_df.values  # 形状: (サンプル数[物質数], タイムステップ数)
    X_train = TimeSeriesScalerMeanVariance().fit_transform(data)

    # ★ 物質数チェック（ここを追加）
    if data.shape[0] < n_clusters:
        st.error(f"There are only {data.shape[0]} valid objects for the specified number of clusters {n_clusters}. Try filtering the data or reducing the number of clusters. Clustering cannot be performed.")
        return

    ks = KShape(n_clusters=n_clusters, verbose=False, random_state=42)
    y_pred = ks.fit_predict(X_train)

    # 4. 結果の可視化（Streamlit対応）
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes] # クラスタ数が1の場合の配列化対策

    for sz in range(n_clusters):
        ax = axes[sz]
        # 各クラスターに属する物質のインデックスを取得
        cluster_indices = np.where(y_pred == sz)[0]
        
        # 各クラスターに属する時系列（標準化後）をプロット
        for idx in cluster_indices:
            sub_name = cluster_df.index[idx]
            ax.plot(X_train[idx].ravel(), alpha=0.4, label=sub_name)
            
        # クラスターの重心（平均的な波形）を太い赤線でプロット
        ax.plot(ks.cluster_centers_[sz].ravel(), "r-", linewidth=3, label="Centroid")
        ax.set_title(f"Cluster {sz + 1} (n = {len(cluster_indices)})")
        ax.grid(True, linestyle="--", alpha=0.5)
        # 物質数が多い場合は凡例が溢れるため、適宜コメントアウトしてください
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1), fontsize='small')

    plt.tight_layout()
    st.pyplot(fig) # Streamlit画面にプロットを出力

    # 5. 結果テーブル
    results = pd.DataFrame({'Substance': cluster_df.index, 'Cluster': y_pred + 1})
    results = results.sort_values(by="Cluster").reset_index(drop=True)

    st.markdown("### 📊 クラスタリング結果")
    st.dataframe(results, use_container_width=True)

    # ============================
    # ★ HTML 出力用の準備
    # ============================

    # プロット画像を PNG に変換
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # DataFrame → HTML
    table_html = results.to_html(index=False)

    # HTML 全体を組み立てる
    html_output = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
        </style>
    </head>
    <body>
        <h2>k-Shape Clustering Result</h2>

        <h3>Cluster Plot</h3>
        <img src="data:image/png;base64,{img_base64}" style="max-width:100%;">

        <h3>Cluster Table</h3>
        {table_html}
    </body>
    </html>
    """

    return results, html_output








  

