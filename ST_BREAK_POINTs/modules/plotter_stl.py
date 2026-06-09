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

# modules/multi_plot.py
import matplotlib.pyplot as plt
from modules.classical_break import calc_moving_slope

def plot_multi_substances(
    df,
    substances,
    group_chem,
    yaxis_mode,
    window,
    pops_year,
    add_regulation_lines,
    create_timeseries,
    calc_stl
):
    """
    複数物質をまとめてプロットする関数。
    Tab6 のコードを完全に関数化したもの。
    """

    # -------------------------
    # グループ選択
    # -------------------------
    group_map = {
        "PCB": ["PCB-Total", "PCB-Mono", "PCB-Di", "PCB-Tri", "PCB-Tetra",
                "PCB-Penta", "PCB-Hexa", "PCB-Hepta", "PCB-Octa", "PCB-Nona", "PCB-Decac"],
        "DDT": ["p,p'-DDT", "p,p'-DDE", "p,p'-DDD", "o,p'-DDT", "o,p'-DDE", "o,p'-DDD"],
        "HCH": ["HCH-α", "HCH-β", "HCH-γ", "HCH-δ"],
        "Chlordane": ["Chlordane-cis", "Chlordane-trans", "Oxychlordane",
                      "Nonachlor-cis", "Nonachlor-trans", "Heptachlor",
                      "Heptachlor epoxide-cis", "Heptachlor epoxide-trans"],
        "PCN": ["PCN-Total", "PCN-Mono", "PCN-Di", "PCN-Tri", "PCN-Tetra",
                "PCN-Penta", "PCN-Hexa", "PCN-Hepta", "PCN-Octa"],
        "HBCD": ["Hexabromocyclododecane_1,2,5,6,9,10-α",
                 "hexabromocyclododecane_1,2,5,6,9,10-β",
                 "hexabromocyclododecane_1,2,5,6,9,10-γ",
                 "hexabromocyclododecane_1,2,5,6,9,10-δ",
                 "hexabromocyclododecane_1,2,5,6,9,10-ε"]
    }

    selected = []
    if group_chem in group_map:
        selected = group_map[group_chem]

    # -------------------------
    # プロット準備
    # -------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

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
            norm = trend - trend.iloc[0]
            ax.plot(norm.index, norm.values, label=sub)

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
