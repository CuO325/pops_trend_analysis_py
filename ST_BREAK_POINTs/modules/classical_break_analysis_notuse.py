# modules/classical_break_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm


def prepare_trend_local(trend_global, ts_log, yaxis_mode, calc_moving_slope):
    """Tab3 の yaxis_mode に応じて trend_local を生成"""
    if yaxis_mode == "STL":
        return trend_global
    elif yaxis_mode == "RAW":
        return ts_log
    elif yaxis_mode == "Normalized":
        return trend_global - trend_global.iloc[0]
    elif yaxis_mode == "Moving slope":
        return calc_moving_slope(trend_global)
    else:
        raise ValueError(f"Unknown yaxis_mode: {yaxis_mode}")


def extract_break_dates(pwlf_model, trend_local):
    """pwlf の breakpoints を日付に変換"""
    break_dates = []
    for bp in pwlf_model.fit_breaks[1:-1]:
        idx = int(np.clip(round(bp), 0, len(trend_local)-1))
        break_dates.append(trend_local.index[idx])
    return break_dates


def compute_interval_ols(trend_local, break_dates):
    """区間ごとに OLS を実行し、回帰線と統計テーブルを返す"""
    break_indices = [0]
    for bd in break_dates:
        idx = trend_local.index.get_indexer([bd], method='nearest')[0]
        break_indices.append(idx)
    break_indices.append(len(trend_local))

    table_data = []
    ols_segments = []

    for i in range(len(break_indices)-1):
        start = break_indices[i]
        end = break_indices[i+1]
        segment = trend_local.iloc[start:end]

        if len(segment) < 12:
            continue

        X = np.arange(len(segment))
        X2 = sm.add_constant(X)
        model_seg = sm.OLS(segment.values, X2).fit()

        intercept = model_seg.params[0]
        slope = model_seg.params[1]
        y_reg = intercept + slope * X

        ols_segments.append((segment.index, y_reg, intercept, slope))

        table_data.append({
            'Section': i+1,
            'Start': segment.index[0].strftime('%Y-%m'),
            'End': segment.index[-1].strftime('%Y-%m'),
            'Slope': round(slope, 5),
            'P-value': round(model_seg.pvalues[1], 5),
            'R²': round(model_seg.rsquared, 4)
        })

    return ols_segments, pd.DataFrame(table_data)


def plot_classical_break(
    trend_local,
    pwlf_model,
    break_dates,
    ols_segments,
    yaxis_mode,
    substance,
    pops_year,
    add_regulation_lines
):
    """Classical Breakpoint Analysis のメイングラフを生成して fig を返す"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # 元系列
    ax.plot(trend_local.index, trend_local.values, 'o-', alpha=0.6,
            color="gray", label=f"Trend ({yaxis_mode})")

    # pwlf フィット線
    x_hat = np.linspace(0, len(trend_local)-1, 300)
    y_hat = pwlf_model.predict(x_hat)
    start_date = trend_local.index[0]
    date_hat = [start_date + relativedelta(months=int(i)) for i in x_hat]
    ax.plot(date_hat, y_hat, 'r-', linewidth=3, label='Piecewise Regression')

    # 区間 OLS
    for idx, (x_index, y_reg, intercept, slope) in enumerate(ols_segments, start=1):
        ax.plot(x_index, y_reg, linewidth=3, linestyle="--", label=f"Segment {idx}")
        text_pos_x = x_index[len(x_index)//2]
        ax.text(text_pos_x, np.mean(y_reg),
                f'y={intercept:.1f}+{slope:.4f}x',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 変曲点
    for bd in break_dates:
        ax.axvline(bd, color='green', linestyle='--', alpha=0.8,
                   label=f"Break: {bd.strftime('%Y-%m')}")

    # 規制年
    if pops_year:
        add_regulation_lines(ax)

    ax.set_title(f"{substance} — Classical Breakpoint Analysis ({yaxis_mode})")
    ax.set_xlabel("Month/Year")
    ax.set_ylabel(f"{yaxis_mode}")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 凡例整理
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc="upper left", bbox_to_anchor=(1.01, 1))

    return fig


def run_classical_break_analysis(
    trend_global,
    ts_log,
    yaxis_mode,
    n_segments,
    piecewise_fit,
    calc_moving_slope,
    pops_year,
    add_regulation_lines,
    substance
):
    """
    Classical Breakpoint Analysis の一連の処理を実行し、
    fig と table を返す。
    """

    # 1) 表示用系列の生成
    trend_local = prepare_trend_local(trend_global, ts_log, yaxis_mode, calc_moving_slope)

    # 2) pwlf フィット
    pwlf_model = piecewise_fit(trend_local, n_segments=n_segments)

    # 3) breakpoints の日付変換
    break_dates = extract_break_dates(pwlf_model, trend_local)

    # 4) 区間 OLS
    ols_segments, df_table = compute_interval_ols(trend_local, break_dates)

    # 5) 図の生成
    fig = plot_classical_break(
        trend_local,
        pwlf_model,
        break_dates,
        ols_segments,
        yaxis_mode,
        substance,
        pops_year,
        add_regulation_lines
    )

    return fig, df_table
