import pandas as pd

def add_regulation_lines(ax):
    # POPs 規制年
    reg_years = {
        "2009-05": "First Addition (9 POPs) / PFOS (Annex B)",
        "2011-04": "Endosulfan",
        "2013-05": "HBCD",
        "2015-05": "PCN / SCCPs / HCBD / PCP",
        "2017-05": "DecaBDE",
        "2019-05": "PFOA (Annex A) / PFOS Review / HCBD Review",
        "2022-06": "PFHxS / Chlordecone / Chlorpyrifos / Dechlorane Plus / UV-328"
    }

    # COVID-19
    pandemic = {
        "2019-12": "COVID-19 outbreak begins",
        "2020-03": "WHO declares pandemic",
        "2023-05": "WHO ends PHEIC"
    }

    # テキスト位置調整
    ylim = ax.get_ylim()
    text_y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.05

    # 規制イベント（赤）
    for date_str, label in reg_years.items():
        x = pd.Timestamp(date_str)
        ax.axvline(x, color='red', linestyle='--', alpha=0.3)
        ax.text(
            x, text_y_pos, label,
            rotation=90, fontsize=10, color='red',
            verticalalignment='top', horizontalalignment='left'
        )

    # パンデミック（緑）
    for date_str, label in pandemic.items():
        x = pd.Timestamp(date_str)
        ax.axvline(x, color='green', linestyle='--', alpha=0.3)
        ax.text(
            x, text_y_pos, label,
            rotation=90, fontsize=10, color='green',
            verticalalignment='top', horizontalalignment='left'
        )

def add_regulation_lines_date(ax, trend):
    reg_years = {
        "2009-05": "First Addition (9 POPs) / PFOS (Annex B)",
        "2011-04": "Endosulfan",
        "2013-05": "HBCD",
        "2015-05": "PCN / SCCPs / HCBD / PCP",
        "2017-05": "DecaBDE",
        "2019-05": "PFOA (Annex A) / PFOS Review / HCBD Review",
        "2022-06": "PFHxS / UV-328 etc."
    }

    pandemic = {
        "2019-12": "COVID-19 outbreak begins",
        "2020-03": "WHO declares pandemic",
        "2023-05": "WHO ends PHEIC"
    }

    ylim = ax.get_ylim()
    text_y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.05

    # --- 規制イベント（赤） ---
    for d_str, label in reg_years.items():
        dt = pd.to_datetime(d_str)
        if trend.index.min() <= dt <= trend.index.max():
            idx = trend.index.get_indexer([dt], method="nearest")[0]
            ax.axvline(idx, color="red", linestyle="--", alpha=0.3)
            ax.text(idx, text_y_pos, label, rotation=90, fontsize=9, color="red",
                    verticalalignment="top")

    # --- パンデミック（緑） ---
    for d_str, label in pandemic.items():
        dt = pd.to_datetime(d_str)
        if trend.index.min() <= dt <= trend.index.max():
            idx = trend.index.get_indexer([dt], method="nearest")[0]
            ax.axvline(idx, color="green", linestyle="--", alpha=0.3)
            ax.text(idx, text_y_pos, label, rotation=90, fontsize=9, color="green",
                    verticalalignment="top")

