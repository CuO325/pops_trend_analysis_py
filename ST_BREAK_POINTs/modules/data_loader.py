import pandas as pd
import numpy as np

def clean_value(val):
    if isinstance(val, str):
        val = (
            val.replace("tr(", "")
               .replace(")", "")
               .replace("---", "")
               .replace("nd", "")
               .replace(",", "")
        )
    try:
        if val == "":
            return np.nan
        return float(val)
    except:
        return np.nan

def create_timeseries(df, substance):
    df_sub = df[df["調査対象物質"] == substance].copy()
    df_sub["Date"] = pd.to_datetime(df_sub["年"].astype(str) + "-" + df_sub["月"].astype(str))
    df_sub.set_index("Date", inplace=True)
    ts_raw = df_sub["月平均値"].apply(clean_value)
    ts_log = np.log(ts_raw)
    return ts_log.sort_index().dropna()
