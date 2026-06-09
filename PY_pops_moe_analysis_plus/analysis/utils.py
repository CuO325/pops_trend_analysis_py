import pandas as pd
import numpy as np

# analysis/utils.py

def load_data(filepath):
    """
    データを読み込み、対数変換を行う。
    不検出データ（concentration <= detection_limit）は、
    検出下限値の 1/sqrt(2) で代替して対数変換する。
    
    前提: データフレームには 'concentration', 'year', 'detection_limit' の列が必要。
    """
    df = pd.read_csv(filepath)
    
    # 不検出データの特定
    is_nondetect = df['concentration'] <= df['detection_limit']
    
    # 不検出データの代替処理: 検出下限値の 1/sqrt(2)
    # これにより、対数変換された値は検出下限値の対数よりも小さくなる
    substitute_value = df['detection_limit'] / np.sqrt(2)
    
    # 不検出箇所を代替値で置換
    df.loc[is_nondetect, 'concentration_sub'] = substitute_value[is_nondetect]
    # 検出箇所はそのまま
    df.loc[~is_nondetect, 'concentration_sub'] = df.loc[~is_nondetect, 'concentration']
    
    # 対数変換
    df['log_conc'] = np.log(df['concentration_sub'])
    
    return df