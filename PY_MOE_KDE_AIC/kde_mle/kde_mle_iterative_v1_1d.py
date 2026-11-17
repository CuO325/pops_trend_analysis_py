#!/usr/bin/env python3
"""KDE-MLE による POPs 年次トレンド解析（環境省「黒本」手法の再現）.

本スクリプトは、環境モニタリングデータ（POPs 等）に対して以下の 5 つの
解析を物質ごとに実行し、結果を CSV に出力する。

1. KDE-MLE（simple）        : 全残差から単一 KDE を構築するベースライン
2. KDE-MLE（iterative）     : 残差 KDE を反復更新する主軸モデル
3. KDE-MLE（MOE Fig.5）     : 年度別残差 KDE を合成する環境省図 5 準拠モデル
4. ブートストラップ検定      : 前期・後期の平均値差 95% CI
5. 二項ロジスティック回帰    : ND（不検出）割合の経年傾向

実行方法::

    cd PY_MOE_KDE_AIC/kde_mle
    python kde_mle_iterative_v1_1d.py            # data/大気温暖期.csv を解析
    python kde_mle_iterative_v1_1d.py 水質 底質   # 複数ファイルをまとめて解析

本モジュールはリファクタリング版であり、公開関数のシグネチャと数値結果は
従来版と同一に保っている。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.utils import resample
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

try:
    from module.module_kde_mle_2a import (
        run_mle_and_aic_iterative,
        run_mle_and_aic_simple,
    )
except ModuleNotFoundError:  # 別ディレクトリから実行された場合のフォールバック
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from module.module_kde_mle_2a import (
        run_mle_and_aic_iterative,
        run_mle_and_aic_simple,
    )


# =====================================================================
# 定数（マジックナンバーの一元管理）
# =====================================================================

LOG_FLOOR = 1e-10
"""対数尤度計算時の 0 除算・log(0) 回避用フロア。"""

BAD_FIT_PENALTY = 1e10
"""最適化が破綻した際に返すペナルティ値（実質 -inf 相当）。"""

ND_SUBSTITUTE_DIVISOR = np.sqrt(2.0)
"""ND 値の代替値 = 検出下限値 / sqrt(2)（環境省法）。"""

MIN_LOG_CONC_STD = 1e-6
"""log 濃度の標準偏差がこれ未満の物質は解析対象外（全 ND 等）。"""

MIN_SAMPLES = 24
"""1 物質あたりの最小サンプル数。これ未満はスキップ。"""

# --- 年度別残差 KDE（環境省 図 5）関連 ---
YEARLY_MIN_BW = 0.15
"""年度別 KDE のバンド幅の下限（データスケール）。ND 集中時のスパイク化を防ぐ。"""

YEARLY_STD_EPS = 1e-8
"""全残差の標準偏差がこれ未満なら年度別 KDE を構築しない。"""

YEARLY_FLAT_STD_EPS = 1e-4
"""年度内標準偏差がこれ未満を「平坦（ND ばかり）」とみなす閾値。"""

YEARLY_FLAT_JITTER = 1e-4
"""平坦とみなした年度に加える微小ノイズの振幅。"""

# --- 反復最尤推定 ---
DEFAULT_MAX_ITER = 50
"""反復 KDE-MLE のデフォルト最大反復回数。"""

DEFAULT_TOLERANCE = 1e-3
"""反復 KDE-MLE のデフォルト収束判定閾値。"""

MAIN_ITERATIVE_MAX_ITER = 30
"""メイン処理で iterative 版に渡す最大反復回数。"""

# --- 判定・検定 ---
P1_THRESHOLD = 0.95
"""AIC 事後確率 p1 がこれ以上なら「傾向あり」と判定する。"""

BOOTSTRAP_R = 10_000
"""ブートストラップのリサンプリング回数。"""

BOOTSTRAP_ALPHA_PERCENTILES = (2.5, 97.5)
"""ブートストラップ 95% 信頼区間のパーセンタイル。"""

# --- 入出力 ---
DEFAULT_FILENAMES: tuple[str, ...] = ("大気温暖期",)
"""引数未指定時に解析するファイル（拡張子なし）の既定値。"""

INPUT_COLUMNS: tuple[str, ...] = (
    "substance",
    "year",
    "site_id",
    "concentration",
    "detection_limit",
)
"""入力 CSV から読み込む必須カラム。"""

CSV_ENCODING = "utf-8-sig"
"""Excel での文字化けを避けるための出力エンコーディング。"""


# =====================================================================
# 汎用ユーティリティ
# =====================================================================

def clean_value_user(val: object) -> float:
    """測定値・検出下限値の文字列表現を float に正規化する.

    ``tr(...)`` の括弧や ``---``（欠測）を除去し、``nd`` / ``ND`` は 0 として
    扱う。変換できない場合は ``NaN`` を返す。

    Parameters
    ----------
    val : object
        元のセル値（文字列・数値・NaN など）。

    Returns
    -------
    float
        変換後の数値。失敗時は ``NaN``。
    """
    if isinstance(val, str):
        val = val.replace("tr(", "").replace(")", "")
        val = val.replace("---", "")
        val = val.replace("nd", "0")
        val = val.replace("ND", "0")
    try:
        if val == "":
            return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def format_ci(ci: object, digits: int = 4) -> list[float]:
    """信用区間を安全に丸めて 2 要素のリストに統一する.

    2 要素かつ有限な値であれば丸めたリストを、それ以外は
    ``[nan, nan]`` を返す。
    """
    try:
        arr = np.asarray(ci, dtype=float)
        if arr.size == 2 and not np.isnan(arr).any():
            return [round(float(arr[0]), digits), round(float(arr[1]), digits)]
    except Exception:
        pass
    return [np.nan, np.nan]


def format_float(val: object, digits: int = 4) -> float:
    """数値を安全に丸める（変換不能なら ``NaN``）。"""
    try:
        return round(float(val), digits)
    except Exception:
        return np.nan


def compute_aic_posterior(
    log_likelihood_linear: float,
    log_likelihood_constant: float,
    n_params_linear: int = 2,
    n_params_constant: int = 1,
) -> tuple[float, float, float]:
    """AIC と AIC 事後確率 p1 を計算する.

    Parameters
    ----------
    log_likelihood_linear : float
        1 次（線形）モデルの最大対数尤度。
    log_likelihood_constant : float
        0 次（定数）モデルの最大対数尤度。
    n_params_linear : int, optional
        線形モデルのパラメータ数（既定 2: 傾き・切片）。
    n_params_constant : int, optional
        定数モデルのパラメータ数（既定 1）。

    Returns
    -------
    tuple of (float, float, float)
        ``(aic0, aic1, p1)``。
    """
    aic1 = -2.0 * log_likelihood_linear + 2.0 * n_params_linear
    aic0 = -2.0 * log_likelihood_constant + 2.0 * n_params_constant
    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    p1 = numerator / (1.0 + numerator)
    return aic0, aic1, p1


# =====================================================================
# 環境省（図5）準拠：年度ごとの残差 KDE を合成するモデル
# =====================================================================

def build_yearly_kdes_old(residuals: np.ndarray, x_years: np.ndarray) -> list:
    """年度ごとに残差 KDE を作成する旧実装（互換性のため残置）.

    .. deprecated::
        バンド幅が年度ごとに独立に決まるため、ND 集中年度でスパイク化する。
        実運用では :func:`build_yearly_kdes` を用いること。
    """
    unique_years = np.unique(x_years)
    yearly_kdes = []
    for yr in unique_years:
        yr_res = residuals[x_years == yr]
        if len(yr_res) >= 2 and np.std(yr_res) > YEARLY_STD_EPS:
            try:
                yearly_kdes.append(gaussian_kde(yr_res))
            except np.linalg.LinAlgError:
                continue
    return yearly_kdes


def build_yearly_kdes(
    residuals: np.ndarray,
    x_years: np.ndarray,
    min_bw: float = YEARLY_MIN_BW,
) -> list:
    """年度ごとに残差の KDE オブジェクトを作成してリストで返す.

    年度ごとにバンド幅を独立に決めると、ND 値が集中した年度で KDE が
    スパイク化する。これを避けるため、全残差から求めた基準バンド幅
    ``target_bw_abs``（``min_bw`` を下限とする）を全年度に共通適用する。

    Parameters
    ----------
    residuals : array-like
        現在のパラメータでの残差。
    x_years : array-like
        各サンプルの年度（中心化済みで可）。
    min_bw : float, optional
        バンド幅の下限（データスケール）。

    Returns
    -------
    list
        ``scipy.stats.gaussian_kde`` のリスト。
        有効な年度が無い場合は空リスト。
    """
    unique_years = np.unique(x_years)
    yearly_kdes = []

    # 全体残差から基準となるバンド幅（標準偏差 × factor）を算出
    overall_std = np.std(residuals)
    if overall_std < YEARLY_STD_EPS:
        return []
    base_kde = gaussian_kde(residuals)
    # 全体のバンド幅（データスケール上の幅）
    target_bw_abs = max(base_kde.factor * overall_std, min_bw)

    for yr in unique_years:
        yr_res = residuals[x_years == yr]
        if len(yr_res) < 2:
            continue

        yr_std = np.std(yr_res)
        if yr_std < YEARLY_FLAT_STD_EPS:
            # 年度内のばらつきが極端に小さい（ND ばかり）場合は、
            # 擬似的に微小ノイズを加えて全体の幅に合わせる
            yr_res_jitter = yr_res + np.linspace(
                -YEARLY_FLAT_JITTER, YEARLY_FLAT_JITTER, len(yr_res)
            )
            bw_factor = target_bw_abs / np.std(yr_res_jitter)
            yearly_kdes.append(gaussian_kde(yr_res_jitter, bw_method=bw_factor))
        else:
            bw_factor = target_bw_abs / yr_std
            yearly_kdes.append(gaussian_kde(yr_res, bw_method=bw_factor))

    return yearly_kdes


def eval_pooled_yearly_pdf(residuals: np.ndarray, yearly_kdes: list) -> np.ndarray:
    """年度別 KDE の確率密度を等重みで合成（平均化）した PDF を返す.

    図 5 の「足し合わせた共通分布」に相当する。年度 KDE が空の場合は
    ``LOG_FLOOR`` で埋めた配列を返す。
    """
    if not yearly_kdes:
        return np.full_like(residuals, LOG_FLOOR, dtype=float)

    total_pdf = np.zeros_like(residuals, dtype=float)
    for kde in yearly_kdes:
        total_pdf += kde.pdf(residuals)

    return total_pdf / len(yearly_kdes)


def neg_log_likelihood_yearly_fixed_kde(
    params: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    yearly_kdes_fixed: list,
) -> float:
    """【線形モデル用・環境省図 5 準拠】固定 KDE リストによる負の対数尤度.

    Parameters
    ----------
    params : array-like, shape (2,)
        ``[a, b]``（傾き・切片）。
    x : array-like
        説明変数（中心化年度）。
    y : array-like
        応答変数（log 濃度）。
    yearly_kdes_fixed : list
        固定された年度別 KDE のリスト。
    """
    a, b = params
    residuals = y - (a * x + b)
    try:
        pdf_values = eval_pooled_yearly_pdf(residuals, yearly_kdes_fixed)
        log_likelihood = np.sum(np.log(pdf_values + LOG_FLOOR))
        return -log_likelihood
    except Exception:
        return BAD_FIT_PENALTY


def neg_log_likelihood_yearly_0(
    params: Sequence[float], x: np.ndarray, y: np.ndarray
) -> float:
    """【定数モデル用（傾き a=0）・環境省図 5 準拠】負の対数尤度.

    残差から都度、年度別 KDE を再構築する。
    """
    b0 = params[0]
    residuals = y - b0
    try:
        yearly_kdes = build_yearly_kdes(residuals, x)
        pdf_values = eval_pooled_yearly_pdf(residuals, yearly_kdes)
        log_likelihood = np.sum(np.log(pdf_values + LOG_FLOOR))
        return -log_likelihood
    except Exception:
        return BAD_FIT_PENALTY


# =====================================================================
# 目的関数群（KDE ベースの負の対数尤度）
# =====================================================================

def neg_log_likelihood_fixed_kde(
    params: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    kde_fixed: gaussian_kde,
) -> float:
    """【線形モデル用】固定された KDE を用いた負の対数尤度.

    反復最尤推定の各ステップで KDE を固定し、その下でパラメータを
    最適化するために用いる。
    """
    a, b = params
    residuals = y - (a * x + b)
    try:
        pdf_values = kde_fixed.pdf(residuals)
        log_likelihood = np.sum(np.log(pdf_values + LOG_FLOOR))
        return -log_likelihood
    except Exception:
        return BAD_FIT_PENALTY


def neg_log_likelihood(params: Sequence[float], x: np.ndarray, y: np.ndarray) -> float:
    """【線形モデル用】残差から KDE を作り直して負の対数尤度を計算する."""
    a, b = params
    residuals = y - (a * x + b)
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + LOG_FLOOR))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return BAD_FIT_PENALTY


def neg_log_likelihood_1(
    params: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
    initial_kde: object = None,
) -> float:
    """【互換用】``neg_log_likelihood`` の別名.

    .. note::
        ``initial_kde`` は受け取るが使用しない（従来版の挙動を維持）。
    """
    return neg_log_likelihood(params, x, y)


def neg_log_likelihood_0(
    params: Sequence[float], x: np.ndarray, y: np.ndarray
) -> float:
    """【定数モデル用】残差から KDE を作り直して負の対数尤度を計算する."""
    b0 = params[0]
    residuals = y - b0
    try:
        kde = gaussian_kde(residuals)
        log_likelihood = np.sum(np.log(kde.pdf(residuals) + LOG_FLOOR))
        return -log_likelihood
    except np.linalg.LinAlgError:
        return BAD_FIT_PENALTY


# =====================================================================
# 反復最尤推定（共通エンジン）
# =====================================================================

KdeObjectiveFactory = Callable[[np.ndarray], Callable[..., float]]
"""残差から目的関数を作るファクトリの型（引数: 残差）。"""


def _iterative_mle(
    x_centered: np.ndarray,
    y: np.ndarray,
    a0: float,
    b0: float,
    make_objective: KdeObjectiveFactory,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, float, float, str, int]:
    """KDE を固定 → 最適化 → 収束判定 を繰り返す共通ループ.

    ``run_mle_and_aic``（全体 KDE）と ``run_mle_and_aic_yearly_pooled``
    （年度別合成 KDE）で処理手順が完全に同一なため、目的関数の生成だけを
    差し替え可能な形で共通化している。

    Parameters
    ----------
    x_centered : array-like
        中心化済みの年度。
    y : array-like
        log 濃度。
    a0, b0 : float
        初期値（傾き・切片）。
    make_objective : callable
        残差を引数に取り、``f(params, x, y) -> float`` を返す関数を返す
        ファクトリ。
    max_iter : int, optional
        最大反復回数。
    tolerance : float, optional
        収束判定閾値（パラメータ変化量のユークリッドノルム）。

    Returns
    -------
    tuple
        ``(a1, b1, log_likelihood_1, converged_status, final_iter)``。
        ``log_likelihood_1`` は最後に実行した最適化の最大対数尤度。
    """
    a_current, b_current = a0, b0
    prev_a, prev_b = a_current, b_current

    converged_status = "未収束"
    final_iter = max_iter
    log_likelihood_1 = -np.inf

    for i in range(max_iter):
        # --- 現在のパラメータで残差を計算し、KDE を固定 ---
        residuals = y - (a_current * x_centered + b_current)
        objective = make_objective(residuals)

        # --- 固定 KDE を使った最適化 ---
        result = minimize(
            objective,
            [a_current, b_current],
            args=(x_centered, y),
            method="L-BFGS-B",
        )
        a_new, b_new = result.x
        log_likelihood_1 = -result.fun

        # --- 収束判定 ---
        param_change = np.linalg.norm([a_new - prev_a, b_new - prev_b])
        if param_change < tolerance:
            converged_status = "収束"
            final_iter = i + 1
            a1, b1 = a_new, b_new
            break

        # --- 次の iteration のために更新 ---
        prev_a, prev_b = a_current, b_current
        a_current, b_current = a_new, b_new
    else:
        # 最大回数到達
        a1, b1 = a_current, b_current
        converged_status = "最大回数到達"
        final_iter = max_iter

    return a1, b1, log_likelihood_1, converged_status, final_iter


def _make_global_kde_objective(residuals: np.ndarray) -> Callable[..., float]:
    """全体残差から単一 KDE を作り、固定 KDE の NLL を返すファクトリ."""
    kde_fixed = gaussian_kde(residuals)

    def objective(params: Sequence[float], x: np.ndarray, y: np.ndarray) -> float:
        return neg_log_likelihood_fixed_kde(params, x, y, kde_fixed)

    return objective


def _make_yearly_pooled_objective(
    x_centered: np.ndarray,
) -> KdeObjectiveFactory:
    """年度別合成 KDE の NLL を返すファクトリを生成する."""

    def factory(residuals: np.ndarray) -> Callable[..., float]:
        yearly_kdes_fixed = build_yearly_kdes(residuals, x_centered)

        def objective(params: Sequence[float], x: np.ndarray, y: np.ndarray) -> float:
            return neg_log_likelihood_yearly_fixed_kde(
                params, x, y, yearly_kdes_fixed
            )

        return objective

    return factory


def run_mle_and_aic(
    df: pd.DataFrame, max_iter: int = DEFAULT_MAX_ITER, tolerance: float = DEFAULT_TOLERANCE
) -> tuple:
    """全体 KDE を反復更新する KDE-MLE と AIC 判定.

    Returns
    -------
    tuple
        ``(a1, b1, aic0, aic1, p1, x_base, converged_status, final_iter)``。
    """
    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values

    # 初期値（OLS）
    a_current, b_current = np.polyfit(x_centered, y, 1)

    a1, b1, ll1, converged_status, final_iter = _iterative_mle(
        x_centered,
        y,
        a_current,
        b_current,
        _make_global_kde_objective,
        max_iter=max_iter,
        tolerance=tolerance,
    )

    # --- 定数モデル（0次式） ---
    result_0 = minimize(
        neg_log_likelihood_0,
        [np.mean(y)],
        args=(x_centered, y),
        method="L-BFGS-B",
    )
    ll0 = -result_0.fun

    aic0, aic1, p1 = compute_aic_posterior(ll1, ll0)
    return a1, b1, aic0, aic1, p1, x_base, converged_status, final_iter


def run_mle_and_aic_yearly_pooled(
    df: pd.DataFrame, max_iter: int = DEFAULT_MAX_ITER, tolerance: float = DEFAULT_TOLERANCE
) -> tuple:
    """環境省（図 5）準拠：年度別残差分布の合成による反復最尤推定と AIC 判定.

    Returns
    -------
    tuple
        ``(a1, b1, aic0, aic1, p1, x_base, converged_status, final_iter)``。
    """
    x_raw = df["year"].values
    x_base = x_raw.min()
    x_centered = x_raw - x_base
    y = df["log_conc"].values

    # 初期値（OLS）
    a_current, b_current = np.polyfit(x_centered, y, 1)

    a1, b1, ll1, converged_status, final_iter = _iterative_mle(
        x_centered,
        y,
        a_current,
        b_current,
        _make_yearly_pooled_objective(x_centered),
        max_iter=max_iter,
        tolerance=tolerance,
    )

    # --- 定数モデル（0次式） ---
    result_0 = minimize(
        neg_log_likelihood_yearly_0,
        [np.mean(y)],
        args=(x_centered, y),
        method="L-BFGS-B",
    )
    ll0 = -result_0.fun

    aic0, aic1, p1 = compute_aic_posterior(ll1, ll0)
    return a1, b1, aic0, aic1, p1, x_base, converged_status, final_iter


# =====================================================================
# ブートストラップ検定（前期 vs 後期）
# =====================================================================

def determine_split_year(df: pd.DataFrame) -> Optional[int]:
    """前期・後期を分割する年（年度の中央値）を返す.

    年度数が 4 未満の場合は ``None``（検定不能）。
    """
    years = df["year"].dropna().unique()
    if len(years) < 4:
        return None
    return int(np.median(years))


def compare_periods(
    df: pd.DataFrame, split_year: int = 2015, R: int = BOOTSTRAP_R
) -> np.ndarray:
    """前期・後期の平均値差をブートストラップで推定し 95% CI を返す.

    Parameters
    ----------
    df : DataFrame
        対象物質のデータ（``year`` と ``log_conc`` を含む）。
    split_year : int
        この年以下を前期、より後を後期とする。
    R : int
        リサンプリング回数。

    Returns
    -------
    ndarray
        ``[2.5, 97.5]`` パーセンタイル。前期・後期いずれかが 2 点未満なら
        ``[nan, nan]``。
    """
    A = df[df["year"] <= split_year]["log_conc"].values
    B = df[df["year"] > split_year]["log_conc"].values
    if len(A) < 2 or len(B) < 2:
        return np.array([np.nan, np.nan])

    diffs = []
    for _ in range(R):
        A_sample = resample(A, replace=True)
        B_sample = resample(B, replace=True)
        diffs.append(np.mean(A_sample) - np.mean(B_sample))

    return np.percentile(diffs, BOOTSTRAP_ALPHA_PERCENTILES)


# =====================================================================
# 二項ロジスティック回帰（ND 割合のトレンド）
# =====================================================================

def binomial_trend_analyze(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """ND（検出下限未満）割合の経年傾向を二項 GLM で評価する.

    全期間の最大検出下限値を統一基準とし、それを下回る地点を ND と
    みなして年ごとの ND / 非 ND 数を集計する。

    Returns
    -------
    tuple
        ``(aic0, aic1, p1, slope)``。
    """
    unified_dl = df["detection_limit"].max()
    df["low_flag"] = (df["concentration"] < unified_dl).astype(int)

    # 年ごとに「ND 数 (low_count)」と「ND でない数 (not_low_count)」を集計
    yearly_data = df.groupby("year").agg(
        low_count=("low_flag", "sum"),
        total_count=("low_flag", "count"),
    ).reset_index()

    yearly_data["not_low_count"] = yearly_data["total_count"] - yearly_data["low_count"]

    x_year = yearly_data["year"].astype(float).to_numpy()
    X1 = sm.add_constant(x_year).astype(float)

    # 応答変数を [ND数, 非ND数] の 2 列行列として作成
    y_binomial = yearly_data[["low_count", "not_low_count"]].to_numpy().astype(float)

    # 有効なデータ（全試行数が 0 でない行）のフィルタリング
    valid_mask = (yearly_data["total_count"] > 0) & (~np.isnan(X1).any(axis=1))
    X1 = X1[valid_mask]
    y_binomial = y_binomial[valid_mask]

    X0 = np.ones((len(y_binomial), 1), dtype=float)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", category=PerfectSeparationWarning)

        # freq_weights は使わず、y に 2 列の行列を渡す
        model1_fit = sm.GLM(y_binomial, X1, family=sm.families.Binomial()).fit()
        aic1 = model1_fit.aic

        model0_fit = sm.GLM(y_binomial, X0, family=sm.families.Binomial()).fit()
        aic0 = model0_fit.aic

    delta_aic = aic1 - aic0
    numerator = np.exp(-0.5 * delta_aic)
    p1 = numerator / (1 + numerator)
    slope = model1_fit.params[1] if len(model1_fit.params) > 1 else np.nan

    return aic0, aic1, p1, slope


# =====================================================================
# 前処理・1 物質ごとの解析（メイン処理の部品）
# =====================================================================

def load_and_preprocess(filepath: str, output_dir: str) -> tuple[pd.DataFrame, str]:
    """入力 CSV を読み込み、ND 補正・log 変換・欠損除去を行う.

    Parameters
    ----------
    filepath : str
        入力 CSV のパス。
    output_dir : str
        変換後データの出力先ディレクトリ。

    Returns
    -------
    tuple of (DataFrame, str)
        前処理済みデータフレームと、変換後 CSV の出力パス。
    """
    os.makedirs(output_dir, exist_ok=True)

    df_all = pd.read_csv(filepath, usecols=list(INPUT_COLUMNS))
    df_all["concentration"] = df_all["concentration"].apply(clean_value_user)
    df_all["detection_limit"] = df_all["detection_limit"].apply(clean_value_user)

    # ND（不検出）は検出下限値 / sqrt(2) で置換する
    is_nondetect = df_all["concentration"] <= df_all["detection_limit"]
    substitute_value = df_all["detection_limit"] / ND_SUBSTITUTE_DIVISOR
    df_all["concentration_sub"] = df_all["concentration"].copy()
    df_all.loc[is_nondetect, "concentration_sub"] = substitute_value[is_nondetect]

    df_all["log_conc"] = np.log10(df_all["concentration_sub"])
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log_conc", "detection_limit", "concentration"]
    )

    filename = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{filename}_conv.csv")
    df_all.to_csv(output_path, index=False, encoding=CSV_ENCODING)  # Excel 文字化け対策
    print(f"→ 変換後のデータを保存しました: {output_path}")

    return df_all, output_path


def analyze_substance(
    df: pd.DataFrame, substance: str, filename: str
) -> Optional[dict]:
    """1 物質について 5 つの解析を実行し、結果レコードを返す.

    Parameters
    ----------
    df : DataFrame
        当該物質のみに絞り込んだデータ（コピー済みであること）。
    substance : str
        物質名。
    filename : str
        ログ表示用のファイル名。

    Returns
    -------
    dict or None
        結果レコード。解析不能（データ不足・エラー）の場合は ``None``。
    """
    print(f"\n==================== {filename} : {substance} ====================")

    # ND のみのケースを念頭に置く。除外する
    if np.std(df["log_conc"]) < MIN_LOG_CONC_STD:
        return None

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    if len(df) < MIN_SAMPLES:
        print("データ不足のためスキップ（地点数 < 24）")
        return None

    # --- ① KDE-MLE（simple）: iterative の初期値も兼ねる ---
    print("\n=== Running KDE-MLE (simple) ===")
    a1, b1, aic0_1, aic1_1, p1_s, x_base = run_mle_and_aic_simple(df)

    print(f"Slope (a): {a1:.4f}")
    print(f"Intercept (b): {b1:.4f}")
    print(f"AIC0 (constant): {aic0_1:.3f}")
    print(f"AIC1 (linear):   {aic1_1:.3f}")
    print(f"AIC posterior weight p1: {p1_s:.4f}")
    trend_s = (
        f"p1 >= {P1_THRESHOLD} 傾向あり"
        if p1_s >= P1_THRESHOLD
        else f"p1 < {P1_THRESHOLD} 傾向なし"
    )
    print(f"→ {trend_s}")

    # --- ② KDE-MLE（iterative） ---
    print("\n=== Running KDE-MLE (iterative) ===")
    try:
        a, b, aic0, aic1, p1, x_base, mle_status, mle_iters, tolerance = (
            run_mle_and_aic_iterative(df, a1, b1, MAIN_ITERATIVE_MAX_ITER)
        )

        print("--- 1. 対数線形回帰モデル (MLE) ---")
        print(f"状態: {mle_status} tolerance: {tolerance:.04f} (反復回数: {mle_iters})")
        print(f"傾き a = {a:.4f}, 切片 b = {b:.4f}, AIC事後確率 p1 = {p1:.4f}")
        print(f"aic1: {aic1} - aic0: {aic0}")

        trend = (
            f"p1 >= {P1_THRESHOLD} 傾向あり"
            if p1 >= P1_THRESHOLD
            else f"p1 < {P1_THRESHOLD} 傾向なし"
        )
        print(f"→ {trend}")

        print("OK　判定は一致しています" if trend == trend_s else "NO! 判定が不一致です!")
        if trend != trend_s:
            trend += f"　simple版とは異なる結果です ({p1_s:.4f})"
        else:
            trend += f"　simple版と同じ結果です ({p1_s:.4f})"
    except Exception as e:
        print(f"最尤推定エラー: {e}")
        return None

    # --- ③ 環境省 図 5 準拠（年度別残差合成 KDE） ---
    print("\n=== Running KDE-MLE (MOE Fig.5 Yearly Pooled) ===")
    try:
        a_moe, b_moe, _, _, p1_moe, _, status_moe, iters_moe = (
            run_mle_and_aic_yearly_pooled(df)
        )
        trend_moe = (
            f"p1 >= {P1_THRESHOLD} 傾向あり"
            if p1_moe >= P1_THRESHOLD
            else f"p1 < {P1_THRESHOLD} 傾向なし"
        )
        print(f"状態: {status_moe} (反復回数: {iters_moe})")
        print(
            f"傾き a = {a_moe:.4f}, 切片 b = {b_moe:.4f}, "
            f"AIC事後確率 p1 = {p1_moe:.4f} → {trend_moe}"
        )
    except Exception as e:
        print(f"年度別合成KDEエラー: {e}")

    # --- ④ ブートストラップ検定 ---
    ci_bs = np.array([np.nan, np.nan])
    try:
        split_year = determine_split_year(df)
        if split_year is None:
            bs_trend = "有意差なし"
        else:
            ci_bs = compare_periods(df, split_year=split_year)
            print("\n--- 3. ブートストラップ検定 ---")
            print(f"分割年: {split_year}")
            print(f"平均値差の95%信頼区間: {ci_bs}")
            if ci_bs[0] > 0 and not np.isnan(ci_bs[0]):
                bs_trend = "後期は有意に低濃度"
            elif ci_bs[1] < 0 and not np.isnan(ci_bs[1]):
                bs_trend = "後期は有意に高濃度"
            else:
                bs_trend = "有意差なし"
            print(f"→ {bs_trend}")
    except Exception as e:
        print(f"ブートストラップエラー: {e}")
        bs_trend = "エラー"
        ci_bs = np.array([np.nan, np.nan])

    # --- ⑤ 二項ロジスティック回帰 ---
    a_bin, p1_bin = np.nan, np.nan
    try:
        print("\n--- 4. 二項ロジスティック回帰モデル ---")
        df_binomial = df[["year", "concentration", "detection_limit"]].copy()
        aic0_bin, aic1_bin, p1_bin, a_bin = binomial_trend_analyze(df_binomial)
        print(
            f"傾き a = {a_bin:.4f}, オッズ比 ≈ {math.exp(a_bin):.4f}, "
            f"AIC事後確率 p1 = {p1_bin:.3f}"
        )
        trend_bin = "傾向あり" if p1_bin >= P1_THRESHOLD else "傾向なし"
        print(f"→ {trend_bin}")
    except Exception as e:
        print(f"二項回帰エラー: {e}")
        trend_bin = "エラー"
        a_bin, p1_bin = np.nan, np.nan

    return {
        "substance": substance,
        "year": df["year"].unique().tolist(),
        "slope_MLE": a,
        "intercept_MLE": b,
        "p1_MLE": p1,
        "trend_MLE": trend,
        "mle_status": mle_status,
        "mle_iters": mle_iters,
        "tolerance": tolerance,
        "bootstrap_CI": ci_bs,
        "bs_trend": bs_trend,
        "binomial_slope": a_bin,
        "binomial_p1": p1_bin,
        "binomial_trend": trend_bin,
    }


def format_results(results: list[dict]) -> pd.DataFrame:
    """結果レコード群を CSV 出力用に整形して DataFrame にする."""
    for r in results:
        # 信用区間の整形（すべて format_ci に統一）
        r["bootstrap_CI"] = format_ci(r.get("bootstrap_CI"))

        # 数値項目の整形
        for key in (
            "slope_MLE",
            "intercept_MLE",
            "binomial_slope",
            "p1_MLE",
            "binomial_p1",
        ):
            r[key] = format_float(r.get(key))

    return pd.DataFrame(results)


def run_analysis_for_file(
    filename: str, data_dir: str = "data", output_dir: str = "outputs"
) -> Optional[pd.DataFrame]:
    """1 つの入力 CSV について、全物質の解析を実行し結果を保存する.

    Returns
    -------
    DataFrame or None
        結果テーブル。入力ファイルが無い場合は ``None``。
    """
    print(f"\n\n==================== ファイル処理開始: {filename} ====================")
    filepath = os.path.join(data_dir, f"{filename}.csv")
    if not os.path.exists(filepath):
        print(f"ファイルが見つかりません: {filepath}")
        return None

    df_all, _ = load_and_preprocess(filepath, output_dir)

    substances = df_all["substance"].unique()
    print(f"対象物質: {substances}")

    results: list[dict] = []
    for substance in substances:
        df_sub = df_all[df_all["substance"] == substance].copy()
        record = analyze_substance(df_sub, substance, filename)
        if record is not None:
            results.append(record)

    df_results = format_results(results)
    print(f"\n--- 最終結果（テーブル形式）  {filename} ---")
    print(df_results.to_string(index=False))

    result_path = os.path.join(output_dir, f"{filename}_2025_moe.csv")
    df_results.to_csv(result_path, index=False, encoding=CSV_ENCODING)
    return df_results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """コマンドライン引数を解析する."""
    parser = argparse.ArgumentParser(
        description="KDE-MLE による POPs 年次トレンド解析（環境省法）",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        default=list(DEFAULT_FILENAMES),
        help=f"解析するファイル名（拡張子なし）。既定: {' '.join(DEFAULT_FILENAMES)}",
    )
    parser.add_argument(
        "--data-dir", default="data", help="入力 CSV のディレクトリ（既定: data）"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="出力 CSV のディレクトリ（既定: outputs）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """エントリポイント。指定された各ファイルを順に解析する."""
    args = parse_args(argv)
    for filename in args.filenames:
        run_analysis_for_file(filename, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
