# mymodel_numpyro.py
# NumPyro / NUTS implementation for your hierarchical Student-t regression (non-centered)
# Requires: numpy, pandas, jax, jaxlib, numpyro, matplotlib (optional)
# Example usage at bottom.

import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
import csv

# ------------------------------
# フォント設定（起動時に一度）
# ------------------------------
plt.rcParams['font.size'] = 10
# japanize_matplotlib を import 済みなので、追加設定は不要

#　入力設定
filenames = ["水質"]  # data/<name>.csv
substance_data = ["総PCB"] #「リストに中身があれば」選択、[]で「すべての物質」の意味

# 出力設定
PLOT_DIR = ""     #　傾きのバーグラフのみ分離して1か所に集めたいときに指定する　空 ""の設定で　{filemane_substance}　
SAVE_NPZ = False
CSV_PATH ="_site_summary.csv"

# データ数による除外 ---
MIN_OBS_FOR_SITE = 5   # Total 観測数（観測年）# サイトごとの最小データ列（例:　11以上を残す）
# 最終年度による除外
MIN_END_YEAR = 2010  # 調査最終年度 # 最終観測年がこの年以下の場合に除外 (例:　2016年以上のサイトを残す)
#　解析期間の範囲指定
MIN_ANALYSIS_YEAR = 1999  # この年より大きい（指定した年は含まない）データを採用
MAX_ANALYSIS_YEAR = 2030  # この年以下（指定年まで含む）データを採用

#PLOTの出力
MAKE_PLOTS = True
MAKE_FACET_PLOTS = True  # Falseで図は描かない
MAKE_PRED_PLOTS = False  # Falseで図は描かない
sid_target="44"
year_target= 2000
MAKE_SITE_PLOTS = False  #['1', '30']   診断図を出力したいサイトIDを指定できる Falseで全指定
MAKE_TRACE_PLOT = False  #['1', '30']   診断図を出力したいサイトIDを指定できる Falseで全指定

DPI_PLOT = 150          # 図の解像度（200→150で保存高速化）

def original_scale_from_samples_all(a_samps_all: np.ndarray, b_samps_all: np.ndarray, x_mean: float, x_scale: float):
    a_o_all = a_samps_all / x_scale
    b_o_all = b_samps_all - a_o_all * x_mean
    return a_o_all, b_o_all

# ------------------------------
# データクリーニング
# ------------------------------

def clean_value_user(val):
    if isinstance(val, str):
        val = val.replace('tr(', '').replace(')', '')
        val = val.replace('---', '')
        val = val.replace('nd', '0')
    try:
        if val == '':
            return np.nan
        return float(val)
    except (ValueError, TypeError):
        return np.nan

# ---------- helper: prepare data ----------
def prepare_data_from_dataframe(df,
                                x_col="year",
                                y_col="log_conc",
                                site_col="site_id",
                                min_records_per_site=2):
    df2 = df[[x_col, y_col, site_col]].dropna()
    df2[site_col] = df2[site_col].astype(str)

    # サイトの出現回数を計算
    counts = df2[site_col].value_counts()
    valid_sites = set(counts[counts >= min_records_per_site].index)

    # フィルタリング
    df2 = df2[df2[site_col].isin(valid_sites)]

    # サイトリストとマッピング
    site_list = sorted(valid_sites)
    site_map = {sid: i for i, sid in enumerate(site_list)}
    site_idx = df2[site_col].map(site_map).to_numpy(dtype=np.int32)

    # x, y の抽出
    x = df2[x_col].to_numpy(dtype=float)
    y = df2[y_col].to_numpy(dtype=float)

    # 標準化
    x_mean = float(x.mean())
    x_std_val = float(x.std())
    x_scale = x_std_val if x_std_val > 0 else 1.0
    x_std = ((x - x_mean) / x_scale).astype(np.float32)

    return {
        "x_std": x_std,
        "y": y.astype(np.float32),
        "site_idx": site_idx,
        "n_sites": len(site_list),
        "x_mean": x_mean,
        "x_scale": x_scale,
        "site_ids": site_list,
        "df": df2.reset_index(drop=True),
    }
def prepare_data_from_dataframe_slow(df,
                                x_col="year",
                                y_col="log_conc",
                                site_col="site_id",
                                min_records_per_site=2):
    """
    - df: pandas DataFrame containing x_col, y_col, site_col.
    - Returns dict with x_std, y, site_idx, n_sites, x_mean, x_scale, site_ids_sorted
    """
    df2 = df[[x_col, y_col, site_col]].dropna().copy()
    # ensure site ids are strings and sorted deterministically
    df2[site_col] = df2[site_col].astype(str)
    site_list = sorted(df2[site_col].unique(), key=lambda s: s)
    # optionally filter sites with too few records
    counts = df2.groupby(site_col).size()
    valid_sites = counts[counts >= min_records_per_site].index.tolist()
    df2 = df2[df2[site_col].isin(valid_sites)].copy()
    site_list = [s for s in site_list if s in valid_sites]
    site_map = {sid: i for i, sid in enumerate(site_list)}
    site_idx = df2[site_col].map(site_map).values.astype(np.int32)

    x = df2[x_col].astype(float).values
    y = df2[y_col].astype(float).values

    # standardize x (same approach as your existing code)
    x_mean = float(x.mean())
    x_scale = float(x.std()) if x.std() > 0 else 1.0
    x_std = (x - x_mean) / x_scale

    return {
        "x_std": x_std.astype(np.float32),
        "y": y.astype(np.float32),
        "site_idx": site_idx.astype(np.int32),
        "n_sites": len(site_list),
        "x_mean": x_mean,
        "x_scale": x_scale,
        "site_ids": site_list,
        "df": df2.reset_index(drop=True),
    }

#---------- model ----------非中心化階層回帰＋t分布尤度。シンプルに「df=4のt分布」
def hier_regression_model(x_data, y_data, site_idx, n_sites,
                          mu_a_sd=0.5, mu_b_sd=5.0,
                          sigma_a_scale=0.05, sigma_b_scale=0.2,
                          sigma_obs_scale=0.2):

    # Hyperpriors　- mu_a, mu_b は切片・傾きの「全体平均」
    mu_a = numpyro.sample("mu_a", dist.Normal(0.0, mu_a_sd))
    mu_b = numpyro.sample("mu_b", dist.Normal(0.0, mu_b_sd))

    #- sigma_a, sigma_b はサイトごとの切片・傾きのばらつき。- sigma は観測誤差のスケール
    sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(scale=sigma_a_scale))
    sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(scale=sigma_b_scale))
    sigma   = numpyro.sample("sigma",   dist.HalfCauchy(scale=sigma_obs_scale))

    # Non-centered parameterization。- 各サイトごとの切片 a と傾き b を直接サンプリングではなく、標準正規乱数 a_raw, b_raw を使って「平均＋スケール×乱数」で生成しています。

    a_raw = numpyro.sample("a_raw", dist.Normal(jnp.zeros(n_sites), 1.0))
    b_raw = numpyro.sample("b_raw", dist.Normal(jnp.zeros(n_sites), 1.0))

    # Convert
    a = mu_a + sigma_a * a_raw
    b = mu_b + sigma_b * b_raw

    # Linear predictor　- サイトごとに異なる回帰直線を持ち、説明変数 x_data に応じて予測値 mu を計算。
    mu = a[site_idx] * x_data + b[site_idx]

    # Likelihood- 観測値 y_data は 自由度4のスチューデントt分布に従うと仮定。
    numpyro.sample("obs", dist.StudentT(df=4.0, loc=mu, scale=sigma), obs=y_data)
    #「誤差分布を正規分布にしたい」なら、
    #numpyro.sample("obs", dist.Normal(loc=mu, scale=sigma), obs=y_data)

# ---------- model ----------より大きめの値に設定 () → ばらつきを広めに許容
def hierarchical_student_t_noncentered(x, y, site_idx, n_sites,
                                       mu_a_sd=0.5, mu_b_sd=5.0,
                                       sigma_a_scale=0.3, sigma_b_scale=0.5, 
                                       sigma_obs_scale=0.5):
    """
    Non-centered hierarchical Student-t regression.
    x: (N,) standardized predictor (float32)
    y: (N,) response (float32)  - observed log10(conc) etc.
    site_idx: (N,) integer indices 0..n_sites-1
    n_sites: int
    Hyperparameter defaults can be adjusted.
    """
    # hyper-priors
    mu_a = numpyro.sample("mu_a", dist.Normal(0.0, mu_a_sd))   # typical slope scale small
    mu_b = numpyro.sample("mu_b", dist.Normal(0.0, mu_b_sd))   # intercept scale larger

    sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(sigma_a_scale))
    sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(sigma_b_scale))
    sigma   = numpyro.sample("sigma",   dist.HalfCauchy(sigma_obs_scale))

    # degrees of freedom for Student-t: >1
    # use 1 + Exponential to keep it >1 but allow heavy tails 
    # 推定値 nu = 1 + Exponential(0.1) → 平均約 11、柔軟に裾の厚さを調整可能
    nu = 1.0 + numpyro.sample("nu_excess", dist.Exponential(0.1))  # mean 10, adjust if you want less heavy tails

    # non-centered local parameters
    a_raw = numpyro.sample("a_raw", dist.Normal(jnp.zeros(n_sites), jnp.ones(n_sites)))
    b_raw = numpyro.sample("b_raw", dist.Normal(jnp.zeros(n_sites), jnp.ones(n_sites)))

    a = mu_a + sigma_a * a_raw   # slope in standardized-x scale
    b = mu_b + sigma_b * b_raw   # intercept in standardized-x scale

    # ★ deterministic に保存する
    numpyro.deterministic("a", a)
    numpyro.deterministic("b", b)


    mu = a[site_idx] * x + b[site_idx]

    # Student-t likelihood (robust)
    numpyro.sample("obs", dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)



def run_numpyro_mcmc(prepared_data, model=hier_regression_model,
                     num_warmup=2000, num_samples=2000,
                     num_chains=4, rng_seed=0, progress_bar=True, **model_kwargs):

    x = prepared_data["x_std"]
    y = prepared_data["y"]
    site_idx = prepared_data["site_idx"]
    n_sites = prepared_data["n_sites"]
    site_ids = prepared_data["site_ids"]

    # モデルを切り替え可能にする
    kernel = NUTS(lambda x_data, y_data, site_idx, n_sites:
                  model(x_data, y_data, site_idx, n_sites, **model_kwargs),
                  target_accept_prob=0.8)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=progress_bar)

    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key, x_data=x, y_data=y, site_idx=site_idx, n_sites=n_sites)

    samples = mcmc.get_samples()
    df_summary = extract_site_summary(samples, site_ids)

    return {
        "samples": samples,
        "df_summary": df_summary,
        "mu_a_mean": float(np.mean(samples["mu_a"])),
        "mu_a_ci": list(np.quantile(samples["mu_a"], [0.025, 0.975])),
        "sigma_a_mean": float(np.mean(samples["sigma_a"])),
        "sigma_mean": float(np.mean(samples["sigma"])),
        "site_summary": df_summary,
        "prepared": prepared_data
    }

# ---------- posterior predictive ----------
def posterior_predictive_intervals(result, x_grid=None, quantiles=(2.5, 50, 97.5), n_samples_predict=None, rng_seed=42):
    """
    result: output of run_numpyro_mcmc (expects keys: 'samples' and 'prepared')
    Returns: dict keyed by site_id with keys 'x' and 'quantiles' (quantiles -> array per quantile)
    """
    from numpyro.infer import Predictive
    import jax.random as random

    # samples as returned by run_numpyro_mcmc
    samples = result.get("samples") or result.get("raw_samples")
    if samples is None:
        raise KeyError("result contains no 'samples' or 'raw_samples' key. check run_numpyro_mcmc return value.")

    prepared = result.get("prepared") or result.get("prepared_data")
    if prepared is None:
        raise KeyError("result contains no 'prepared' or 'prepared_data' key. check run_numpyro_mcmc return value.")

    site_ids = prepared["site_ids"]
    x_mean = prepared["x_mean"]
    x_scale = prepared["x_scale"]
    df_prepared = prepared["df"]
    n_sites = prepared["n_sites"]

    # pick x grid covering observed years
    years = df_prepared["year"].values
    if x_grid is None:
        x_grid = np.linspace(years.min(), years.max(), 25)
    # standardize grid
    x_grid_std = (x_grid - x_mean) / x_scale
    x_grid_std = x_grid_std.astype(np.float32)

    # If user requested a subset of posterior draws for predict, do so
    # samples are arrays with shape (n_draws, ...)
    if n_samples_predict is not None:
        # uniformly subsample indices
        total_draws = next(iter(samples.values())).shape[0]
        n_samples_predict = min(n_samples_predict, total_draws)
        idx = np.random.choice(total_draws, size=n_samples_predict, replace=False)
        samples_sub = {k: v[idx] for k, v in samples.items()}
    else:
        samples_sub = samples

    predictive = Predictive(hier_regression_model, samples_sub)

    preds = {}
    key = random.PRNGKey(rng_seed)
    for i, sid in enumerate(site_ids):
        # site index repeated for each x_grid point
        site_idx_grid = np.full(shape=x_grid_std.shape, fill_value=i, dtype=np.int32)

        out = predictive(key, x_data=x_grid_std, y_data=None, site_idx=site_idx_grid, n_sites=n_sites)
        # out['obs'] shape: (n_draws, len(x_grid))
        obs = np.array(out["obs"])  # convert to numpy

        qs = np.percentile(obs, quantiles, axis=0)  # shape (len(quantiles), len(x_grid))
        # Build dict of quantiles with readable keys
        quantile_dict = {str(q): qs[k] for k, q in enumerate(quantiles)}

        preds[sid] = {
            "x": x_grid,               # original year grid (not standardized)
            "quantiles": quantile_dict # quantile -> array over x_grid
        }
    return preds




def summarize_parameters(result):
    samples = result["samples"]
    prepared = result["prepared"]

    # --- 全体要約 ---
    summary_global = []
    r2_summary = bayesian_r2(result, prepared)
    summary_global.append({
        "parameter": "Bayesian R^2",
        "mean": r2_summary["mean"],
        "ci_lower": r2_summary["ci_lower"],
        "ci_upper": r2_summary["ci_upper"]
    })
    for key in ["mu_a", "mu_b", "sigma", "sigma_a", "sigma_b", "nu_excess"]:
        if key in samples:
            vals = np.array(samples[key])
            summary_global.append({
                "parameter": key,
                "mean": vals.mean(),
                "sd": vals.std(),
                "ci_lower": np.percentile(vals, 2.5),
                "ci_upper": np.percentile(vals, 97.5),
                "posterior_prob_gt0": float(np.mean(vals > 0)),
                "posterior_prob_lt0": float(np.mean(vals < 0))

            })

    # --- サイト別パラメータ ---
    summary_sites = []
    site_ids = prepared["site_ids"]
    a_samples = np.array(samples["a"])
    b_samples = np.array(samples["b"])
    for j, sid in enumerate(site_ids):
        vals_a = a_samples[:, j]
        vals_b = b_samples[:, j]

        summary_sites.append({
            "site_id": sid,
            "parameter": "a",
            "mean": a_samples[:, j].mean(),
            "sd": a_samples[:, j].std(),
            "ci_lower": np.percentile(a_samples[:, j], 2.5),
            "ci_upper": np.percentile(a_samples[:, j], 97.5),
            #"ci_upper": np.percentile(vals_a, 97.5),
            "posterior_prob_gt0": float(np.mean(vals_a > 0)),
            "posterior_prob_lt0": float(np.mean(vals_a < 0))

        })
        summary_sites.append({
            "site_id": sid,
            "parameter": "b",
            "mean": b_samples[:, j].mean(),
            "sd": b_samples[:, j].std(),
            "ci_lower": np.percentile(b_samples[:, j], 2.5),
            "ci_upper": np.percentile(b_samples[:, j], 97.5),
            "posterior_prob_gt0": float(np.mean(vals_b > 0)),
            "posterior_prob_lt0": float(np.mean(vals_b < 0))

        })

    # --- サイト別 R² ---
    r2_site_summary = bayesian_r2_by_site(result, prepared)
    summary_r2_sites = []
    for sid, summary in r2_site_summary.items():
        summary_r2_sites.append({
            "site_id": sid,
            #"parameter": "Bayesian R^2",
            "mean": summary["mean"],
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"]
        })

    return (
        pd.DataFrame(summary_global),
        pd.DataFrame(summary_sites),
        pd.DataFrame(summary_r2_sites)
    )

def extract_site_summary(samples, site_ids):
    """
    samples: mcmc samples dict
    site_ids: list of site names in order
    """
    rows = []

    # グローバルパラメータ（mu_a, mu_b, sigma など）
    for name, values in samples.items():
        # a_raw, b_raw 等の階層パラメータ（長いベクトル）は除外
        if values.ndim == 1:
            v = values
            mean = np.mean(v)
            sd = np.std(v)
            ci_lower, ci_upper = np.percentile(v, [2.5, 97.5])
            prob_gt0 = np.mean(v > 0)

            rows.append({
                "parameter": name,
                "mean": mean,
                "sd": sd,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "posterior_prob_gt0": prob_gt0
            })

    # 各サイトの a[i], b[i]
    if "a" in samples and "b" in samples:
        a_samples = samples["a"]   # shape (draws, n_sites)
        b_samples = samples["b"]

        for i, site in enumerate(site_ids):
            a_i = a_samples[:, i]
            b_i = b_samples[:, i]

            # slope
            rows.append({
                "parameter": f"a[{site}]",
                "mean": np.mean(a_i),
                "sd": np.std(a_i),
                "ci_lower": np.percentile(a_i, 2.5),
                "ci_upper": np.percentile(a_i, 97.5),
                "posterior_prob_gt0": np.mean(a_i > 0)
            })

            # intercept
            rows.append({
                "parameter": f"b[{site}]",
                "mean": np.mean(b_i),
                "sd": np.std(b_i),
                "ci_lower": np.percentile(b_i, 2.5),
                "ci_upper": np.percentile(b_i, 97.5),
                "posterior_prob_gt0": np.mean(b_i > 0)
            })

    return pd.DataFrame(rows)

import matplotlib.pyplot as plt
import math


def plot_posterior_predictive_facet(preds, result, prepared):
    # 中央値と平均値のplot 
    #preds = posterior_predictive_intervals(result)
    site_ids = sorted(prepared["site_ids"],key=int)
    #site_ids = prepared["site_ids"]
    df_obs = prepared["df"]

    n_sites = len(site_ids)
    ncols = 4
    nrows = math.ceil(n_sites / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=False)
    axes = axes.flatten()

    for ax, sid in zip(axes, site_ids):
        pr = preds[str(sid)]
        x_plot = pr["x"]

        # quantiles を float key に変換
        q_raw = pr["quantiles"]
        q = {float(k): v for k, v in q_raw.items()}

        # 95% CI
        ax.fill_between(x_plot, q[2.5], q[97.5], alpha=0.25)

        # 中央値線
        ax.plot(x_plot, q[50.0], linewidth=2, label="median")

        # 平均線を result["samples"] から再計算
        site_idx = prepared["site_ids"].index(sid)
        x_std = (x_plot - prepared["x_mean"]) / prepared["x_scale"]
        a_samples = result["samples"]["a"][:, site_idx]
        b_samples = result["samples"]["b"][:, site_idx]
        y_samples = np.outer(a_samples, x_std) + b_samples[:, None]
        mean_line = y_samples.mean(axis=0)

        ax.plot(x_plot, mean_line, linestyle="--", color="red", label="mean")

        # 観測点
        df_site = df_obs[df_obs["site_id"] == sid]
        ax.scatter(df_site["year"], df_site["log_conc"], s=10, c="black")

        ax.set_title(f"Site {sid}")
        ax.set_xlabel("Year")
        ax.set_ylabel("log(conc)")
        ax.legend()

    for j in range(len(site_ids), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return fig



def bayesian_r2(result, prepared):
    y_obs = prepared["df"]["log_conc"].values
    x_obs = prepared["df"]["year"].values
    x_std = (x_obs - prepared["x_mean"]) / prepared["x_scale"]
    site_ids = prepared["df"]["site_id"].values

    # JAX → NumPy に変換
    a_samples = np.array(result["samples"]["a"])
    b_samples = np.array(result["samples"]["b"])

    r2_samples = []

    for draw in range(a_samples.shape[0]):
        y_hat = np.zeros_like(y_obs)
        for i, sid in enumerate(site_ids):
            site_idx = prepared["site_ids"].index(sid)
            y_hat[i] = a_samples[draw, site_idx] * x_std[i] + b_samples[draw, site_idx]

        resid = y_obs - y_hat
        var_yhat = np.var(y_hat)
        var_resid = np.var(resid)
        r2 = var_yhat / (var_yhat + var_resid)
        r2_samples.append(r2)

    return {
        "mean": np.mean(r2_samples),
        "median": np.median(r2_samples),
        "ci_lower": np.percentile(r2_samples, 2.5),
        "ci_upper": np.percentile(r2_samples, 97.5)
    }


def bayesian_r2_by_site(result, prepared):
    df = prepared["df"]
    #site_ids = prepared["site_ids"]
    site_ids = sorted(prepared["site_ids"],key=int)

    a_samples = np.array(result["samples"]["a"])  # shape (n_draws, n_sites)
    b_samples = np.array(result["samples"]["b"])

    r2_by_site = {}

    for j, sid in enumerate(site_ids):
        df_site = df[df["site_id"] == sid]
        x_obs = df_site["year"].values
        y_obs = df_site["log_conc"].values
        x_std = (x_obs - prepared["x_mean"]) / prepared["x_scale"]

        r2_samples = []
        for draw in range(a_samples.shape[0]):
            y_hat = a_samples[draw, j] * x_std + b_samples[draw, j]
            resid = y_obs - y_hat
            var_yhat = np.var(y_hat)
            var_resid = np.var(resid)
            r2 = var_yhat / (var_yhat + var_resid)
            r2_samples.append(r2)

        r2_by_site[sid] = {
            "mean": np.mean(r2_samples),
            "median": np.median(r2_samples),
            "ci_lower": np.percentile(r2_samples, 2.5),
            "ci_upper": np.percentile(r2_samples, 97.5)
        }

    return r2_by_site


def main():

    for filename in filenames:
        print(f"\n\n==================== ファイル処理開始: {filename} ====================")
        filepath = f"data/{filename}.csv"
        if not os.path.exists(filepath):
            print(f"ファイルが見つかりません: {filepath}")
            continue

        # 読み込み＋クリーニング
        df_all = pd.read_csv(
            filepath,
            usecols=['substance', 'year', 'site_id', 'concentration', 'detection_limit'],
            dtype={'substance': str, 'site_id': str}
        )
        df_all['concentration'] = df_all['concentration'].apply(clean_value_user)
        df_all['detection_limit'] = df_all['detection_limit'].apply(clean_value_user)

        # ND 代入
        is_nondetect = (df_all['concentration'] <= df_all['detection_limit']) & (~df_all['detection_limit'].isna())
        substitute_value = df_all['detection_limit'] / np.sqrt(2)
        df_all['concentration_sub'] = df_all['concentration'].copy()
        df_all.loc[is_nondetect, 'concentration_sub'] = substitute_value[is_nondetect]

        df_all['log_conc'] = np.log10(df_all['concentration_sub'].replace(0, np.nan))
        df_all = df_all.replace([np.inf, -np.inf], np.nan)
        df_all = df_all.dropna(subset=['log_conc', 'detection_limit', 'concentration'])


        # === ここで「グローバル除外」 ===
        min_samples = 2  # 2以下を除外 → 少なくとも 2以上を残す
        counts_all = df_all.groupby('site_id').size()               # site ごとの総観測数
        valid_sites_all = counts_all[counts_all >= min_samples].index
        removed_sites = set(df_all['site_id'].unique()) - set(valid_sites_all)

        df_all = df_all[df_all['site_id'].isin(valid_sites_all)].copy()

        print(f"ファイル名: {filename} 内で観測数（全データ数） {min_samples-1} 以下で除外された site: {sorted(list(removed_sites))}")
        print(f"残存 site 数: {df_all['site_id'].nunique()} / 総レコード数: {len(df_all)}")


        # 「if substance_data:」だけで「リストに中身があれば」という意味になります
        if substance_data:
            substances = substance_data 
        else:
            substances = df_all['substance'].unique()
        print(f"対象物質: {substances}")

        for substance in substances:
            print(f"\n==================== {substance} ====================")
            df = df_all[df_all['substance'] == substance].copy()

            #"""-----------
            # --- データ準備　パラメータ定義 ---
            # サイトごとの最小観測数（11以上を残す）
            # --------------------

            # site 観測数チェック (countsは pandas.Series, インデックスが site_id)
            counts = df.groupby("site_id").size()

            # 観測数が MIN_OBS_FOR_SITE 以上のサイトIDのみを取得
            valid_sites = counts[counts > MIN_OBS_FOR_SITE].index 

            # 除外されたサイトIDを特定 (診断用)
            removed_sites_local = set(df['site_id'].unique()) - set(valid_sites)

            # 除外処理: データフレーム df をフィルタリング
            df = df[df["site_id"].isin(valid_sites)].copy()

            # 診断結果の出力
            print(f"  > {substance}のデータ数が {MIN_OBS_FOR_SITE}以下のサイトを除外しました。")
            print(f"    - 除外されたサイト: {removed_sites_local}")
            print(f"    - 除外されたサイト数: {len(removed_sites_local)}")
            print(f"    - 残存サイト数: {df['site_id'].nunique()} / 総レコード数: {len(df)}")
            
            #------------------"""
            # --- データ準備　 パラメータ定義 ---
            #MIN_END_YEAR = 2020   # 最終観測年がこの年以下の場合に除外 (2016年以上のサイトを残す)
            # --------------------

            # site ごとの年データの最大値 (最終観測年) を取得
            g_year = df.groupby("site_id")["year"]
            df_year_info = pd.DataFrame({
                "site_id": g_year.apply(lambda s: s.name),
                "year_end": g_year.max(),
            }).reset_index(drop=True)
            # 最終観測年が MIN_END_YEAR より大きいサイトIDのみを取得 (例: 2016年以上のサイト)
            valid_sites_by_year = df_year_info[df_year_info["year_end"] > MIN_END_YEAR]["site_id"]

            # 除外されたサイトIDを特定 (診断用)
            all_sites_current = set(df['site_id'].unique())
            removed_sites_year = all_sites_current - set(valid_sites_by_year)

            # データフレーム df をフィルタリング
            df = df[df["site_id"].isin(valid_sites_by_year)].copy()

            # 診断結果の出力
            print(f"  > 最終観測年 {MIN_END_YEAR}以下で除外されたサイト: {removed_sites_year}")
            print(f"    - 最終観測年 {MIN_END_YEAR}以下で除外されたサイト数: {len(removed_sites_year)}")
            print(f"    - 残存サイト数: {df['site_id'].nunique()} / 総レコード数: {len(df)}")

            # --------------------
            # 年度の型変換が確実に行われているか再確認
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year']).copy() 
            df['year'] = df['year'].astype(int) 

            
            # 条件1: year > 2015 (2016年以降)
            condition_min = (df['year'] > MIN_ANALYSIS_YEAR)       
            condition_max = (df['year'] <= MAX_ANALYSIS_YEAR)

            # 両方の条件を満たすレコードのみを抽出
            df_filtered = df[condition_min & condition_max].copy() 

            # 診断結果の出力
            original_records = len(df)
            filtered_records = len(df_filtered)

            print(f"  > 観測年 [{MIN_ANALYSIS_YEAR + 1}年〜{MAX_ANALYSIS_YEAR}年] のデータに限定しました。")
            print(f"    - 除外されたレコード数: {original_records - filtered_records}")
            print(f"    - 残存サイト数: {df_filtered['site_id'].nunique()} / 残存レコード数: {filtered_records}")

            # ★ この後の df_filtered を使って MCMC 解析にかけます
            df = df_filtered
            # ----------------------------------------------------
            # concentrationの平均と標準偏差を計算
            mean_conc = df['concentration'].mean()
            std_conc = df['concentration'].std()
            
            # 出力
            print(f"濃度 (concentration) の平均: {mean_conc:.4f}")
            print(f"濃度 (concentration) の標準偏差: {std_conc:.4f}")
            # ----------------------------------


            # 年度整形
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(float)

            # site 観測数チェック
            counts = df.groupby("site_id").size()
            valid_sites = counts[counts >= 2].index
            df = df[df["site_id"].isin(valid_sites)]
            if df['site_id'].nunique() < 10:                 #観測数を設定
                print("地点数が不足 → スキップ")
                continue
            if len(df) < 10:
                print("総観測数が不足 → スキップ")
                continue

            MODEL_MAP = {
                "hier_regression_model": hier_regression_model,
                "hierarchical_student_t_noncentered": hierarchical_student_t_noncentered
            }

            MODEL = "hierarchical_student_t_noncentered"
            print(f"階層ベイズ解析（{MODEL} t分布モデル）を開始します...")

            prepared = prepare_data_from_dataframe(df, x_col="year", y_col="log_conc", site_col="site_id", min_records_per_site=5)

            result = run_numpyro_mcmc(prepared,
                          model=MODEL_MAP[MODEL],   # ← 関数に変換
                          num_warmup=1500,
                          num_samples=2000,
                          num_chains=4,
                          rng_seed=0,
                          progress_bar=True,
                          mu_a_sd=0.5, mu_b_sd=5.0,
                          sigma_a_scale=0.05, sigma_b_scale=0.2, sigma_obs_scale=0.2)

            #df_summary = summarize_parameters(result)      # ← 修正
            #df_summary.to_csv(f"{filename}_{substance}_parameter_summary2.csv", index=False)
            print("Compiling statistical information... summarize_parameters")
            df_global, df_sites, df_r2_sites = summarize_parameters(result)

            # 各 DataFrame を CSV に保存
            df_global.to_csv(f"outputs/{filename}_{substance}_summary_global.csv", index=False)
            df_sites.to_csv(f"outputs/{filename}_{substance}_summary_sites.csv", index=False)
            df_r2_sites.to_csv(f"outputs/{filename}_{substance}_summary_r2_sites.csv", index=False)

            # summarize_parameters の返り値を受け取る
            df_global, df_sites, df_r2_sites = summarize_parameters(result)

            # サイト別総合表を作成（前の例の df_summary_sites）
            df_sites_pivot = df_sites.pivot(
                index="site_id",
                columns="parameter",
                values=["mean", "sd", "ci_lower", "ci_upper",
                        "posterior_prob_gt0", "posterior_prob_lt0"]
            )
            df_sites_pivot.columns = [f"{param}_{stat}" for stat, param in df_sites_pivot.columns]
            df_sites_pivot = df_sites_pivot.reset_index()

            df_r2_sites_renamed = df_r2_sites.rename(
                columns={"mean": "R2_mean", "ci_lower": "R2_ci_lower", "ci_upper": "R2_ci_upper"}
            )

            df_summary_sites = pd.merge(df_sites_pivot, df_r2_sites_renamed, on="site_id", how="left")

            # --- site_id でソート ---
            df_summary_sites["site_id"] = pd.to_numeric(df_summary_sites["site_id"], errors="coerce").astype("Int64")
            df_summary_sites = df_summary_sites.sort_values(by="site_id", ascending=True)
            #df_summary_sites = df_summary_sites.sort_values(by="site_id")

            # CSV 保存
            df_summary_sites.to_csv(f"outputs/{filename}_{substance}_summary_sites_combined_sorted.csv", index=False)



            print(f"outputs/{filename}_{substance}_parameter_summary2.csv has saved")
            print(result["df_summary"].head())

            print("mu_a mean:", result["mu_a_mean"], "mu_a CI:", result["mu_a_ci"])
            print("sigma_a mean:", result["sigma_a_mean"], "sigma mean:", result["sigma_mean"])


            # 実行して出力を教えてください
            print("prepared keys:", prepared.keys())
            print("n_sites:", prepared.get("n_sites"))
            print("len df:", len(prepared.get("df", pd.DataFrame())))
            print("site_ids:", prepared.get("site_ids"))
            print("x_mean:", prepared.get("x_mean"))
            print("x_scale:", prepared.get("x_scale"))
            print("first rows of df:\n", prepared.get("df").head())

            print("prepare_data_from_dataframe for plot ..........please wait")             
            # posterior predictive example
            if  MAKE_PLOTS:            
                preds = posterior_predictive_intervals(result)
                print("prepare_data_from_dataframe for plot ..........finished")   

            MAKE_FACET_PLOTS = True
            if MAKE_FACET_PLOTS:  # Falseで図は描かない
                print("ploting plot_posterior_predictive_facet")
                fig = plot_posterior_predictive_facet(preds, result, prepared)
                fig.tight_layout()
                p2 = os.path.join("outputs/plots_regression/", f"{filename}_facet_{substance}.png")
                fig.savefig(p2, dpi=300); plt.close()
                print(f"[diagnose] trace 保存: {p2}")

            if MAKE_PRED_PLOTS:
                print("ploting plot_site_with_extrapolation")

                plot_site_with_extrapolation(preds, result, prepared, sid_target, year_target)
            """--
            #サイトでの予測
            sid = prepared["site_ids"][0]  # サイトを選択
            year_target = 2025
            pr = preds[str(sid)]
            x_plot = pr["x"]
            q_raw = pr["quantiles"]
            q = {float(k): v for k, v in q_raw.items()}

            # 指定年度に最も近いインデックスを探す
            idx = np.argmin(np.abs(x_plot - year_target))

            # その年度の予測値
            median_pred = q[50.0][idx]
            lower_pred = q[2.5][idx]
            upper_pred = q[97.5][idx]

            plt.fill_between(x_plot, q[2.5], q[97.5], alpha=0.25, label="95% CI")
            plt.plot(x_plot, q[50.0], label="median")

            # 観測点
            df_site = prepared["df"][prepared["df"]["site_id"] == sid]
            plt.scatter(df_site["year"], df_site["log_conc"], s=10, c="black")

            # 指定年度の予測点を赤で表示
            plt.scatter(year_target, median_pred, c="red", s=50, label="prediction")
            plt.errorbar(year_target, median_pred,
                        yerr=[[median_pred - lower_pred], [upper_pred - median_pred]],
                        fmt="o", color="red")

            plt.title(f"Site {sid}, Year {year_target}")
            plt.legend()
            plt.show()
            ---"""
            comparison=compare_sites(result, prepared)
            print(comparison)
            # 例: compare_sites() の結果を df に格納済み
            fig = plot_comparison_heatmaps(comparison)

            # 保存したい場合
            fig.savefig("outputs/site_comparison_heatmaps.png", dpi=300)
            print(f"{filename}_{substance} all finished ")

                        # 傾きでクラスタリング
            cluster_sites(comparison, target="prob_a_gt0")

            # 切片でクラスタリング
            cluster_sites(comparison, target="prob_b_gt0")


            fig, clusters = plot_group_regressions(result, prepared, comparison,
                                       target="prob_a_gt0", n_clusters=3)
            print(clusters)
            
            df_summary = summarize_clusters(result, prepared, comparison,
                                target="prob_a_gt0", n_clusters=3)

            print(df_summary)

            df_report = report_clusters(result, prepared, comparison,
                                        target="prob_a_gt0", n_clusters=3)

            print(df_report.to_string(index=False))

            # 例: summarize_clusters() や plot_group_regressions() で得た clusters を利用
            df_plot = plot_slope_intercept_clusters(result, prepared, clusters)


        print(f"{filename} all finished ")

    print("ALL finished")
#===================================以下、検討課題
def plot_site_with_extrapolation(preds, result, prepared, sid, year_target=None):
    """preds = posterior_predictive_intervals(result)
    pr = preds[str(sid)]
    x_plot = pr["x"]

    # quantiles を float key に変換
    q_raw = pr["quantiles"]
    q = {float(k): v for k, v in q_raw.items()}

    # 観測範囲を取得
    df_site = prepared["df"][prepared["df"]["site_id"] == sid]
    obs_years = df_site["year"].values
    xmin, xmax = obs_years.min(), obs_years.max()

    # 観測範囲内と外挿範囲を分ける
    in_range = (x_plot >= xmin) & (x_plot <= xmax)
    out_range = ~in_range

    plt.figure(figsize=(8, 5))

    # 観測範囲内: 濃い色で表示
    plt.fill_between(x_plot[in_range], q[2.5][in_range], q[97.5][in_range],
                    alpha=0.3, color="blue", label="95% CI (in range)")
    plt.plot(x_plot[in_range], q[50.0][in_range], color="blue", linewidth=2, label="median (in range)")

    # 外挿範囲: グレーで表示
    #plt.fill_between(x_plot[out_range], q[2.5][out_range], q[97.5][out_range],
    #                alpha=0.2, color="gray", label="95% CI (extrapolation)")
    plt.plot(x_plot[out_range], q[50.0][out_range], color="gray", linestyle="--", linewidth=2, label="median (extrapolation)")

    # 観測点
    plt.scatter(df_site["year"], df_site["log_conc"], s=10, c="black", label="observed")

    # 指定年度の予測点を赤で表示（オプション）
    if year_target is not None:
        idx = np.argmin(np.abs(x_plot - year_target))
        median_pred = q[50.0][idx]
        lower_pred = q[2.5][idx]
        upper_pred = q[97.5][idx]
        plt.scatter(year_target, median_pred, c="red", s=50, label="prediction")
        plt.errorbar(year_target, median_pred,
                    yerr=[[median_pred - lower_pred], [upper_pred - median_pred]],
                    fmt="o", color="red")

    plt.title(f"Site {sid}")
    plt.xlabel("Year")
    plt.ylabel("log(conc)")
    plt.legend()
    plt.show()
    """
    #sid = prepared["site_ids"][0]  # サイトを選択
    year_target = 2025
    pr = preds[str(sid)]
    x_plot = pr["x"]
    q_raw = pr["quantiles"]
    q = {float(k): v for k, v in q_raw.items()}

    # 指定年度に最も近いインデックスを探す
    idx = np.argmin(np.abs(x_plot - year_target))

    # その年度の予測値
    median_pred = q[50.0][idx]
    lower_pred = q[2.5][idx]
    upper_pred = q[97.5][idx]

    plt.fill_between(x_plot, q[2.5], q[97.5], alpha=0.25, label="95% CI")
    plt.plot(x_plot, q[50.0], label="median")

    # 観測点
    df_site = prepared["df"][prepared["df"]["site_id"] == sid]
    plt.scatter(df_site["year"], df_site["log_conc"], s=10, c="black")

    # 指定年度の予測点を赤で表示
    plt.scatter(year_target, median_pred, c="red", s=50, label="prediction")
    plt.errorbar(year_target, median_pred,
                yerr=[[median_pred - lower_pred], [upper_pred - median_pred]],
                fmt="o", color="red")

    plt.title(f"Site {sid}, Year {year_target}")
    plt.legend()
    plt.show()



import numpy as np
import pandas as pd

def compare_sites(result, prepared):
    """
    各サイトの回帰パラメータ (a: slope, b: intercept) の事後分布を比較し、
    サイト間の差がゼロを含むかどうかを確率で評価する。
    """

    site_ids = prepared["site_ids"]
    a_samples = result["samples"]["a"]  # shape: (n_draws, n_sites)
    b_samples = result["samples"]["b"]  # shape: (n_draws, n_sites)

    comparisons = []

    for i, sid_i in enumerate(site_ids):
        for j, sid_j in enumerate(site_ids):
            if j <= i:
                continue

            # 差の事後分布
            diff_a = a_samples[:, i] - a_samples[:, j]
            diff_b = b_samples[:, i] - b_samples[:, j]

            # 確率評価
            prob_a_gt0 = np.mean(diff_a > 0)
            prob_b_gt0 = np.mean(diff_b > 0)

            comparisons.append({
                "site_i": sid_i,
                "site_j": sid_j,
                "mean_diff_a": diff_a.mean(),
                "prob_a_gt0": prob_a_gt0,
                "mean_diff_b": diff_b.mean(),
                "prob_b_gt0": prob_b_gt0,
            })

    return pd.DataFrame(comparisons)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparison_heatmaps(df):
    """
    df: compare_sites() の結果 DataFrame
        必須列: site_i, site_j, prob_a_gt0, prob_b_gt0
    """

    # サイト一覧を取得
    sites = sorted(set(df["site_i"]).union(set(df["site_j"])), key=int)

    # 行列を初期化
    mat_a = pd.DataFrame(np.nan, index=sites, columns=sites)
    mat_b = pd.DataFrame(np.nan, index=sites, columns=sites)

    # 値を埋める
    for _, row in df.iterrows():
        i, j = row["site_i"], row["site_j"]
        mat_a.loc[i, j] = row["prob_a_gt0"]
        mat_a.loc[j, i] = 1 - row["prob_a_gt0"]  # 対称性を持たせる
        mat_b.loc[i, j] = row["prob_b_gt0"]
        mat_b.loc[j, i] = 1 - row["prob_b_gt0"]

    # ヒートマップ描画
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(mat_a, cmap="coolwarm", vmin=0, vmax=1, ax=axes[0], annot=False)
    axes[0].set_title("Probability slope difference (prob_a_gt0)")

    sns.heatmap(mat_b, cmap="coolwarm", vmin=0, vmax=1, ax=axes[1], annot=False)
    axes[1].set_title("Probability intercept difference (prob_b_gt0)")

    plt.tight_layout()
    plt.show()

    return fig

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

def cluster_sites(df, target="prob_a_gt0"):
    """
    サイト間比較結果からクラスタリングを行う。
    target: "prob_a_gt0"（傾き）か "prob_b_gt0"（切片）を指定
    """

    # サイト一覧
    sites = sorted(set(df["site_i"]).union(set(df["site_j"])), key=int)

    # 行列を初期化
    mat = pd.DataFrame(np.nan, index=sites, columns=sites)

    # 値を埋める
    for _, row in df.iterrows():
        i, j = row["site_i"], row["site_j"]
        mat.loc[i, j] = row[target]
        mat.loc[j, i] = 1 - row[target]  # 対称性を持たせる

    # NaN を 0.5（曖昧）で埋める
    mat = mat.fillna(0.5)

    # seaborn の clustermap を利用
    g = sns.clustermap(mat, cmap="coolwarm", vmin=0, vmax=1,
                       figsize=(10, 8), annot=False)

    plt.title(f"Clustering of sites based on {target}", y=1.05)
    plt.show()

    return g
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster

def plot_group_regressions(result, prepared, df_comparison, target="prob_a_gt0", n_clusters=3):
    """
    サイト間比較結果からクラスタリングを行い、グループごとの代表回帰式を描画する。
    
    result: MCMC結果 (samples["a"], samples["b"] を含む)
    prepared: データ準備結果 (site_ids, x_mean, x_scale, df を含む)
    df_comparison: compare_sites() の結果 DataFrame
    target: "prob_a_gt0" (傾き) または "prob_b_gt0" (切片)
    n_clusters: クラスタ数
    """

    site_ids = prepared["site_ids"]
    a_samples = result["samples"]["a"]
    b_samples = result["samples"]["b"]

    # サイト間距離行列を作成
    sites = sorted(set(df_comparison["site_i"]).union(set(df_comparison["site_j"])), key=int)
    mat = pd.DataFrame(np.nan, index=sites, columns=sites)
    for _, row in df_comparison.iterrows():
        i, j = row["site_i"], row["site_j"]
        mat.loc[i, j] = row[target]
        mat.loc[j, i] = 1 - row[target]
    mat = mat.fillna(0.5)

    # linkage によるクラスタリング
    Z = linkage(mat, method="average")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    # グループごとの代表回帰式を描画
    fig, ax = plt.subplots(figsize=(8,6))
    x_plot = np.linspace(prepared["df"]["year"].min(), prepared["df"]["year"].max(), 100)
    x_std = (x_plot - prepared["x_mean"]) / prepared["x_scale"]

    for k in range(1, n_clusters+1):
        group_sites = [sid for sid, c in zip(sites, clusters) if c == k]
        idxs = [site_ids.index(sid) for sid in group_sites]

        # グループ平均の回帰パラメータ
        a_group = a_samples[:, idxs].mean(axis=1)
        b_group = b_samples[:, idxs].mean(axis=1)
        y_group = np.outer(a_group, x_std) + b_group[:, None]

        # 代表線と予測区間
        mean_line = y_group.mean(axis=0)
        lower = np.percentile(y_group, 2.5, axis=0)
        upper = np.percentile(y_group, 97.5, axis=0)

        ax.plot(x_plot, mean_line, label=f"Cluster {k}")
        ax.fill_between(x_plot, lower, upper, alpha=0.2)

    ax.set_title("Representative regression lines by cluster")
    ax.set_xlabel("Year")
    ax.set_ylabel("log(conc)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig, clusters

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def summarize_clusters(result, prepared, df_comparison, target="prob_a_gt0", n_clusters=3):
    """
    サイト間比較結果からクラスタリングを行い、各クラスタの代表パラメータを表形式でまとめる。
    
    result: MCMC結果 (samples["a"], samples["b"] を含む)
    prepared: データ準備結果 (site_ids, df, x_mean, x_scale を含む)
    df_comparison: compare_sites() の結果 DataFrame
    target: "prob_a_gt0" (傾き) または "prob_b_gt0" (切片)
    n_clusters: クラスタ数
    """

    site_ids = prepared["site_ids"]
    a_samples = result["samples"]["a"]
    b_samples = result["samples"]["b"]

    # サイト間距離行列を作成
    sites = sorted(set(df_comparison["site_i"]).union(set(df_comparison["site_j"])), key=int)
    mat = pd.DataFrame(np.nan, index=sites, columns=sites)
    for _, row in df_comparison.iterrows():
        i, j = row["site_i"], row["site_j"]
        mat.loc[i, j] = row[target]
        mat.loc[j, i] = 1 - row[target]
    mat = mat.fillna(0.5)

    # linkage によるクラスタリング
    Z = linkage(mat, method="average")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    # 各クラスタの代表パラメータを計算
    cluster_summary = []
    for k in range(1, n_clusters+1):
        group_sites = [sid for sid, c in zip(sites, clusters) if c == k]
        idxs = [site_ids.index(sid) for sid in group_sites]

        a_group = a_samples[:, idxs].mean(axis=1)
        b_group = b_samples[:, idxs].mean(axis=1)

        cluster_summary.append({
            "cluster": k,
            "sites": group_sites,
            "mean_slope": a_group.mean(),
            "slope_CI95": (np.percentile(a_group, 2.5), np.percentile(a_group, 97.5)),
            "mean_intercept": b_group.mean(),
            "intercept_CI95": (np.percentile(b_group, 2.5), np.percentile(b_group, 97.5))
        })

    return pd.DataFrame(cluster_summary)

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def report_clusters(result, prepared, df_comparison, target="prob_a_gt0", n_clusters=3):
    """
    サイト間比較結果からクラスタリングを行い、政策評価用の報告書フォーマットで表をまとめる。
    
    result: MCMC結果 (samples["a"], samples["b"] を含む)
    prepared: データ準備結果 (site_ids, df, x_mean, x_scale を含む)
    df_comparison: compare_sites() の結果 DataFrame
    target: "prob_a_gt0" (傾き) または "prob_b_gt0" (切片)
    n_clusters: クラスタ数
    """

    site_ids = prepared["site_ids"]
    a_samples = result["samples"]["a"]
    b_samples = result["samples"]["b"]

    # サイト間距離行列を作成
    sites = sorted(set(df_comparison["site_i"]).union(set(df_comparison["site_j"])), key=int)
    mat = pd.DataFrame(np.nan, index=sites, columns=sites)
    for _, row in df_comparison.iterrows():
        i, j = row["site_i"], row["site_j"]
        mat.loc[i, j] = row[target]
        mat.loc[j, i] = 1 - row[target]
    mat = mat.fillna(0.5)

    # linkage によるクラスタリング
    Z = linkage(mat, method="average")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    # 各クラスタの代表パラメータを計算
    cluster_summary = []
    for k in range(1, n_clusters+1):
        group_sites = [sid for sid, c in zip(sites, clusters) if c == k]
        idxs = [site_ids.index(sid) for sid in group_sites]

        a_group = a_samples[:, idxs].mean(axis=1)
        b_group = b_samples[:, idxs].mean(axis=1)

        cluster_summary.append({
            "クラスタ番号": k,
            "所属サイト": ", ".join(map(str, group_sites)),
            "代表傾き (平均)": round(a_group.mean(), 3),
            "傾き95%区間": f"({round(np.percentile(a_group, 2.5),3)}, {round(np.percentile(a_group, 97.5),3)})",
            "代表切片 (平均)": round(b_group.mean(), 3),
            "切片95%区間": f"({round(np.percentile(b_group, 2.5),3)}, {round(np.percentile(b_group, 97.5),3)})"
        })

    # DataFrame化
    df_report = pd.DataFrame(cluster_summary)

    return df_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_slope_intercept_clusters(result, prepared, clusters):
    """
    各サイトの傾きと切片を散布図にし、クラスタごとに色分けして表示。
    
    result: MCMC結果 (samples["a"], samples["b"] を含む)
    prepared: データ準備結果 (site_ids を含む)
    clusters: 各サイトのクラスタ番号 (list or array, site_idsと同じ順序)
    """

    site_ids = prepared["site_ids"]
    a_samples = result["samples"]["a"]
    b_samples = result["samples"]["b"]

    # 各サイトの代表値（平均）を計算
    slopes = a_samples.mean(axis=0)
    intercepts = b_samples.mean(axis=0)

    df_plot = pd.DataFrame({
        "site_id": site_ids,
        "slope": slopes,
        "intercept": intercepts,
        "cluster": clusters
    })

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df_plot, x="slope", y="intercept",
                    hue="cluster", palette="Set1", s=80)

    for _, row in df_plot.iterrows():
        plt.text(row["slope"], row["intercept"], str(row["site_id"]),
                 fontsize=8, ha="right")

    plt.title("Slope vs Intercept by Cluster")
    plt.xlabel("Slope (mean)")
    plt.ylabel("Intercept (mean)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

    return df_plot
# ---------- Example usage ----------
if __name__ == "__main__":
    main()