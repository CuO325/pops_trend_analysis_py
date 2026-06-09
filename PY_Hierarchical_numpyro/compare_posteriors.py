import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import japanize_matplotlib

def compare_posteriors(emcee_samples, numpyro_samples, param_name="傾き (slope)"):
    """
    emceeとNumPyroの事後分布を重ねてプロットする
    """
    plt.figure(figsize=(10, 6))

    # emceeの分布 (青)
    sns.kdeplot(emcee_samples, fill=True, color="skyblue", label="emcee (v4.2)", alpha=0.5)
    # NumPyroの分布 (橙)
    sns.kdeplot(numpyro_samples, fill=True, color="orange", label="NumPyro (NUTS)", alpha=0.5)

    # 統計量の計算
    emcee_mean = np.mean(emcee_samples)
    numpyro_mean = np.mean(numpyro_samples)

    # 平均値を垂直線で表示
    plt.axvline(emcee_mean, color="blue", linestyle="--", alpha=0.8)
    plt.axvline(numpyro_mean, color="darkorange", linestyle="--", alpha=0.8)

    plt.title(f"手法間の事後分布比較: {param_name}")
    plt.xlabel("値")
    plt.ylabel("密度")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 違いの数値表示
    diff = abs(emcee_mean - numpyro_mean)
    plt.annotate(f"平均値の差: {diff:.6f}", xy=(0.05, 0.95), xycoords='axes fraction')

    plt.show()

# --- データの抽出例 ---
# emcee版 (n_sitesはサイト数)
# emcee_slope = results['samples'][:, 2 * n_sites] 

# NumPyro版
# numpyro_slope = mcmc.get_samples()["mu_a"]