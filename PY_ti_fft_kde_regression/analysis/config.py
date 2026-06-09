# analysis/config.py

import numpy as np

class BayesConfig:
    """
    FFT-KDE ベイズ回帰・TI・高速MCMC の共通設定クラス
    beta_schedule: の"default", "fine", "coarse", "test" を指定する。最終解析にはfineを推奨
    """
    def __init__(
        self,
        mode_loo: bool = True,
        grid_size_lookup: int = 4096,
        bw_factor: float = 1.0,
        n_walkers: int = 32,    #20～50程度が安定して動作する範囲。多すぎると計算負荷が増大する可能性あり。
        n_steps: int = 3000,    #2000～5,000程度に増やすと logZ の精度が向上する可能性あり
        burn_frac: float = 0.5,
        beta_schedule: str = "test",  # "default", "fine", "coarse", "test" などを選択可能にする
	):
        self.mode_loo = mode_loo
        self.grid_size_lookup = grid_size_lookup
        self.bw_factor = bw_factor
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.burn_frac = burn_frac
        self.beta_schedule = beta_schedule


    def __repr__(self) -> str:
        return (
            f"BayesConfig("
            f"mode_loo={self.mode_loo}, "
            f"grid_size_lookup={self.grid_size_lookup}, "
            f"bw_factor={self.bw_factor}, "
            f"n_walkers={self.n_walkers}, "
            f"n_steps={self.n_steps}, "
            f"burn_frac={self.burn_frac}, "
            f"beta_schedule='{self.beta_schedule}')"
        )


def build_beta_grid(cfg: BayesConfig):

    # -------------------------------------------------
    # default
    # -------------------------------------------------
    if cfg.beta_schedule == "default":

        # 一般用途
        # β=0.9〜1.0 をやや強化

        betas = np.concatenate([
            np.linspace(0, 0.9, 30, endpoint=False),
            np.linspace(0.9, 1.0, 21),
        ])

    # -------------------------------------------------
    # coarse
    # -------------------------------------------------
    elif cfg.beta_schedule == "coarse":

        # 粗いスケジュール
        # 高速確認用

        betas = np.linspace(0.0, 1.0, 21)

    # -------------------------------------------------
    # test
    # -------------------------------------------------
    elif cfg.beta_schedule == "test":

        # デバッグ用

        betas = np.linspace(0.0, 1.0, 3)

    # -------------------------------------------------
    # fine
    # -------------------------------------------------
    elif cfg.beta_schedule == "fine":

        # 高精度TI
        # β=0 と β=1 を強化

        betas = np.concatenate([

            np.linspace(0, 0.1, 20, endpoint=False),

            np.linspace(0.1, 0.9, 30, endpoint=False),

            np.linspace(0.9, 1.0, 31),

        ])

    # -------------------------------------------------
    # recommended
    # -------------------------------------------------
    elif cfg.beta_schedule == "recommended":

        # FFT-KDE + TI 推奨
        #
        # β→0 singularity 対策版
        #
        # 特徴:
        # - β=0近傍を超高密度化
        # - posterior領域も十分確保
        # - FFT-KDE evidence計算向け
        #
        # 実務推奨

        betas = np.concatenate([

            # ultra-low beta
            np.array([
                0.0,
                0.0005,
                0.001,
                0.002,
                0.003,
                0.005,
                0.0075,
                0.01,
            ]),

            # low beta
            np.linspace(0.01, 0.05, 12, endpoint=False),

            # medium beta
            np.linspace(0.05, 0.3, 20, endpoint=False),

            # posterior transition
            np.linspace(0.3, 0.8, 20, endpoint=False),

            # near posterior
            np.linspace(0.8, 0.95, 15, endpoint=False),

            # high beta
            np.linspace(0.95, 1.0, 26),

        ])

    # -------------------------------------------------
    # geometric
    # -------------------------------------------------
    elif cfg.beta_schedule == "geometric":

        # 幾何級数版
        # β=0付近を自然に高密度化

        low = np.geomspace(1e-4, 1.0, 80)

        betas = np.concatenate([
            [0.0],
            low
        ])

    # -------------------------------------------------
    # ultra_fine
    # -------------------------------------------------
    elif cfg.beta_schedule == "ultra_fine":

        # 非常に高精度
        # 論文検証レベル
        # 計算時間はかなり増加

        betas = np.concatenate([

            np.array([
                0.0,
                1e-4,
                2e-4,
                5e-4,
                1e-3,
                2e-3,
                5e-3,
            ]),

            np.linspace(0.005, 0.05, 30, endpoint=False),

            np.linspace(0.05, 0.2, 30, endpoint=False),

            np.linspace(0.2, 0.8, 40, endpoint=False),

            np.linspace(0.8, 0.95, 20, endpoint=False),

            np.linspace(0.95, 1.0, 41),

        ])

    # -------------------------------------------------
    # fallback
    # -------------------------------------------------
    else:

        betas = np.linspace(0.0, 1.0, 51)

    return np.unique(np.sort(betas))