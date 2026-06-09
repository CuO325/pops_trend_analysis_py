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
    # "default", "fine", "coarse", "linear", "geometric" などのスケジュールを選択可能にする
    if cfg.beta_schedule == "default":  # 傾向が強いデータ向けのスケジュール（TIの精度を高めるために、β=0.9付近を強化）
        betas = np.concatenate([
            np.linspace(0, 0.9, 30, endpoint=False),
            np.linspace(0.9, 1.0, 21),
        ])
    elif cfg.beta_schedule == "coarse":     # 粗いスケジュール（TIの計算時間を短縮したい場合に）
        betas = np.linspace(0.0, 1.0, 21)
    elif cfg.beta_schedule == "test": 
        betas = np.linspace(0.0, 1.0, 3)  # テスト用の極端に粗いスケジュール
    elif cfg.beta_schedule == "fine":
        # 傾向が弱いデータ向けの高精度スケジュール
        # 尤度の変化が激しい β=0 付近と β=1 付近を重点的にサンプリング
        betas = np.concatenate([
            np.linspace(0, 0.1, 20, endpoint=False),  # 低β域を強化 (15->20)
            np.linspace(0.1, 0.9, 30, endpoint=False),
            np.linspace(0.9, 1.0, 31),               # 高β域も強化 (25->31)
        ])
    else:
        # 未定義のスケジュール名が指定された場合のフォールバック
        betas = np.linspace(0.0, 1.0, 51)
        
    return np.unique(np.sort(betas))