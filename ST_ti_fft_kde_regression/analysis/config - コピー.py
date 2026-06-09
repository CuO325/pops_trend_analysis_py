# analysis/config.py


class BayesConfig:
    """
    FFT-KDE ベイズ回帰・TI・高速MCMC の共通設定クラス
    """
    def __init__(
        self,
        mode_loo: bool = True,
        grid_size_lookup: int = 4096,
        bw_factor: float = 1.0,
        n_walkers: int = 32,    #20～50程度が安定して動作する範囲。多すぎると計算負荷が増大する可能性あり。
        n_steps: int = 3000,    #2000～5,000程度に増やすと logZ の精度が向上する可能性あり
        burn_frac: float = 0.5,
        beta_schedule: str = "coarse",  # _build_beta_grid(cfg: BayesConfig)関数の"default", "fine", "coarse" を指定する。 マッチしなければ"linear" betas = np.linspace(0.0, 1.0, 51) 
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
