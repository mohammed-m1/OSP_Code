"""
BASELINES: what pressure would we expect if there were no leak?

LEVEL A  matched baseline (idealised)   re-run the scenario's own saved .inp WITHOUT a leak. Same demands, same
                                        perturbed network, same patterns: the ONLY difference is the leak.
                                        An operator cannot do this; it shows what is detectable under perfect knowledge.
LEVEL B  estimated baseline (realistic) built only from the scenario's own PAST pressure data. Never uses labels,
                                        future observations or hidden true demands.
     B1 frozen  : median pressure per time-of-day slot over the first `train_days` days, then held fixed.
     B2 rolling : median per slot over a trailing window, skipping the most recent `guard_days` days so a developing
                  leak does not contaminate the baseline immediately. Updated once per day.
"""
import numpy as np
import pandas as pd

SLOTS_PER_DAY = 48            # 30-minute steps
EXPECTED_WNTR = "1.2.0"       # the dataset was generated with this version


# ---------------------------------------------------------------- Level A
def matched_baseline(scn, upto_step: int) -> pd.DataFrame:
    """Pressures (m) for steps 0 .. upto_step-1 from the scenario .inp re-run with NO leak."""
    import wntr
    from wntr.epanet.util import from_si, HydParam, FlowUnits
    if wntr.__version__ != EXPECTED_WNTR:
        print(f"WARNING: WNTR {wntr.__version__} differs from the generator's {EXPECTED_WNTR}; check reconstruction error.")
    wn = wntr.network.WaterNetworkModel(str(scn.inp_path))
    wn.options.hydraulic.demand_model = "PDD"            # the generator switches to PDD AFTER writing the .inp
    wn.options.time.duration = (upto_step - 1) * 30 * 60
    res = wntr.sim.WNTRSimulator(wn).run_sim()
    p = from_si(FlowUnits.LPS, res.node["pressure"], HydParam.Pressure).round(5)     # same rounding as the CSVs
    p.index = scn.pressures.index[:len(p)]
    return p[list(getattr(scn, "all_pressures", scn.pressures).columns)]


def reconstruction_error(scn, base: pd.DataFrame) -> dict:
    """Compare the re-run with the dataset pressures on steps NOT affected by any leak (label == 0)."""
    n = len(base)
    clean = ~scn.labels[:n]
    full = getattr(scn, "all_pressures", scn.pressures)
    diff = (full.iloc[:n][clean] - base[clean]).abs()
    bad = diff.max(axis=1) > 0.01                       # any candidate differs by more than 1 cm at this step
    return dict(steps_compared=int(clean.sum()), max_abs_error_m=float(diff.max().max()),
                mean_abs_error_m=float(diff.mean().mean()), steps_over_0p01m=int(bad.sum()),
                share_over_0p01m=round(float(bad.sum()) / max(1, int(clean.sum())), 5))


# ---------------------------------------------------------------- Level B
def _slot(index) -> np.ndarray:
    return (index.hour * 2 + index.minute // 30).to_numpy()


def frozen_baseline(p: pd.DataFrame, train_days: int) -> pd.DataFrame:
    """Median per slot from the first train_days days; NaN before the training window ends (not yet available)."""
    n_train = train_days * SLOTS_PER_DAY
    slot = _slot(p.index)
    med = p.iloc[:n_train].groupby(slot[:n_train]).median()
    base = med.reindex(slot).set_axis(p.index)
    base.iloc[:n_train] = np.nan
    return base


def rolling_baseline(p: pd.DataFrame, window_days: int = 28, guard_days: int = 7) -> pd.DataFrame:
    """Causal rolling baseline. Day d uses only days d-guard-window .. d-guard-1 (never the guard period, never today)."""
    n_days = len(p) // SLOTS_PER_DAY
    arr = p.to_numpy()[:n_days * SLOTS_PER_DAY].reshape(n_days, SLOTS_PER_DAY, -1)
    out = np.full_like(arr, np.nan)
    for d in range(window_days + guard_days, n_days):
        out[d] = np.median(arr[d - guard_days - window_days: d - guard_days], axis=0)
    full = np.full(p.shape, np.nan)
    full[:n_days * SLOTS_PER_DAY] = out.reshape(n_days * SLOTS_PER_DAY, -1)
    return pd.DataFrame(full, index=p.index, columns=p.columns)


def warmup_steps(method: str, train_days: int = 56, window_days: int = 28, guard_days: int = 7) -> int:
    """First step at which the estimated baseline exists."""
    return (train_days if method == "frozen" else window_days + guard_days) * SLOTS_PER_DAY
