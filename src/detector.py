"""
DETECTOR: residuals, thresholds, persistence, event scoring.  The detector NEVER receives labels.

SIGN CONVENTION   drop(t) = baseline(t) - observed(t)        positive = pressure LOWER than expected.
ALARM DIRECTION   one-sided: a sample is an exceedance when drop > threshold.
                  Why one-sided: Zhao et al. (2020), Du et al. (2024) and Quintiliani et al. (2020) all detect pressure
                  DROPS. Whether pressure rises matter in this dataset is TESTED (rise_check), not assumed.
THRESHOLD         tau_i = max(empirical q-quantile of the normal-period drop at node i, sensor accuracy).
                  The quantile comes from NORMAL data (leak-free scenarios and pre-leak history) in the development set.
                  Sensor accuracy is a floor, NOT a Gaussian noise SD. Persistence k = consecutive exceedances needed.
"""
import numpy as np
import pandas as pd


def drop_residual(baseline: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    return baseline - observed


def run_length(flag: np.ndarray) -> np.ndarray:
    """Length of the current run of True values at each position (0 where flag is False)."""
    idx = np.arange(len(flag))
    last_false = np.maximum.accumulate(np.where(~flag, idx, -1))
    return idx - last_false


def alarm_state(drop: pd.DataFrame, tau, persistence: int) -> pd.DataFrame:
    """True while an alarm is active: at least `persistence` consecutive exceedances. NaN residual never alarms."""
    tau = pd.Series(tau, index=drop.columns) if np.isscalar(tau) else tau.reindex(drop.columns)
    exceed = (drop.gt(tau, axis=1)) & drop.notna()
    state = exceed.apply(lambda c: run_length(c.to_numpy()) >= persistence, axis=0, result_type="broadcast")
    return state.astype(bool)


def alarm_onsets(state: pd.DataFrame) -> pd.DataFrame:
    """Steps at which an alarm STARTS (state True now, False the step before)."""
    prev = state.shift(1, fill_value=False)
    return state & ~prev


def calibrate_thresholds(normal_drops: pd.DataFrame, q: float, accuracy_m: float) -> pd.Series:
    """Per-node tau = max(q-quantile of normal drop, accuracy). `normal_drops` must hold NORMAL data only."""
    qs = normal_drops.quantile(q)
    return qs.clip(lower=accuracy_m)


def score_events(state: pd.DataFrame, drop: pd.DataFrame, events, horizon_steps: int) -> pd.DataFrame:
    """
    One row per (event, node). The labels are used HERE ONLY, to score, never to detect.
    Detected = an alarm STARTS inside the event window [start, min(end, start+horizon)).
    An alarm already running when the event starts is conservatively NOT counted.
    """
    onsets = alarm_onsets(state)
    rows = []
    for ei, ev in enumerate(events):
        stop = min(ev.end, ev.start + horizon_steps)
        w_on, w_drop = onsets.iloc[ev.start:stop], drop.iloc[ev.start:stop]
        for node in state.columns:
            hit = np.flatnonzero(w_on[node].to_numpy())
            rows.append(dict(event=ei, node=node, detected=bool(len(hit)),
                             delay_h=float(hit[0]) * 0.5 if len(hit) else np.nan,
                             max_drop_m=float(w_drop[node].max()) if w_drop[node].notna().any() else np.nan))
    return pd.DataFrame(rows)


def false_alarm_rate(state: pd.DataFrame, normal_mask: np.ndarray) -> pd.Series:
    """Alarm episodes per day, per node, counted only on steps where normal_mask is True."""
    onsets = alarm_onsets(state)
    days = normal_mask.sum() / 48
    return onsets[normal_mask].sum() / days if days > 0 else onsets.iloc[0:0].sum() * np.nan


def rise_check(baseline: pd.DataFrame, observed: pd.DataFrame, event_mask: np.ndarray, tau) -> dict:
    """Does a leak ever make pressure RISE beyond tau? (tests the one-sided choice)"""
    rise = (observed - baseline)[event_mask]
    drop = (baseline - observed)[event_mask]
    tau = pd.Series(tau, index=rise.columns) if np.isscalar(tau) else tau
    return dict(share_samples_rise_over_tau=float((rise.gt(tau, axis=1)).to_numpy().mean()),
                share_samples_drop_over_tau=float((drop.gt(tau, axis=1)).to_numpy().mean()),
                max_rise_m=float(rise.max().max()), max_drop_m=float(drop.max().max()))
