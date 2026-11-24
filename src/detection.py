"""
DETECTABILITY: when does a candidate logger see a leak?

residual r(t) = pressure with leak - pressure in the matching no-leak run   (metres of head)

RULE (experimental choice, NOT an SA Water standard):
  A scenario is observable at a candidate when |r| > tau for at least `persistence`
  CONSECUTIVE reported samples between leak onset and leak end.
  The detection time is the time of the sample that completes the run.
"""
import numpy as np
import pandas as pd

from cost_func import KPA_PER_M_HEAD


def detect_scenario(residual: pd.Series, onset_s: int, end_s: int, tau_m: float, persistence: int = 3):
    """Return (detected, detection_time_s, max_abs_residual_m)."""
    window = residual[(residual.index >= onset_s) & (residual.index < end_s)]
    if window.empty:
        return False, float("nan"), float("nan")
    max_abs = float(window.abs().max())
    run = 0
    for t, exceeded in zip(window.index, (window.abs() > tau_m).to_numpy()):
        run = run + 1 if exceeded else 0
        if run >= persistence:
            return True, float(t), max_abs
    return False, float("nan"), max_abs


def build_detection_table(baseline_pressure, scenarios, runs, candidates, tau_m, persistence=3) -> pd.DataFrame:
    """One row per (scenario, candidate). Failed scenarios stay in the table as NOT detected."""
    rows = []
    for sc in scenarios:
        run = runs[sc.scenario_id]
        for c in candidates:
            if run["pressure"] is None:
                rows.append(dict(scenario_id=sc.scenario_id, candidate_id=c, detected=False,
                                 first_detection_time=np.nan, detection_delay_min=np.nan,
                                 max_residual=np.nan, scenario_status=run["status"]))
                continue
            residual = run["pressure"][c] - baseline_pressure[c]
            ok, t, mx = detect_scenario(residual, sc.start_s, sc.end_s, tau_m, persistence)
            rows.append(dict(scenario_id=sc.scenario_id, candidate_id=c, detected=ok,
                             first_detection_time=t, detection_delay_min=(t - sc.start_s) / 60 if ok else np.nan,
                             max_residual=mx, scenario_status=run["status"]))
    return pd.DataFrame(rows)


def range_check(baseline_pressure, runs, candidates, device) -> pd.DataFrame:
    """Mandatory: exclude candidates whose simulated pressure leaves the logger's range."""
    frames = [baseline_pressure[candidates]] + [r["pressure"][candidates] for r in runs.values() if r["pressure"] is not None]
    allp = pd.concat(frames)
    rows = []
    for c in candidates:
        hi, lo = float(allp[c].max()) * KPA_PER_M_HEAD, float(allp[c].min()) * KPA_PER_M_HEAD
        reason = ""
        if hi > device.range_max_kpa:
            reason = f"max {hi:.0f} kPa above range {device.range_max_kpa:.0f} kPa"
        elif lo < device.range_min_kpa:
            reason = f"min {lo:.0f} kPa below range {device.range_min_kpa:.0f} kPa"
        rows.append(dict(candidate_id=c, max_kpa=round(hi, 1), min_kpa=round(lo, 1),
                         eligible=(reason == ""), reason=reason))
    return pd.DataFrame(rows)
