"""
MINIMUM DETECTABLE FLOW (stretch task, one fixed operating condition)

For each leak node and each candidate sensor node: the smallest leak flow at which the pressure (head) DROP
at the sensor exceeds a stated threshold (metres).  Detected means drop > threshold (strict).

HOW THE LEAK IS MODELLED: the leak flow is ADDED DEMAND at the leak node (demand-adding method, a constant
flow, NOT a leak area / emitter).  Simulator: wntr.sim.WNTRSimulator, demand-driven analysis.
OPERATING CONDITION: one time step.  duration = 0, `pattern_start = time_s` picks the demand-pattern multiplier
at that time; tanks stay at their initial level.  Baseline = same model without the added flow.

SEARCH RULES
 * the response over an ascending flow grid is checked for monotonicity FIRST;
 * monotone + crosses inside the grid  -> "resolved": smallest grid flow above the threshold, optionally refined
   by bisection on DIRECT hydraulic runs between the last-below and first-above grid flows (no interpolation);
 * exceeded already at the smallest grid flow -> "below_range" (UNRESOLVED, no finite value);
 * never exceeded up to the largest flow       -> "above_range" (UNRESOLVED, no finite value);
 * non-monotone                                -> "non_monotonic" (FLAGGED, first grid crossing kept separately,
   no bisection, excluded from the summary mean);
 * failed / NaN simulation                     -> "sim_failed" (UNRESOLVED).
Unweighted means are called "mean minimum detectable flow". This is NOT Du et al.'s ADT.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

RESOLVED, BELOW, ABOVE, NONMONO, FAILED = "resolved", "below_range", "above_range", "non_monotonic", "sim_failed"
UNRESOLVED_STATUSES = (BELOW, ABOVE, FAILED)


def geometric_flows(f_min, f_max, n):
    if not (0 < f_min < f_max) or n < 2:
        raise ValueError("need 0 < f_min < f_max and n >= 2")
    return list(np.geomspace(f_min, f_max, n))


def is_monotone_non_decreasing(drops, tol=1e-9):
    d = np.asarray(drops, dtype=float)
    return bool(np.all(np.diff(d) >= -(tol + 1e-9 * np.nanmax(np.abs(d)))))


@dataclass
class FlowResult:
    status: str
    min_flow: float          # NaN unless status == resolved (or the non_monotonic first crossing in `first_crossing`)
    first_crossing: float    # smallest grid flow with drop > threshold (NaN if none)
    bracket: tuple           # (last grid flow below, first grid flow above) or (nan, nan)
    n_runs: int = 0


def find_min_flow(drop_fn, threshold_m, flows, refine_steps=0):
    """
    drop_fn(q) -> head drop (m) at ONE sensor for added flow q (from a direct hydraulic run).
    `flows` ascending grid. Returns FlowResult. Bisection only if monotone and refine_steps > 0.
    """
    flows = list(flows)
    if flows != sorted(flows) or len(flows) < 2:
        raise ValueError("flows must be ascending with at least 2 values")
    drops = [drop_fn(q) for q in flows]
    runs = len(flows)
    if any(d is None or not np.isfinite(d) for d in drops):
        return FlowResult(FAILED, math.nan, math.nan, (math.nan, math.nan), runs)
    exceeded = [d > threshold_m for d in drops]
    first = next((q for q, e in zip(flows, exceeded) if e), math.nan)
    if not is_monotone_non_decreasing(drops):
        return FlowResult(NONMONO, math.nan, first, (math.nan, math.nan), runs)
    if not any(exceeded):
        return FlowResult(ABOVE, math.nan, math.nan, (math.nan, math.nan), runs)
    if exceeded[0]:
        return FlowResult(BELOW, math.nan, first, (math.nan, flows[0]), runs)
    i = exceeded.index(True)
    lo, hi = flows[i - 1], flows[i]
    bracket = (lo, hi)
    for _ in range(refine_steps):
        mid = 0.5 * (lo + hi)
        d = drop_fn(mid)
        runs += 1
        if d is None or not np.isfinite(d):
            return FlowResult(FAILED, math.nan, first, bracket, runs)
        lo, hi = (lo, mid) if d > threshold_m else (mid, hi)
    return FlowResult(RESOLVED, hi, first, bracket, runs)


# ------------------------------------------------------------------ hydraulics
def _run_pressures(wn_factory, leak_node, flow_m3s, time_s):
    import wntr
    wn = wn_factory()
    wn.options.hydraulic.demand_model = "DDA"
    wn.options.time.duration = 0
    wn.options.time.pattern_start = int(time_s)
    if flow_m3s:
        node = wn.get_node(leak_node)
        node.add_demand(flow_m3s, None)                  # constant added demand, no pattern
    try:
        res = wntr.sim.WNTRSimulator(wn).run_sim()
        p = res.node["pressure"][wn.junction_name_list].iloc[0]
        return p
    except Exception:
        return None


def sweep_leak_node(wn_factory, leak_node, sensors, threshold_m, flows, time_s=0, refine_steps=0):
    """All sensors for one leak node. Hydraulic runs are cached by flow."""
    cache = {}
    base = _run_pressures(wn_factory, leak_node, 0.0, time_s)
    if base is None:
        raise RuntimeError("baseline simulation failed")

    def drops_at(q):
        if q not in cache:
            p = _run_pressures(wn_factory, leak_node, q, time_s)
            cache[q] = None if p is None else (base - p)
        return cache[q]

    out = {}
    for s in sensors:
        def f(q, s=s):
            d = drops_at(q)
            return None if d is None else float(d[s])
        out[s] = find_min_flow(f, threshold_m, flows, refine_steps)
    return out


def build_matrix(wn_factory, leak_nodes, sensors, threshold_m, flows, time_s=0, refine_steps=0):
    """Long table: one row per (leak_node, sensor). Flows in m3/s."""
    rows = []
    for ln in leak_nodes:
        for s, r in sweep_leak_node(wn_factory, ln, sensors, threshold_m, flows, time_s, refine_steps).items():
            rows.append(dict(leak_node=ln, sensor=s, status=r.status, min_flow_m3s=r.min_flow,
                             first_grid_crossing_m3s=r.first_crossing, bracket_lo=r.bracket[0], bracket_hi=r.bracket[1]))
    return pd.DataFrame(rows)


def summarise(matrix, percentile=90):
    """Mean / upper percentile over RESOLVED pairs only, plus the shares that are not resolved."""
    n = len(matrix)
    res = matrix[matrix.status == RESOLVED]["min_flow_m3s"]
    share = lambda sts: float(matrix.status.isin(sts).mean()) if n else math.nan
    return dict(n_pairs=n, n_resolved=int(len(res)),
                mean_minimum_detectable_flow_m3s=float(res.mean()) if len(res) else math.nan,
                upper_percentile=percentile,
                upper_percentile_flow_m3s=float(np.percentile(res, percentile)) if len(res) else math.nan,
                unresolved_share=share(UNRESOLVED_STATUSES), non_monotonic_share=share([NONMONO]),
                metric_name="mean minimum detectable flow (unweighted; not Du et al.'s ADT)")
