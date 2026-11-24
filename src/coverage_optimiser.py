"""
COVERAGE OPTIMISER (synthetic-matrix friendly, no hydraulics)

Inputs (NumPy arrays)
  D  bool  (scenarios, candidates)  candidate i detects scenario s
  T  float (scenarios, candidates)  detection delay, minutes; NaN where not detected (may be None)
  w  float (scenarios,)             scenario weights (default: equal)
  candidates  list of node-ID strings (default "0","1",...)
  k  number of sensors

Coverage = sum of weights of detected scenarios / sum of ALL weights.
Scenarios nobody detects therefore STAY IN THE DENOMINATOR.

Declared tie-break for layouts with equal objective:
  1 highest weighted coverage, 2 shortest mean detection delay (earliest alarm per detected
  scenario, averaged over detected scenarios), 3 natural node-ID order (J2 before J10).

Status strings: "optimal" ONLY when HiGHS reports optimal. Others: "time_limit_feasible",
"time_limit_no_solution", "infeasible", "solver_error", "trivial", "exhaustive", "heuristic".
"""
import itertools
import math
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

DEFAULT_ENUM_LIMIT = 200_000       # max layouts enumerated for the MIP tie-break step
DEFAULT_EXHAUSTIVE_LIMIT = 2_000_000


def natural_key(name: str):
    """'J10' sorts after 'J2'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


@dataclass
class LayoutResult:
    method: str
    indexes: List[int]
    nodes: List[str]
    objective: float                # weighted detected scenarios (sum of w_s for detected s)
    coverage: float                 # objective / sum(w); missed scenarios stay in denominator
    n_detected: int
    mean_delay_min: float           # inf if nothing detected
    status: str
    gap: Optional[float] = None     # relative MIP gap if the solver reported one
    runtime_s: float = 0.0
    solver_objective: Optional[float] = None   # raw HiGHS objective before the tie-break step
    tie_break_applied: bool = False
    note: str = ""


# ---------------------------------------------------------------- validation
def _prepare(D, T, w, candidates, k):
    D = np.asarray(D, dtype=bool)
    if D.ndim != 2:
        raise ValueError("D must be 2-D (scenarios, candidates).")
    S, n = D.shape
    if S == 0:
        raise ValueError("D has no scenarios.")
    if T is not None:
        T = np.asarray(T, dtype=float)
        if T.shape != D.shape:
            raise ValueError("T must have the same shape as D.")
        if np.isnan(T[D]).any():
            raise ValueError("T is NaN where D says the scenario is detected.")
    w = np.ones(S) if w is None else np.asarray(w, dtype=float)
    if w.shape != (S,) or (w < 0).any() or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("w must have one finite non-negative weight per scenario, with a positive sum.")
    names = [str(i) for i in range(n)] if candidates is None else [str(c) for c in candidates]
    if len(names) != n or len(set(names)) != n:
        raise ValueError("candidates must be unique and match the columns of D.")
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer.")
    return D, T, w, names, min(int(k), n)


def layout_metrics(D, T, w, idx):
    """Weighted coverage, detected count and mean earliest delay of the layout `idx`."""
    idx = list(idx)
    if not idx:
        return dict(objective=0.0, coverage=0.0, n_detected=0, mean_delay=math.inf)
    sub = D[:, idx]
    mask = sub.any(axis=1)
    obj = float(w[mask].sum())
    if not mask.any():
        delay = math.inf
    elif T is None:
        delay = 0.0
    else:
        earliest = np.where(sub, T[:, idx], np.inf).min(axis=1)
        delay = float(earliest[mask].mean())
    return dict(objective=obj, coverage=(obj / float(w.sum()) if w.sum() > 0 else float("nan")), n_detected=int(mask.sum()), mean_delay=delay)


def _key(idx, D, T, w, names):
    m = layout_metrics(D, T, w, idx)
    order = sorted(idx, key=lambda i: natural_key(names[i]))
    return (-round(m["objective"], 9), round(m["mean_delay"], 9), [natural_key(names[i]) for i in order])


def _result(method, idx, D, T, w, names, status, **kw):
    idx = sorted(idx, key=lambda i: natural_key(names[i]))
    m = layout_metrics(D, T, w, idx)
    return LayoutResult(method=method, indexes=list(idx), nodes=[names[i] for i in idx],
                        objective=m["objective"], coverage=m["coverage"], n_detected=m["n_detected"],
                        mean_delay_min=m["mean_delay"], status=status, **kw)


# ---------------------------------------------------------------- exhaustive
def exhaustive_coverage(D, T=None, w=None, k=1, candidates=None, max_combinations=DEFAULT_EXHAUSTIVE_LIMIT):
    """Brute force over all layouts of size min(k, n). Provably optimal; small cases only."""
    t0 = time.time()
    D, T, w, names, k_eff = _prepare(D, T, w, candidates, k)
    n = D.shape[1]
    if math.comb(n, k_eff) > max_combinations:
        raise ValueError(f"{math.comb(n, k_eff)} layouts exceed max_combinations={max_combinations}.")
    best, best_key = (), None
    for combo in itertools.combinations(range(n), k_eff):
        key = _key(list(combo), D, T, w, names)
        if best_key is None or key < best_key:
            best, best_key = combo, key
    return _result("exhaustive", list(best), D, T, w, names, "exhaustive", runtime_s=time.time() - t0)


# ---------------------------------------------------------------- greedy
def greedy_coverage(D, T=None, w=None, k=1, candidates=None):
    """Marginal-coverage greedy; ties by delay then node order. A heuristic: no optimality claim."""
    t0 = time.time()
    D, T, w, names, k_eff = _prepare(D, T, w, candidates, k)
    chosen = []
    for _ in range(k_eff):
        rest = [c for c in range(D.shape[1]) if c not in chosen]
        chosen.append(min(rest, key=lambda c: _key(chosen + [c], D, T, w, names)))
    return _result("greedy", chosen, D, T, w, names, "heuristic", runtime_s=time.time() - t0)


# ---------------------------------------------------------------- random baseline
def random_layouts(D, w=None, k=1, n=100, seed=0, T=None, candidates=None):
    """n random layouts of size min(k, candidates). Same seed -> identical output."""
    D, T, w, names, k_eff = _prepare(D, T, w, candidates, k)
    rng = np.random.default_rng(seed)
    layouts = np.array([np.sort(rng.choice(D.shape[1], k_eff, replace=False)) for _ in range(n)],
                       dtype=int).reshape(n, k_eff)
    cov = np.array([layout_metrics(D, T, w, row)["coverage"] for row in layouts])
    return dict(layouts=layouts, coverage=cov, mean_coverage=float(cov.mean()) if n else float("nan"),
                best_coverage=float(cov.max()) if n else float("nan"), seed=seed, k=k_eff)


# ---------------------------------------------------------------- sensitivity-ranked
def sensitivity_ranked(D, T=None, w=None, k=1, candidates=None):
    """Top-k candidates by INDIVIDUAL weighted coverage (ignores overlap; a naive baseline)."""
    t0 = time.time()
    D, T, w, names, k_eff = _prepare(D, T, w, candidates, k)
    order = sorted(range(D.shape[1]), key=lambda c: _key([c], D, T, w, names))
    return _result("sensitivity_ranked", order[:k_eff], D, T, w, names, "heuristic", runtime_s=time.time() - t0)


# ---------------------------------------------------------------- MIP
def maximise_coverage_mip(D, w=None, k=1, time_limit_s=60.0, T=None, candidates=None,
                          tie_break_enum_limit=DEFAULT_ENUM_LIMIT):
    """
    maximise sum_s w_s y_s   s.t.  y_s <= sum_i D[s,i] x_i,  sum_i x_i <= k,  x,y binary (scipy.optimize.milp / HiGHS).

    status == "optimal" only if HiGHS says so. After an optimal solve, if the number of layouts is
    <= tie_break_enum_limit, the declared tie-break is applied among layouts with the SAME objective
    (tie_break_applied=True); otherwise the solver's own layout is returned (padded deterministically
    to size k if it used fewer). `solver_objective` always holds the raw HiGHS value.
    """
    t0 = time.time()
    D, T, w, names, k_eff = _prepare(D, T, w, candidates, k)
    if not time_limit_s or time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive.")
    S, n = D.shape
    if k_eff == 0 or n == 0:
        return _result("mip", [], D, T, w, names, "trivial", runtime_s=time.time() - t0,
                       note="no sensors can be placed; solver not run")
    c = np.concatenate([np.zeros(n), -w])
    A = np.vstack([np.hstack([-D.astype(float), np.eye(S)]),
                   np.concatenate([np.ones(n), np.zeros(S)])[None, :]])
    cons = LinearConstraint(A, -np.inf, np.concatenate([np.zeros(S), [k_eff]]))
    try:
        res = milp(c, constraints=cons, integrality=np.ones(n + S), bounds=Bounds(0, 1),
                   options={"time_limit": float(time_limit_s), "disp": False})
    except Exception as exc:                                          # never hide a solver failure
        return _result("mip", [], D, T, w, names, "solver_error", runtime_s=time.time() - t0, note=repr(exc))
    gap = getattr(res, "mip_gap", None)
    gap = None if gap is None or (isinstance(gap, float) and not np.isfinite(gap)) else float(gap)
    msg = str(getattr(res, "message", ""))
    x = getattr(res, "x", None)
    if res.status == 0 and x is not None:
        status = "optimal"
    elif res.status == 1:
        status = "time_limit_feasible" if x is not None else "time_limit_no_solution"
    elif res.status == 2:
        status = "infeasible"
    else:
        status = "solver_error" if x is None else "time_limit_feasible"
    if x is None:
        return _result("mip", [], D, T, w, names, status, gap=gap, runtime_s=time.time() - t0, note=msg)
    idx = [i for i in range(n) if x[i] > 0.5]
    raw = float(-res.fun) if getattr(res, "fun", None) is not None else None
    applied = False
    if status == "optimal":
        if math.comb(n, k_eff) <= tie_break_enum_limit:
            ex = exhaustive_coverage(D, T, w, k_eff, names, max_combinations=tie_break_enum_limit)
            if abs(ex.objective - layout_metrics(D, T, w, idx)["objective"]) <= 1e-7:
                idx, applied = ex.indexes, True
        while len(idx) < k_eff:                    # unused budget: add by the declared tie-break
            rest = [c_ for c_ in range(n) if c_ not in idx]
            idx.append(min(rest, key=lambda c_: _key(idx + [c_], D, T, w, names)))
    return _result("mip", idx, D, T, w, names, status, gap=gap, runtime_s=time.time() - t0,
                   solver_objective=raw, tie_break_applied=applied, note=msg)
