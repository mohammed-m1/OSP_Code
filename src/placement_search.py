"""
PLACEMENT: smallest (cheapest) set of loggers that detects at least a target share of the simulated leaks.

    minimise  c * |S|     subject to     detected scenarios / ALL evaluated scenarios  >=  q

A scenario counts as detected when ANY selected logger detects it.
Scenarios nobody can detect STAY IN THE DENOMINATOR.
Tie-breaks (declared): 1 lowest cost, 2 highest coverage, 3 shortest mean detection delay, 4 node-ID order.
Layouts are enumerated exhaustively, so this only suits small candidate lists.
"""
import itertools
import math
import re

import numpy as np
import pandas as pd

INFEASIBLE_MSG = "Target infeasible with these candidates, device constraints, scenario set and detection rule."
MAX_ENUMERATION_CANDIDATES = 20


def natural_key(name: str):
    """'J10' sorts after 'J2'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def to_matrices(det: pd.DataFrame, scenario_ids, candidates):
    """D[s, c] = detected?   T[s, c] = detection delay in minutes (NaN if not detected)."""
    d = det.pivot(index="scenario_id", columns="candidate_id", values="detected").reindex(index=scenario_ids, columns=candidates)
    t = det.pivot(index="scenario_id", columns="candidate_id", values="detection_delay_min").reindex(index=scenario_ids, columns=candidates)
    return d.fillna(False).to_numpy(dtype=bool), t.to_numpy(dtype=float)


def layout_stats(idx, D, T):
    """Coverage and timing of the loggers at column indexes `idx` (existing loggers included by the caller)."""
    S = D.shape[0]
    if len(idx) == 0:
        return dict(n_detected=0, coverage=0.0, mean_delay_min=float("inf"), detected_mask=np.zeros(S, bool))
    sub_d, sub_t = D[:, idx], T[:, idx]
    mask = sub_d.any(axis=1)
    earliest = np.where(sub_d, sub_t, np.inf).min(axis=1)
    mean_delay = float(earliest[mask].mean()) if mask.any() else float("inf")
    return dict(n_detected=int(mask.sum()), coverage=float(mask.mean()), mean_delay_min=mean_delay, detected_mask=mask)


def search_layouts(D, T, candidates, eligible_idx, existing_idx, unit_price, targets):
    """Solve every coverage target. Returns {target: result dict}."""
    if len(eligible_idx) > MAX_ENUMERATION_CANDIDATES:
        raise ValueError(f"{len(eligible_idx)} candidates is too many for enumeration "
                         f"(limit {MAX_ENUMERATION_CANDIDATES}). Use a subset or an integer-programming solver.")
    S = D.shape[0]
    existing_idx = list(existing_idx)
    # evaluate every layout size once; reuse for all targets
    by_k = {}
    for k in range(0, len(eligible_idx) + 1):             # k = 0 means: existing loggers alone
        rows = []
        for combo in itertools.combinations(eligible_idx, k):
            st = layout_stats(existing_idx + list(combo), D, T)
            rows.append((combo, st))
        by_k[k] = rows
    best_n_by_k = {k: max(st["n_detected"] for _, st in rows) for k, rows in by_k.items()}
    max_possible = best_n_by_k[len(eligible_idx)] / S

    results = {}
    for q in targets:
        needed = math.ceil(q * S - 1e-9)
        found = None
        for k in range(0, len(eligible_idx) + 1):
            feas = [(combo, st) for combo, st in by_k[k] if st["n_detected"] >= needed]
            if feas:
                feas.sort(key=lambda cs: (-cs[1]["n_detected"], cs[1]["mean_delay_min"],
                                          [natural_key(candidates[i]) for i in sorted(cs[0], key=lambda i: natural_key(candidates[i]))]))
                found = (k, feas)
                break
        if found is None:
            results[q] = dict(target=q, status="infeasible", message=INFEASIBLE_MSG,
                              max_possible_coverage=max_possible, best_n_by_k=best_n_by_k, total_scenarios=S)
            continue
        k, feas = found
        combo, st = feas[0]
        layout = sorted((candidates[i] for i in combo), key=natural_key)
        alternatives = [dict(layout=sorted((candidates[i] for i in c), key=natural_key),
                             coverage=s["coverage"], mean_delay_min=s["mean_delay_min"]) for c, s in feas[:10]]
        results[q] = dict(target=q, status="feasible", k=k, layout=layout, layout_idx=list(combo),
                          coverage=st["coverage"], n_detected=st["n_detected"], total_scenarios=S,
                          mean_delay_min=st["mean_delay_min"], cost_aud_ex_gst=round(k * unit_price, 2),
                          missed_idx=[i for i, m in enumerate(st["detected_mask"]) if not m],
                          best_coverage_with_fewer=(best_n_by_k[k - 1] / S if k >= 1 else float('nan')), n_equal_cost_feasible=len(feas),
                          alternatives=alternatives, max_possible_coverage=max_possible, best_n_by_k=best_n_by_k)
    return results


def random_layout_baseline(D, T, eligible_idx, existing_idx, k, target, n_samples=1000, seed=42):
    """Coverage of random layouts with the same number of new loggers."""
    S = D.shape[0]
    total = math.comb(len(eligible_idx), k)
    if total <= n_samples:
        combos, exhaustive = list(itertools.combinations(eligible_idx, k)), True
    else:
        rng = np.random.default_rng(seed)
        combos = [tuple(rng.choice(eligible_idx, k, replace=False)) for _ in range(n_samples)]
        exhaustive = False
    cov = np.array([layout_stats(list(existing_idx) + list(c), D, T)["coverage"] for c in combos])
    return dict(k=k, n_layouts=len(combos), exhaustive=exhaustive, mean_coverage=float(cov.mean()),
                median_coverage=float(np.median(cov)), min_coverage=float(cov.min()), max_coverage=float(cov.max()),
                share_meeting_target=float((cov >= target - 1e-9).mean()))


def adjacency_baseline(wn, candidates, eligible_idx, D, T):
    """OPTIONAL COMPARISON ONLY: the old one-hop rule (every junction has a logger or a neighbour with one)."""
    adj = {j: set() for j in wn.junction_name_list}
    for p in wn.pipe_name_list:
        pipe = wn.get_link(p)
        a, b = pipe.start_node_name, pipe.end_node_name
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    for k in range(1, len(eligible_idx) + 1):
        feas = []
        for combo in itertools.combinations(eligible_idx, k):
            sel = {candidates[i] for i in combo}
            if all(n in sel or adj[n] & sel for n in adj):
                feas.append((combo, layout_stats(list(combo), D, T)))
        if feas:
            feas.sort(key=lambda cs: (-cs[1]["n_detected"], cs[1]["mean_delay_min"]))
            combo, st = feas[0]
            return dict(k=k, layout=sorted((candidates[i] for i in combo), key=natural_key), coverage=st["coverage"])
    return None
