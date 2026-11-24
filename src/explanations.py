"""
EXPLANATIONS built only from calculated quantities (no unsupported claims such as 'high-risk area').
"""
import math

import numpy as np
import pandas as pd

from placement_search import layout_stats, natural_key


def explain_layout(result, D, T, candidates, eligible_idx, existing_idx, unit_price, scenario_ids):
    """Return (per-sensor DataFrame, list of layout-level sentences) for one feasible target result."""
    q, S = result["target"], result["total_scenarios"]
    needed = math.ceil(q * S - 1e-9)
    layout_idx, existing_idx = list(result["layout_idx"]), list(existing_idx)
    base = layout_stats(existing_idx + layout_idx, D, T)
    rows = []
    for i in sorted(layout_idx, key=lambda i: natural_key(candidates[i])):
        others = [j for j in layout_idx if j != i]
        without = layout_stats(existing_idx + others, D, T)
        rest = existing_idx + others
        other_cover = D[:, rest].any(axis=1) if rest else np.zeros(S, dtype=bool)
        unique_mask = D[:, i] & ~other_cover               # detected here and by no other logger in the layout
        # in how many scenarios does this logger alarm first (or tied first)?
        cols = existing_idx + layout_idx
        first_count = 0
        for s in range(S):
            if D[s, i]:
                times = [T[s, j] for j in cols if D[s, j]]
                if T[s, i] <= min(times) + 1e-9:
                    first_count += 1
        # equal-cost replacements: swap this logger for any other eligible candidate
        swaps = []
        for j in eligible_idx:
            if j in layout_idx:
                continue
            new = existing_idx + others + [j]
            st = layout_stats(new, D, T)
            lost = [scenario_ids[s] for s in range(S) if base["detected_mask"][s] and not st["detected_mask"][s]]
            gained = [scenario_ids[s] for s in range(S) if st["detected_mask"][s] and not base["detected_mask"][s]]
            swaps.append(dict(node=candidates[j], n_detected=st["n_detected"], coverage=st["coverage"],
                              meets_target=st["n_detected"] >= needed, lost=lost, gained=gained,
                              mean_delay_min=st["mean_delay_min"]))
        valid = [s["node"] for s in swaps if s["meets_target"]]
        best = sorted(swaps, key=lambda s: (-s["n_detected"], s["mean_delay_min"], natural_key(s["node"])))[0] if swaps else None
        below = without["n_detected"] < needed
        node = candidates[i]
        if below:
            reason = (f"Required: removing {node} reduces coverage from {base['coverage']:.1%} to "
                      f"{without['coverage']:.1%}, below the {q:.0%} target; it alone covers {int(unique_mask.sum())} scenario(s).")
        else:
            reason = (f"Not individually required: coverage stays {without['coverage']:.1%} without {node}, "
                      f"still meeting the {q:.0%} target.")
        if valid:
            swap_txt = "Equal-cost replacements that also meet the target: " + ", ".join(sorted(valid, key=natural_key)) + "."
        elif best:
            swap_txt = (f"No single equal-cost replacement meets the target. Best replacement {best['node']} gives "
                        f"{best['coverage']:.1%}, losing {len(best['lost'])} and gaining {len(best['gained'])} scenario(s).")
        else:
            swap_txt = "No other candidate available."
        own_delays = T[D[:, i], i]
        rows.append(dict(
            selected_location=node, hardware_cost_aud_ex_gst=unit_price,
            scenarios_observable_here=int(D[:, i].sum()), scenarios_uniquely_covered=int(unique_mask.sum()),
            first_to_alarm_scenarios=first_count, layout_coverage=round(base["coverage"], 4),
            coverage_if_removed=round(without["coverage"], 4), below_target_if_removed=bool(below),
            mean_detection_delay_min_here=round(float(own_delays.mean()), 1) if len(own_delays) else float("nan"),
            equal_cost_replacements_meeting_target=";".join(sorted(valid, key=natural_key)),
            replacement_note=swap_txt, reason_retained=reason))
    table = pd.DataFrame(rows)

    k = result["k"]
    missed = [scenario_ids[i] for i in result["missed_idx"]]
    sentences = [
        f"This layout meets the {q:.0%} scenario-coverage target ({result['n_detected']}/{S} scenarios, {result['coverage']:.1%}) "
        f"using {k} reference logger(s) at AUD {result['cost_aud_ex_gst']:,.2f} excluding GST (hardware only).",
        f"Fewer loggers cannot meet it: the best {k - 1}-logger layout reaches {result['best_coverage_with_fewer']:.1%}."
        if k > 1 else "A single logger is the smallest possible layout.",
        f"{result['n_equal_cost_feasible']} equal-cost layout(s) meet the target; the chosen one wins the declared tie-breaks "
        f"(highest coverage, then shortest mean delay of {result['mean_delay_min']:.0f} min, then node-ID order).",
        f"{len(missed)} scenario(s) are missed by this layout" + (f": {', '.join(missed[:12])}{' ...' if len(missed) > 12 else ''}." if missed else "."),
    ]
    return table, sentences
