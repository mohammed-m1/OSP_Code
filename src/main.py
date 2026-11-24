"""
EXPLAINABLE, PURCHASE-COST-AWARE PLACEMENT OF PRESSURE LOGGERS FOR SIMULATED LEAKS
South Australia-informed BENCHMARK study (not a model of SA Water's network).

Run (from the project folder):
    python src/main.py                         # tiny debugging network
    python src/main.py --network Net1          # first benchmark (WNTR's bundled Net1.inp)
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wntr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_func import PressureLoggerCostFunction, load_device_catalogue
from detection import build_detection_table, range_check
from explanations import explain_layout
from placement_search import adjacency_baseline, natural_key, random_layout_baseline, search_layouts, to_matrices
from reporting import (assumptions_register, plot_cost_coverage, plot_layout, plot_pressure_traces, run_metadata)
from scenario_runner import SimConfig, build_scenarios, load_model, run_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    network_name: str
    network_path: Path
    device_id: str = "EFENTO_BT_689KPA"
    existing_nodes: tuple = ()                 # zero existing sensors
    leak_nodes: tuple = None                   # None = every junction
    diameters_mm: tuple = (5, 10, 20)          # ASSUMED opening sizes
    start_hours: tuple = (6, 15)               # ASSUMED start times
    duration_hours: float = 6                  # ASSUMED sustained-event duration
    primary_tau_m: float = 1.0                 # ASSUMED primary threshold (m head)
    tau_sensitivity_m: tuple = (0.25, 0.5, 1.0, 2.0)
    persistence: int = 3                       # consecutive samples
    targets: tuple = (0.5, 0.75, 0.9, 1.0)
    random_samples: int = 1000
    seed: int = 42
    leak_nodes_used: tuple = ()


def resolve_network(name):
    if name.lower() in ("tiny", "tiny_network"):
        return "tiny", PROJECT_ROOT / "network" / "tiny_network.inp"
    lib = Path(wntr.__path__[0]) / "library" / "networks" / f"{name}.inp"
    if lib.is_file():
        return name, lib
    p = Path(name)
    if p.is_file():
        return p.stem, p
    raise FileNotFoundError(f"Unknown network: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="tiny")
    ap.add_argument("--out", default=None)
    ap.add_argument("--existing", nargs="*", default=[], help="junctions that ALREADY have a logger (default: none)")
    args = ap.parse_args()
    t0 = time.time()

    name, path = resolve_network(args.network)
    exp = ExperimentConfig(network_name=name, network_path=path, existing_nodes=tuple(args.existing))
    cfg = SimConfig()
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)

    # ---------- devices and cost ----------
    catalogue = load_device_catalogue(PROJECT_ROOT / "data" / "devices" / "pressure_logger_catalog.csv")
    device = catalogue[exp.device_id]
    cost_fn = PressureLoggerCostFunction(device, exp.existing_nodes)
    price = device.price_aud_ex_gst
    print(f"Device: {device.product}\n  price AUD {price} ex GST | range {device.range_min_kpa:.0f}-{device.range_max_kpa:.0f} kPa "
          f"| accuracy {device.accuracy_kpa:.2f} kPa ({device.accuracy_m_head:.3f} m head)")
    if not (device.interval_min_s <= cfg.report_step_s <= device.interval_max_s):
        raise ValueError("Sampling interval outside the device's configurable range.")
    print(f"  memory at {cfg.report_step_s // 60}-minute sampling: {device.memory_samples * cfg.report_step_s / 86400:.0f} days")

    # ---------- network and candidates ----------
    wn0 = load_model(path, cfg)
    junctions = sorted(wn0.junction_name_list, key=natural_key)
    leak_nodes = tuple(exp.leak_nodes or junctions)
    exp.leak_nodes_used = leak_nodes
    bad = set(exp.existing_nodes) - set(junctions)
    if bad:
        raise ValueError(f"Existing-logger nodes not in network: {sorted(bad)}")
    candidates = list(junctions)                                          # every junction has a detection column
    new_candidates = [j for j in junctions if j not in exp.existing_nodes]   # only these can be bought
    original = wntr.network.WaterNetworkModel(str(path)).options.time
    modified_notes = (f"time steps set to {cfg.report_step_s} s (file had hydraulic {original.hydraulic_timestep} s, "
                      f"report {original.report_timestep} s); demands and controls unchanged by code")
    print(f"\nNetwork {name}: {len(junctions)} junctions, {len(wn0.pipe_name_list)} pipes, "
          f"{len(wn0.reservoir_name_list)} reservoirs, {len(wn0.tank_name_list)} tanks")

    # ---------- baseline and leak simulations ----------
    base = run_simulation(path, cfg)
    if base["pressure"] is None:
        raise RuntimeError(f"Baseline simulation failed: {base['message']}")
    if base["flags"]:
        print("WARNING baseline flags:", base["flags"])
    scenarios = build_scenarios(leak_nodes, exp.diameters_mm, exp.start_hours, exp.duration_hours)
    print(f"Running {len(scenarios)} leak scenarios (locations x sizes x start times)...")
    runs = {sc.scenario_id: run_simulation(path, cfg, sc) for sc in scenarios}
    n_ok = sum(r["status"] == "ok" for r in runs.values())
    n_flag = sum(r["status"] == "flagged" for r in runs.values())
    n_fail = sum(r["status"] == "failed" for r in runs.values())
    print(f"  ok {n_ok} | flagged (e.g. negative pressure) {n_flag} | failed {n_fail}  (failed ones count as NOT detected)")

    scen_table = pd.DataFrame([dict(
        scenario_id=s.scenario_id, leak_node=s.node, diameter_mm=s.diameter_mm, area_m2=round(s.area_m2, 8),
        start_h=s.start_s / 3600, duration_h=s.duration_s / 3600, status=runs[s.scenario_id]["status"],
        flags=";".join(runs[s.scenario_id]["flags"]), message=runs[s.scenario_id]["message"],
        peak_leak_discharge_lps=round(runs[s.scenario_id]["peak_leak_lps"], 3)) for s in scenarios])
    scen_table.to_csv(out / "scenarios.csv", index=False)

    # ---------- range check (mandatory) ----------
    rc = range_check(base["pressure"], runs, new_candidates, device)
    rc.to_csv(out / "range_check.csv", index=False)
    excluded = rc[~rc["eligible"]]
    if len(excluded):
        print("EXCLUDED by device range:\n", excluded.to_string(index=False))
        print("  -> for these sites request an appropriately rated, SOURCED-price alternative; no price is assumed.")
    eligible = [c for c in new_candidates if bool(rc.loc[rc.candidate_id == c, "eligible"].iloc[0])]
    print(f"Eligible candidates: {len(eligible)} of {len(new_candidates)}  ({max(rc.max_kpa):.0f} kPa max simulated)")

    # ---------- detection and placement for every threshold ----------
    sids = [s.scenario_id for s in scenarios]
    elig_idx = [candidates.index(c) for c in eligible]
    exist_idx = [candidates.index(n) for n in exp.existing_nodes]   # existing loggers: free, always in the layout
    searches, det_tables = {}, {}
    for tau in exp.tau_sensitivity_m:
        det = build_detection_table(base["pressure"], scenarios, runs, candidates, tau, exp.persistence)
        det_tables[tau] = det
        D, T = to_matrices(det, sids, candidates)
        searches[tau] = (search_layouts(D, T, candidates, elig_idx, exist_idx, price, exp.targets), D, T)
    det_tables[exp.primary_tau_m].to_csv(out / "detection_matrix.csv", index=False)

    rows = []
    for tau, (res, D, T) in searches.items():
        for q, r in res.items():
            rows.append(dict(threshold_m=tau, target=q, status=r["status"],
                             loggers=r.get("k"), hardware_cost_aud_ex_gst=r.get("cost_aud_ex_gst"),
                             coverage=r.get("coverage"), layout=" ".join(r.get("layout", [])),
                             mean_delay_min=r.get("mean_delay_min"),
                             missed=len(r["missed_idx"]) if r["status"] == "feasible" else None,
                             max_possible_coverage=r["max_possible_coverage"],
                             n_equal_cost_feasible=r.get("n_equal_cost_feasible")))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "placement_results.csv", index=False)
    print("\nPLACEMENT RESULTS (threshold sensitivity)")
    print(summary.round(3).to_string(index=False))
    for tau, (res, _, _) in searches.items():
        for q, r in res.items():
            if r["status"] == "infeasible":
                print(f"  threshold {tau} m, target {q:.0%}: {r['message']} (best possible {r['max_possible_coverage']:.1%})")

    # ---------- explanations and random-layout comparison, for every threshold that has a feasible target ----------
    all_expl = []
    for tau, (res_t, D_t, T_t) in searches.items():
        feasible = [(q, r) for q, r in res_t.items() if r["status"] == "feasible"]
        tag = " (PRIMARY)" if tau == exp.primary_tau_m else ""
        print(f"\nEXPLANATIONS: threshold {tau} m{tag}, {exp.persistence} consecutive {cfg.report_step_s // 60}-minute samples")
        if not feasible:
            print("   no coverage target is feasible at this threshold")
        for q, r in feasible:
            if r["k"] == 0:
                print(f"\n -- target {q:.0%}: the existing logger(s) already reach {r['coverage']:.1%}; no purchase needed (AUD 0).")
                continue
            table, sentences = explain_layout(r, D_t, T_t, candidates, elig_idx, exist_idx, price, sids)
            table.insert(0, "target", q)
            table.insert(0, "threshold_m", tau)
            all_expl.append(table)
            print(f"\n -- target {q:.0%}: layout {r['layout']}")
            for line in sentences:
                print("    " + line)
            print(table[["selected_location", "scenarios_observable_here", "scenarios_uniquely_covered",
                         "coverage_if_removed", "hardware_cost_aud_ex_gst"]].to_string(index=False))
            print("    " + table["reason_retained"].iloc[0])
            rnd = random_layout_baseline(D_t, T_t, elig_idx, exist_idx, r["k"], q, exp.random_samples, exp.seed)
            print(f"    random {r['k']}-logger layouts: mean coverage {rnd['mean_coverage']:.1%}, best {rnd['max_coverage']:.1%}, "
                  f"{rnd['share_meeting_target']:.0%} meet the target ({'all' if rnd['exhaustive'] else 'sampled'} {rnd['n_layouts']})")
    if all_expl:
        pd.concat(all_expl).to_csv(out / "explanations.csv", index=False)
    res, D, T = searches[exp.primary_tau_m]

    adj = adjacency_baseline(wn0, candidates, elig_idx, D, T)
    if adj:
        print(f"\nOptional old one-hop adjacency baseline: {adj['k']} loggers {adj['layout']} -> "
              f"leak coverage {adj['coverage']:.1%} (adjacency does not guarantee observability)")

    # ---------- plots and registers ----------
    ex = next((s for s in scenarios if s.diameter_mm == 10 and s.start_s == exp.start_hours[0] * 3600
               and s.node == leak_nodes[len(leak_nodes) // 2]), scenarios[0])
    mx = det_tables[exp.primary_tau_m].query("scenario_id == @ex.scenario_id").set_index("candidate_id")["max_residual"]
    if runs[ex.scenario_id]["pressure"] is not None and len(mx.dropna()):
        plot_pressure_traces(out / "pressure_residual_example.png", base["pressure"], runs[ex.scenario_id], ex,
                             mx.idxmax(), mx.idxmin(), exp.primary_tau_m,
                             f"{name}: {ex.diameter_mm:.0f} mm leak at {ex.node}, scenario {ex.scenario_id} (peak {runs[ex.scenario_id]['peak_leak_lps']:.2f} L/s)")
    plot_cost_coverage(out / "cost_vs_coverage.png", {t: v[0] for t, v in searches.items()}, len(sids), price,
                       exp.targets, f"{name}: hardware cost vs simulated leak coverage")
    chosen = None                                       # layout map: strictest feasible target, lowest threshold first
    for tau, (res_t, _, _) in sorted(searches.items()):
        feas = [(q, r) for q, r in res_t.items() if r["status"] == "feasible" and r["k"] > 0]
        if feas:
            q, r = max(feas, key=lambda qr: qr[0])
            chosen = (tau, q, r)
            break
    if chosen:
        tau, q, r = chosen
        plot_layout(out / "selected_layout.png", wn0, r["layout"], f"{name}: layout for {q:.0%} target (threshold {tau} m)")
    assumptions_register(exp, cfg, device, modified_notes).to_csv(out / "assumptions_register.csv", index=False)
    run_metadata(cfg, device, exp, n_ok, n_flag, n_fail).to_csv(out / "run_metadata.csv", index=False)
    print(f"\nSaved results to {out}  ({time.time() - t0:.0f}s)")
    print("Scope: simulated pressure-response observability on a benchmark with a known baseline; "
          "NOT validated real-world leak detection, and not a model of SA Water's network.")


if __name__ == "__main__":
    main()
