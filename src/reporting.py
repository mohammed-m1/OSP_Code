"""Tables, plots, assumptions register and run metadata."""
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                         # save pictures to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr

from cost_func import EXCLUDED_COST_ITEMS


def assumptions_register(exp, cfg, device, modified_notes):
    """Every input is labelled sourced / derived / assumed."""
    S = "sourced"; A = "assumed"; D = "derived"
    rows = [
        ("price", "Logger hardware price", f"AUD {device.price_aud_ex_gst} ex GST", S, "provenance P1"),
        ("range", "Logger range", f"{device.range_min_kpa:.0f}-{device.range_max_kpa:.0f} kPa gauge", S, "provenance P2"),
        ("accuracy", "Logger accuracy", f"+/-{device.accuracy_pct_fso}% FSO = {device.accuracy_kpa:.2f} kPa = {device.accuracy_m_head:.3f} m", D, "provenance P3"),
        ("interval", "Sampling interval", f"{cfg.report_step_s} s (device allows {device.interval_min_s}-{device.interval_max_s} s)", A, "within published range"),
        ("excluded", "Costs NOT modelled", "; ".join(EXCLUDED_COST_ITEMS), A, "scope decision: purchase cost only"),
        ("candidates", "Candidate sites", "all junctions not excluded by range; installation feasibility UNVERIFIED", A, "hypothetical installation opportunities"),
        ("existing", "Existing loggers", str(exp.existing_nodes or "none"), A, "zero existing sensors"),
        ("leak_nodes", "Leak locations", ",".join(exp.leak_nodes_used), A, "junction leaks"),
        ("leak_sizes", "Equivalent opening diameters", f"{list(exp.diameters_mm)} mm", A, "NOT local measurements"),
        ("leak_start", "Leak start times", f"{list(exp.start_hours)} h", A, "pattern-aligned"),
        ("leak_duration", "Sustained leak duration", f"{exp.duration_hours} h", A, "one duration"),
        ("discharge", "Leak discharge coefficient", str(cfg.discharge_coeff), A, "WNTR default"),
        ("solver", "Hydraulic solver", f"WNTRSimulator, {cfg.demand_model}", A, "same settings for normal and leak runs"),
        ("tau", "Primary detection threshold", f"{exp.primary_tau_m} m head", A, "experimental setting, not an SA Water standard"),
        ("tau_sens", "Threshold sensitivity", f"{list(exp.tau_sensitivity_m)} m", A, "lower and higher thresholds"),
        ("persistence", "Persistence rule", f"{exp.persistence} consecutive samples", A, "experimental setting"),
        ("targets", "Coverage targets", str(list(exp.targets)), A, "not utility requirements"),
        ("network", "Network", f"{exp.network_name} ({exp.network_path.name})", A, modified_notes),
        ("baseline", "Baseline", "no-leak run of same model", A, "residual = leak run - baseline run"),
        ("claims", "NOT claimed", "leak localisation, failure prevention, detection of all small leaks, measured false alarms, SA Water network behaviour", A, "scope"),
    ]
    return pd.DataFrame(rows, columns=["id", "item", "value", "label", "note"])


def run_metadata(cfg, device, exp, n_ok, n_flagged, n_failed):
    return pd.DataFrame([
        ("python", sys.version.split()[0]), ("platform", platform.platform()),
        ("wntr", wntr.__version__), ("numpy", np.__version__), ("pandas", pd.__version__),
        ("simulator", "wntr.sim.WNTRSimulator"), ("demand_model", cfg.demand_model),
        ("report_step_s", cfg.report_step_s), ("discharge_coeff", cfg.discharge_coeff),
        ("device", device.device_id), ("device_price_accessed", device.accessed),
        ("scenarios_ok", n_ok), ("scenarios_flagged", n_flagged), ("scenarios_failed", n_failed),
        ("random_seed", exp.seed),
    ], columns=["key", "value"])


def plot_pressure_traces(path, baseline, run, scenario, strong, weak, tau, title):
    """Baseline vs leak pressure at one logger, plus residuals at a strong and a weak candidate."""
    t_h = baseline.index / 3600
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t_h, baseline[strong], "--", label=f"no leak ({strong})")
    ax[0].plot(t_h, run["pressure"][strong], label=f"leak at {scenario.node}")
    ax[0].set_ylabel("Pressure (m head)"); ax[0].legend(); ax[0].set_title(title)
    for node, style in [(strong, "-"), (weak, ":")]:
        ax[1].plot(t_h, run["pressure"][node] - baseline[node], style, label=f"residual at {node}")
    ax[1].axhline(-tau, color="grey", lw=.8); ax[1].axhline(tau, color="grey", lw=.8, label=f"+/-{tau} m threshold")
    for a in ax:
        a.axvspan(scenario.start_s / 3600, scenario.end_s / 3600, alpha=.12, color="red")
        a.grid(alpha=.3)
    ax[1].set_xlabel("Time (h)"); ax[1].set_ylabel("Residual (m)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cost_coverage(path, searches, S, price, targets, title):
    """Best achievable coverage for each number of loggers, one line per threshold."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tau, res in searches.items():
        any_res = next(iter(res.values()))
        ks = sorted(any_res["best_n_by_k"])
        ax.plot([k * price for k in ks], [any_res["best_n_by_k"][k] / S * 100 for k in ks], marker="o", label=f"threshold {tau} m")
    for q in targets:
        ax.axhline(q * 100, color="grey", lw=.6, ls="--")
    ax.set_xlabel("Hardware cost (AUD ex GST)"); ax.set_ylabel("Simulated leak scenarios detected (%)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_layout(path, wn, layout, title):
    attr = pd.Series({n: (1.0 if n in layout else 0.0) for n in wn.junction_name_list})
    ax = wntr.graphics.plot_network(wn, node_attribute=attr, node_size=[160 if a else 40 for a in attr],
                                    node_cmap=plt.cm.coolwarm, node_labels=True, title=title, add_colorbar=False)
    plt.savefig(path, dpi=150); plt.close()
