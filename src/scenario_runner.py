"""
LEAK SIMULATION with WNTR's native simulator (WNTRSimulator).
Every run loads a FRESH copy of the network. Normal and leak runs differ ONLY by the leak.
Pressures from WNTR are in metres of head.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import wntr


@dataclass(frozen=True)
class SimConfig:
    demand_model: str = "DDA"       # EXPERIMENTAL SETTING, same for normal and leak runs
    report_step_s: int = 300        # hydraulic AND report step: 5 minutes
    discharge_coeff: float = 0.75   # ASSUMED: WNTR's default leak discharge coefficient


@dataclass(frozen=True)
class LeakScenario:
    scenario_id: str
    node: str
    diameter_mm: float      # ASSUMED equivalent circular opening
    start_s: int
    duration_s: int

    @property
    def area_m2(self) -> float:
        return math.pi * (self.diameter_mm / 2000.0) ** 2        # A = pi (d/2)^2, d converted mm -> m

    @property
    def end_s(self) -> int:
        return self.start_s + self.duration_s


def build_scenarios(leak_nodes, diameters_mm, start_hours, duration_hours):
    """Full factorial: every location x size x start time. One leak per simulation."""
    scenarios = []
    for node in leak_nodes:
        for d in diameters_mm:
            for h in start_hours:
                sid = f"S{len(scenarios) + 1:03d}"
                scenarios.append(LeakScenario(sid, node, float(d), int(h * 3600), int(duration_hours * 3600)))
    return scenarios


def load_model(network_path, cfg: SimConfig):
    """Fresh model with the experiment's solver settings applied."""
    wn = wntr.network.WaterNetworkModel(str(Path(network_path)))
    wn.options.hydraulic.demand_model = cfg.demand_model
    wn.options.time.hydraulic_timestep = cfg.report_step_s
    wn.options.time.report_timestep = cfg.report_step_s
    return wn


def run_simulation(network_path, cfg: SimConfig, scenario: LeakScenario = None) -> dict:
    """Run normal (scenario=None) or leak simulation. Failures are RECORDED, never hidden."""
    wn = load_model(network_path, cfg)
    if scenario is not None:
        if scenario.end_s > wn.options.time.duration:
            return dict(status="failed", flags=[], message="leak ends after the simulation ends",
                        pressure=None, leak_lps=None, peak_leak_lps=float("nan"))
        wn.get_node(scenario.node).add_leak(
            wn, area=scenario.area_m2, discharge_coeff=cfg.discharge_coeff,
            start_time=scenario.start_s, end_time=scenario.end_s)
    try:
        results = wntr.sim.WNTRSimulator(wn).run_sim()
    except Exception as exc:                                   # convergence failure etc.
        return dict(status="failed", flags=[], message=f"{type(exc).__name__}: {exc}",
                    pressure=None, leak_lps=None, peak_leak_lps=float("nan"))

    pressure = results.node["pressure"][wn.junction_name_list]  # time (s) x junction, metres
    flags = []
    if pressure.isna().any().any():
        flags.append("nan_pressure")
    if (pressure < 0).any().any():
        flags.append("negative_pressure")                       # invalid for a DDA run: flagged, not hidden
    leak_lps, peak = None, 0.0
    if scenario is not None:
        leak_lps = results.node["leak_demand"][scenario.node] * 1000.0   # m3/s -> L/s
        peak = float(leak_lps.max())
    status = "failed" if "nan_pressure" in flags else ("flagged" if flags else "ok")
    return dict(status=status, flags=flags, message="", pressure=pressure, leak_lps=leak_lps, peak_leak_lps=peak)
