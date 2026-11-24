"""
LOADER for LeakG3PD-format scenario folders (the format written by the official generator).

Folder layout expected (file names are listed in FILE_NAMES; change them there if the downloaded archive differs):
    <network folder>/<Net>.inp                 base network (original, un-perturbed junction list)
    <network folder>/Scenario-<k>/Node_pressures.csv, Labels.csv, Scenario-<k>_info.csv,
                                  <Net>_Scenario-<k>.inp, Leaks/Leak_<node>_info.csv

Units (checked against the generator code and data): pressure in metres of head, 30-minute steps.
Candidate sensor sites = the ORIGINAL junctions of the base network. Scenario-specific inserted leak nodes
(names starting 'leak_node') and reservoirs/tanks are never candidates.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STEP_MIN = 30
LEAK_PREFIX = "leak_node"
FILE_NAMES = {"pressure": ["Node_pressures.csv", "Pressures.csv"], "labels": ["Labels.csv"]}


@dataclass(frozen=True)
class Leak:
    node: str
    start: pd.Timestamp
    end: pd.Timestamp          # timestamp of the LAST active step
    area_m2: float
    diameter_m: float
    kind: str


@dataclass(frozen=True)
class Event:
    start: int                 # index of first labelled step
    end: int                   # index one past the last labelled step


@dataclass
class Scenario:
    network: str
    sid: int
    folder: Path
    inp_path: Path
    pressures: pd.DataFrame            # candidates only, metres of head
    labels: np.ndarray                 # bool per step
    leaks: list
    info: dict
    extra_columns: list = field(default_factory=list)

    @property
    def is_leak_free(self) -> bool:
        return not self.labels.any()

    @property
    def events(self) -> list:
        """Maximal contiguous runs of label == 1. Overlapping leaks merge into one event."""
        padded = np.concatenate([[False], self.labels, [False]])
        d = np.diff(padded.astype(int))
        return [Event(int(s), int(e)) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def parse_kv_csv(path) -> dict:
    """Read the generator's 'Description , Value' files."""
    out = {}
    for line in Path(path).read_text().splitlines()[1:]:
        if "," in line:
            k, v = line.split(",", 1)
            out[k.strip()] = v.strip()
    return out


def _first_existing(folder: Path, names) -> Path:
    for n in names:
        if (folder / n).is_file():
            return folder / n
    raise FileNotFoundError(f"None of {names} found in {folder}")


def base_network_nodes(net_dir) -> dict:
    """Junction / tank / reservoir names of the base network (the stable, original node IDs)."""
    import wntr
    inps = sorted(Path(net_dir).glob("*.inp"))
    if not inps:
        raise FileNotFoundError(f"No base .inp directly inside {net_dir}")
    wn = wntr.network.WaterNetworkModel(str(inps[0]))
    return dict(junctions=list(wn.junction_name_list), tanks=list(wn.tank_name_list),
                reservoirs=list(wn.reservoir_name_list), inp=inps[0])


def load_scenario(net_dir, scenario_dir, nodes=None) -> Scenario:
    net_dir, scenario_dir = Path(net_dir), Path(scenario_dir)
    nodes = nodes or base_network_nodes(net_dir)
    sid = int(re.findall(r"(\d+)$", scenario_dir.name)[0])

    raw = pd.read_csv(_first_existing(scenario_dir, FILE_NAMES["pressure"]), index_col=0, parse_dates=True)
    step = (raw.index[1] - raw.index[0]).total_seconds() / 60
    if step != STEP_MIN:
        raise ValueError(f"Expected {STEP_MIN}-minute steps, found {step}")
    missing = [j for j in nodes["junctions"] if j not in raw.columns]
    if missing:
        raise ValueError(f"Candidate junctions missing from pressure file: {missing[:5]}")
    extra = [c for c in raw.columns if c not in nodes["junctions"]]
    unknown = [c for c in extra if not c.startswith(LEAK_PREFIX)
               and c not in nodes["tanks"] and c not in nodes["reservoirs"]]
    if unknown:
        raise ValueError(f"Unrecognised extra pressure columns: {unknown[:5]}")

    lab = pd.read_csv(_first_existing(scenario_dir, FILE_NAMES["labels"]), index_col=0, parse_dates=True).iloc[:, 0]
    if len(lab) != len(raw) or not (lab.index == raw.index).all():
        raise ValueError("Labels and pressures are not aligned in time")

    leaks = []
    for f in sorted((scenario_dir / "Leaks").glob("Leak_*_info.csv")) if (scenario_dir / "Leaks").is_dir() else []:
        d = parse_kv_csv(f)
        leaks.append(Leak(d["Leak Node"], pd.Timestamp(d["Leak Start"]), pd.Timestamp(d["Leak End"]),
                          float(d["Leak Area"]), float(d["Leak Diameter"]), d["Leak Type"]))
    info_files = list(scenario_dir.glob("*_info.csv"))
    info = parse_kv_csv(info_files[0]) if info_files else {}
    inp = next(iter(sorted(scenario_dir.glob("*.inp"))), None)
    return Scenario(network=info.get("Network_Name", net_dir.name), sid=sid, folder=scenario_dir, inp_path=inp,
                    pressures=raw[nodes["junctions"]].astype(float), labels=lab.to_numpy() > 0.5,
                    leaks=leaks, info=info, extra_columns=extra)


def load_all(net_dir) -> list:
    net_dir = Path(net_dir)
    nodes = base_network_nodes(net_dir)
    folders = sorted((p for p in net_dir.glob("Scenario-*") if (p / "Labels.csv").is_file()),
                     key=lambda p: int(re.findall(r"(\d+)$", p.name)[0]))
    return [load_scenario(net_dir, p, nodes) for p in folders]


def labels_match_leaks(scn: Scenario) -> bool:
    """Do the 0/1 labels equal the union of the leak intervals listed in the leak info files?"""
    idx = scn.pressures.index
    mask = np.zeros(len(idx), dtype=bool)
    for lk in scn.leaks:
        mask |= (idx >= lk.start) & (idx <= lk.end)
    return bool((mask == scn.labels).all())


def audit(scn: Scenario, device_max_kpa: float = None, kpa_per_m: float = 9.80665) -> dict:
    """Step-1 checks: units, timing, candidate identity, labels, hydraulic validity, device range."""
    p = scn.pressures
    out = dict(network=scn.network, scenario=scn.sid, steps=len(p), days=round(len(p) / 48, 1),
               candidates=p.shape[1], extra_columns=len(scn.extra_columns),
               nan_values=int(p.isna().sum().sum()), negative_values=int((p < 0).sum().sum()),
               min_m=round(float(p.min().min()), 2), max_m=round(float(p.max().max()), 2),
               max_kpa=round(float(p.max().max()) * kpa_per_m, 1),
               leak_free=scn.is_leak_free, n_leaks=len(scn.leaks), n_events=len(scn.events),
               labels_match_leaks=labels_match_leaks(scn),
               leak_kinds=",".join(sorted({lk.kind for lk in scn.leaks})),
               onset_day=round(scn.events[0].start / 48, 1) if scn.events else None,
               event_days=round(sum(e.end - e.start for e in scn.events) / 48, 1))
    if device_max_kpa is not None:
        over = (p.max() * kpa_per_m) > device_max_kpa
        out["candidates_over_range"] = int(over.sum())
        out["over_range_nodes"] = ",".join(over.index[over])
    return out
