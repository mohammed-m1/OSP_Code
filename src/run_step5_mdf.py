"""
STEP 5: minimum detectable flow on the BASE network (one fixed operating condition, demand-driven, leak = added demand).

    python src/run_step5_mdf.py --net-dir <folder with Hanoi.inp> --step12-out results/leakg3pd/Hanoi --out results/leakg3pd/Hanoi/final

Two thresholds (metres of pressure drop), each stated as an experimental setting:
  idealised : the logger accuracy (0.176 m)      -> matches the matched-baseline (Level A) analysis
  realistic : the median FROZEN Level B threshold -> matches the estimated-baseline (Level B) analysis
Unresolved pairs (never detected within the flow range) are NEVER turned into finite thresholds here;
they are saved with their status and handled explicitly by the layout comparison.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import min_detectable_flow as mdf
from cost_func import load_device_catalogue

ROOT = Path(__file__).resolve().parents[1]
F_MIN, F_MAX, N_FLOWS = 0.0005, 0.5, 40               # m3/s, geometric grid (ASSUMED range; unresolved if exceeded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-dir", required=True); ap.add_argument("--step12-out", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import wntr
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base_inp = sorted(Path(a.net_dir).glob("*.inp"))[0]
    screen = pd.read_csv(Path(a.step12_out) / "node_range_screen.csv", index_col=0, dtype={0: str})
    nodes = [str(n) for n in screen.index[screen.eligible]]
    settings = json.loads((Path(a.step12_out) / "detector_settings_FROZEN.json").read_text())
    acc = settings["accuracy_floor_m"]; tau_real = float(np.median(list(settings["thresholds_m"].values())))
    factory = lambda: wntr.network.WaterNetworkModel(str(base_inp))
    flows = mdf.geometric_flows(F_MIN, F_MAX, N_FLOWS)
    for name, thr in (("idealised", acc), ("realistic", tau_real)):
        t0 = time.time()
        M = mdf.build_matrix(factory, nodes, nodes, thr, flows, time_s=0, refine_steps=0)
        M["threshold_m"] = thr
        M.to_csv(out / f"mdf_matrix_{name}.csv", index=False)
        s = mdf.summarise(M)
        s.update(threshold_m=round(thr, 3), seconds=round(time.time() - t0), status_counts=M.status.value_counts().to_dict())
        pd.Series(s).to_json(out / f"mdf_summary_{name}.json", indent=2)
        print(name, "threshold %.3f m:" % thr, {k: v for k, v in s.items() if k != "metric_name"})


if __name__ == "__main__":
    main()
