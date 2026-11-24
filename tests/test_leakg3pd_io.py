import sys, tempfile, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leakg3pd_io import audit, labels_match_leaks, load_scenario  # noqa: E402

NODES = dict(junctions=["J1", "J2"], tanks=[], reservoirs=["R1"])


def make_scenario(tmp, labels, step_min=30, extra_cols=None, leaks=(), pressures=None):
    idx = pd.date_range("2024-01-01", periods=len(labels), freq=f"{step_min}min")
    cols = {"J1": 50.0, "J2": 60.0, "R1": 0.0}
    cols.update(extra_cols or {})
    df = pd.DataFrame({c: [v] * len(idx) for c, v in cols.items()}, index=idx)
    if pressures is not None:
        df["J1"] = pressures
    folder = Path(tmp) / "Scenario-3"
    (folder / "Leaks").mkdir(parents=True)
    df.to_csv(folder / "Node_pressures.csv", index_label="Timestamp")
    pd.DataFrame({"Label": np.asarray(labels, float)}, index=idx).to_csv(folder / "Labels.csv", index_label="Timestamp")
    (folder / "Scenario-3_info.csv").write_text("Description , Value\nNetwork_Name , Toy\n")
    for n, (s, e) in enumerate(leaks):
        (folder / "Leaks" / f"Leak_leak_node{n}_info.csv").write_text(
            "Description , Value\nLeak Node , leak_node%d\nLeak Area , 0.001\nLeak Diameter , 0.03\nLeak Type , small\n"
            "Leak Start , %s\nLeak End , %s\n" % (n, idx[s], idx[e]))
    return folder


class TestLoader(unittest.TestCase):
    def test_candidates_exclude_leak_nodes_and_reservoirs(self):
        with tempfile.TemporaryDirectory() as t:
            f = make_scenario(t, [0] * 10, extra_cols={"leak_node0": 40.0})
            s = load_scenario(t, f, NODES)
            self.assertEqual(list(s.pressures.columns), ["J1", "J2"])
            self.assertEqual(sorted(s.extra_columns), ["R1", "leak_node0"])

    def test_events_from_labels_and_overlap_merges(self):
        with tempfile.TemporaryDirectory() as t:
            f = make_scenario(t, [0, 1, 1, 0, 0, 1, 0])
            ev = load_scenario(t, f, NODES).events
            self.assertEqual([(e.start, e.end) for e in ev], [(1, 3), (5, 6)])

    def test_leak_free_scenario(self):
        with tempfile.TemporaryDirectory() as t:
            s = load_scenario(t, make_scenario(t, [0] * 6), NODES)
            self.assertTrue(s.is_leak_free)
            self.assertEqual(s.events, [])

    def test_wrong_time_step_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                load_scenario(t, make_scenario(t, [0] * 6, step_min=60), NODES)

    def test_unrecognised_extra_column_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                load_scenario(t, make_scenario(t, [0] * 6, extra_cols={"MYSTERY": 1.0}), NODES)

    def test_labels_match_leak_files(self):
        with tempfile.TemporaryDirectory() as t:
            f = make_scenario(t, [0, 1, 1, 1, 0], leaks=[(1, 3)])
            self.assertTrue(labels_match_leaks(load_scenario(t, f, NODES)))

    def test_label_leak_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as t:
            f = make_scenario(t, [0, 1, 1, 0, 0], leaks=[(1, 3)])
            self.assertFalse(labels_match_leaks(load_scenario(t, f, NODES)))

    def test_audit_counts_negatives_and_over_range(self):
        with tempfile.TemporaryDirectory() as t:
            f = make_scenario(t, [0] * 4, pressures=[50.0, -1.0, 80.0, 50.0])
            a = audit(load_scenario(t, f, NODES), device_max_kpa=689)
            self.assertEqual(a["negative_values"], 1)
            self.assertEqual(a["candidates_over_range"], 1)      # 80 m = 784.5 kPa
            self.assertEqual(a["over_range_nodes"], "J1")


if __name__ == "__main__":
    unittest.main()
