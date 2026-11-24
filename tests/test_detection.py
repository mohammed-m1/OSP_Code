import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cost_func import load_device_catalogue  # noqa: E402
from detection import detect_scenario, range_check  # noqa: E402


def series(values, step=300):
    return pd.Series(values, index=[i * step for i in range(len(values))], dtype=float)


class TestPersistenceRule(unittest.TestCase):
    def test_two_exceedances_are_not_enough(self):
        ok, t, _ = detect_scenario(series([0, 0, 2, 2, 0, 0]), 0, 1800, tau_m=1.0, persistence=3)
        self.assertFalse(ok)

    def test_three_consecutive_detected_at_third_sample(self):
        ok, t, mx = detect_scenario(series([0, 2, 2, 2, 0]), 0, 1500, tau_m=1.0, persistence=3)
        self.assertTrue(ok)
        self.assertEqual(t, 900)            # third exceeding sample is at index 3 -> 900 s
        self.assertEqual(mx, 2.0)

    def test_interrupted_run_resets(self):
        ok, _, _ = detect_scenario(series([2, 2, 0, 2, 2, 0]), 0, 1800, tau_m=1.0, persistence=3)
        self.assertFalse(ok)

    def test_exceedance_before_onset_is_ignored(self):
        ok, _, _ = detect_scenario(series([2, 2, 2, 0, 0, 0]), 900, 1800, tau_m=1.0, persistence=3)
        self.assertFalse(ok)

    def test_negative_residual_counts(self):
        ok, _, _ = detect_scenario(series([0, -2, -2, -2, 0]), 0, 1500, tau_m=1.0, persistence=3)
        self.assertTrue(ok)

    def test_threshold_is_strict(self):
        ok, _, _ = detect_scenario(series([1, 1, 1, 1]), 0, 1200, tau_m=1.0, persistence=3)
        self.assertFalse(ok)


class TestRangeCheck(unittest.TestCase):
    def test_high_pressure_candidate_is_excluded(self):
        device = load_device_catalogue(ROOT / "data" / "devices" / "pressure_logger_catalog.csv")["EFENTO_BT_689KPA"]
        base = pd.DataFrame({"A": [30.0, 31.0], "B": [80.0, 81.0]}, index=[0, 300])   # 80 m = 785 kPa
        rc = range_check(base, {}, ["A", "B"], device).set_index("candidate_id")
        self.assertTrue(rc.loc["A", "eligible"])
        self.assertFalse(rc.loc["B", "eligible"])


if __name__ == "__main__":
    unittest.main()
