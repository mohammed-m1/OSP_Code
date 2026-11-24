import inspect, sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import detector  # noqa: E402
from detector import (alarm_onsets, alarm_state, calibrate_thresholds, drop_residual, false_alarm_rate,  # noqa: E402
                      rise_check, run_length, score_events)
from leakg3pd_io import Event  # noqa: E402


def frame(values):
    return pd.DataFrame({"A": values}, index=pd.date_range("2024-01-01", periods=len(values), freq="30min"), dtype=float)


class TestDetector(unittest.TestCase):
    def test_drop_sign_positive_when_pressure_is_lower(self):
        self.assertEqual(float(drop_residual(frame([10]), frame([8])).iloc[0, 0]), 2.0)

    def test_run_length(self):
        self.assertEqual(list(run_length(np.array([1, 1, 0, 1, 1, 1], bool))), [1, 2, 0, 1, 2, 3])

    def test_persistence_requires_consecutive_exceedances(self):
        d = frame([0, 5, 5, 0, 5, 5, 5, 0])
        st = alarm_state(d, 1.0, 3)["A"].tolist()
        self.assertEqual(st, [False] * 6 + [True, False])           # only the run of three completes
        self.assertEqual(alarm_onsets(alarm_state(d, 1.0, 3))["A"].sum(), 1)

    def test_threshold_is_strict(self):
        self.assertFalse(alarm_state(frame([1, 1, 1]), 1.0, 1)["A"].any())

    def test_nan_residual_never_alarms(self):
        self.assertFalse(alarm_state(frame([np.nan] * 5), 0.1, 1)["A"].any())

    def test_rises_do_not_alarm(self):
        self.assertFalse(alarm_state(frame([-5, -5, -5]), 1.0, 1)["A"].any())

    def test_threshold_quantile_and_accuracy_floor(self):
        d = pd.DataFrame({"A": np.linspace(0, 10, 1001), "B": np.zeros(1001)})
        tau = calibrate_thresholds(d, 0.9, accuracy_m=0.2)
        self.assertAlmostEqual(float(tau["A"]), 9.0, places=2)
        self.assertEqual(float(tau["B"]), 0.2)                       # floor applies where normal spread is tiny

    def test_detector_functions_take_no_labels(self):
        for fn in (alarm_state, alarm_onsets, calibrate_thresholds, drop_residual):
            self.assertNotIn("label", " ".join(inspect.signature(fn).parameters))

    def test_alarm_state_identical_whatever_the_events(self):
        d = frame([0, 5, 5, 5, 0, 0])
        a = alarm_state(d, 1.0, 2)
        self.assertTrue(a.equals(alarm_state(d, 1.0, 2)))
        self.assertTrue(score_events(a, d, [Event(0, 4)], 10).detected.iloc[0] and not score_events(a, d, [Event(4, 6)], 10).detected.iloc[0])

    def test_score_events_delay_and_window(self):
        d = frame([0, 0, 5, 5, 5, 0, 0]); st = alarm_state(d, 1.0, 2)         # alarm starts at step 3
        r = score_events(st, d, [Event(2, 6)], 10).iloc[0]
        self.assertTrue(r.detected); self.assertEqual(r.delay_h, 0.5)           # one step after event start
        self.assertFalse(score_events(st, d, [Event(2, 3)], 10).iloc[0].detected)   # window ends before alarm
        self.assertFalse(score_events(st, d, [Event(2, 6)], 1).iloc[0].detected)    # horizon too short

    def test_alarm_already_running_at_event_start_is_not_counted(self):
        d = frame([5, 5, 5, 5, 5, 0]); st = alarm_state(d, 1.0, 2)
        self.assertFalse(score_events(st, d, [Event(3, 5)], 10).iloc[0].detected)

    def test_false_alarms_counted_only_on_normal_steps(self):
        d = frame([5, 5, 0, 0, 5, 5, 0, 0]); st = alarm_state(d, 1.0, 2)
        mask = np.array([True] * 4 + [False] * 4)
        self.assertAlmostEqual(float(false_alarm_rate(st, mask)["A"]), 1 / (4 / 48))

    def test_rise_check_reports_rises(self):
        base, obs = frame([10, 10, 10]), frame([12, 10, 10])
        rc = rise_check(base, obs, np.array([True, True, True]), 1.0)
        self.assertGreater(rc["share_samples_rise_over_tau"], 0)
        self.assertEqual(rc["max_rise_m"], 2.0)

    def test_end_to_end_synthetic_leak_detected_after_onset_only(self):
        rng = np.random.default_rng(1)
        obs = frame(50 + rng.normal(0, 0.1, 400)); obs.iloc[300:, 0] -= 3.0      # persistent 3 m drop at step 300
        base = frame([50.0] * 400)
        d = drop_residual(base, obs)
        tau = calibrate_thresholds(d.iloc[:250], 0.999, 0.05)
        st = alarm_state(d, tau, 3)
        self.assertFalse(st.iloc[:300]["A"].any())
        self.assertTrue(st.iloc[302:]["A"].all())


if __name__ == "__main__":
    unittest.main()
