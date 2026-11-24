import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline import frozen_baseline, reconstruction_error, rolling_baseline, warmup_steps  # noqa: E402


def toy(days=60, seed=0):
    idx = pd.date_range("2024-01-01", periods=days * 48, freq="30min")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(50, 1, (len(idx), 2)), index=idx, columns=["A", "B"])


class TestBaselines(unittest.TestCase):
    def test_rolling_is_nan_until_window_plus_guard(self):
        p = toy(); b = rolling_baseline(p, window_days=10, guard_days=3)
        self.assertTrue(b.iloc[:13 * 48].isna().all().all())
        self.assertFalse(b.iloc[13 * 48:].isna().any().any())

    def test_rolling_is_causal(self):
        """Changing data on day 40 must not change any baseline up to and including day 40 + guard."""
        p = toy(); p2 = p.copy(); p2.iloc[40 * 48:41 * 48] += 100
        b1, b2 = rolling_baseline(p, 10, 3), rolling_baseline(p2, 10, 3)
        same_until = (40 + 3 + 1) * 48
        pd.testing.assert_frame_equal(b1.iloc[:same_until], b2.iloc[:same_until])
        self.assertFalse(b1.iloc[same_until:].equals(b2.iloc[same_until:]))

    def test_rolling_ignores_todays_and_guard_data(self):
        p = toy(); p.iloc[30 * 48:] = 0        # sudden change from day 30
        b = rolling_baseline(p, 10, 3)
        self.assertAlmostEqual(float(b.iloc[30 * 48:31 * 48].mean().mean()), 50, delta=1.0)   # still the old level

    def test_frozen_matches_slot_median_and_is_nan_in_training(self):
        p = toy(); b = frozen_baseline(p, train_days=14)
        self.assertTrue(b.iloc[:14 * 48].isna().all().all())
        slot0 = p.iloc[:14 * 48].iloc[::48]["A"].median()
        self.assertAlmostEqual(float(b.iloc[20 * 48]["A"]), float(slot0))

    def test_frozen_does_not_move_after_training(self):
        p = toy(); p2 = p.copy(); p2.iloc[30 * 48:] += 5
        pd.testing.assert_frame_equal(frozen_baseline(p, 14), frozen_baseline(p2, 14))

    def test_reconstruction_error_counts_only_non_leak_steps_and_flags_divergence(self):
        from types import SimpleNamespace
        p = toy(days=2); base = p.copy()
        base.iloc[10, 0] += 0.5                    # a mismatch on a non-leak step
        base.iloc[50, 0] += 9.0                    # a big difference on a LEAK step (must be ignored)
        labels = np.zeros(len(p), bool); labels[50] = True
        r = reconstruction_error(SimpleNamespace(pressures=p, labels=labels), base)
        self.assertEqual(r["steps_over_0p01m"], 1)
        self.assertAlmostEqual(r["max_abs_error_m"], 0.5)
        self.assertEqual(r["steps_compared"], len(p) - 1)

    def test_warmup_steps(self):
        self.assertEqual(warmup_steps("frozen", 56, 28, 7), 56 * 48)
        self.assertEqual(warmup_steps("rolling", 56, 28, 7), 35 * 48)


if __name__ == "__main__":
    unittest.main()
