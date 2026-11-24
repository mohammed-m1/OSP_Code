import sys, unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_steps_3_7 import jaccard, to_matrix  # noqa: E402


class TestFinalHelpers(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(jaccard([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(jaccard([1, 2], [3, 4]), 0.0)
        self.assertAlmostEqual(jaccard([1, 2, 3], [2, 3, 4]), 0.5)
        self.assertEqual(jaccard([], []), 1.0)

    def test_to_matrix_orders_rows_and_columns_and_keeps_missing_as_nan(self):
        sc = pd.DataFrame(dict(key=["E1", "E1", "E2"], node=["a", "b", "a"], detected=[True, False, True]))
        m = to_matrix(sc, ["E2", "E1"], ["b", "a"], "detected")
        self.assertTrue(np.isnan(m[0, 0]))                    # E2 has no row for node b
        self.assertEqual(m[0, 1], True); self.assertEqual(m[1, 0], False); self.assertEqual(m[1, 1], True)


if __name__ == "__main__":
    unittest.main()
