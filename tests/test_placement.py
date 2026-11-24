import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from placement_search import INFEASIBLE_MSG, MAX_ENUMERATION_CANDIDATES, search_layouts  # noqa: E402

NAN = np.nan


class TestPlacement(unittest.TestCase):
    def setUp(self):
        # 4 scenarios, candidates A, B, C.  A sees s0,s1   B sees s2,s3   C sees s1,s2
        self.cands = ["A", "B", "C"]
        self.D = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=bool)
        self.T = np.array([[10, NAN, NAN], [10, NAN, 5], [NAN, 10, 5], [NAN, 10, NAN]], dtype=float)

    def test_cheapest_layout_for_half_coverage_is_one_logger(self):
        r = search_layouts(self.D, self.T, self.cands, [0, 1, 2], [], 726.36, [0.5])[0.5]
        self.assertEqual(r["status"], "feasible")
        self.assertEqual(r["k"], 1)
        self.assertEqual(r["cost_aud_ex_gst"], 726.36)

    def test_full_coverage_needs_two_loggers(self):
        r = search_layouts(self.D, self.T, self.cands, [0, 1, 2], [], 726.36, [1.0])[1.0]
        self.assertEqual(r["k"], 2)
        self.assertAlmostEqual(r["cost_aud_ex_gst"], 1452.72, places=2)
        self.assertEqual(r["layout"], ["A", "B"])

    def test_undetectable_scenarios_stay_in_denominator(self):
        D = self.D.copy(); D[3, :] = False          # nobody can see s3
        r = search_layouts(D, self.T, self.cands, [0, 1, 2], [], 726.36, [1.0, 0.75])
        self.assertEqual(r[1.0]["status"], "infeasible")
        self.assertEqual(r[1.0]["message"], INFEASIBLE_MSG)
        self.assertAlmostEqual(r[1.0]["max_possible_coverage"], 0.75)
        self.assertEqual(r[0.75]["total_scenarios"], 4)

    def test_tie_break_is_deterministic_and_prefers_faster_detection(self):
        # B and C each see 2 scenarios; C is faster (5 min) so C wins the 50% single-logger tie
        D = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
        T = np.array([[NAN, 10, NAN], [NAN, 10, NAN], [NAN, NAN, 5], [NAN, NAN, 5]], dtype=float)
        r = search_layouts(D, T, self.cands, [0, 1, 2], [], 726.36, [0.5])[0.5]
        self.assertEqual(r["layout"], ["C"])
        self.assertEqual(r["n_equal_cost_feasible"], 2)

    def test_natural_node_order_breaks_remaining_ties(self):
        cands = ["J10", "J2"]
        D = np.array([[1, 1], [1, 1]], dtype=bool)
        T = np.array([[5, 5], [5, 5]], dtype=float)
        r = search_layouts(D, T, cands, [0, 1], [], 100.0, [1.0])[1.0]
        self.assertEqual(r["layout"], ["J2"])        # J2 before J10

    def test_existing_logger_can_make_purchase_unnecessary(self):
        # existing logger at A already sees 2 of 4 scenarios: the 50% target needs ZERO new loggers
        r = search_layouts(self.D, self.T, self.cands, [1, 2], [0], 726.36, [0.5, 1.0])
        self.assertEqual(r[0.5]["k"], 0)
        self.assertEqual(r[0.5]["cost_aud_ex_gst"], 0.0)
        self.assertEqual(r[1.0]["k"], 1)             # B completes the coverage
        self.assertEqual(r[1.0]["layout"], ["B"])

    def test_too_many_candidates_is_refused(self):
        n = MAX_ENUMERATION_CANDIDATES + 1
        with self.assertRaises(ValueError):
            search_layouts(np.zeros((2, n), bool), np.full((2, n), NAN), [f"N{i}" for i in range(n)],
                           list(range(n)), [], 1.0, [0.5])


if __name__ == "__main__":
    unittest.main()
