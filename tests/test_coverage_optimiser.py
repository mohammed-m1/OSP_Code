import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import coverage_optimiser as co  # noqa: E402
from coverage_optimiser import (exhaustive_coverage, greedy_coverage, maximise_coverage_mip,  # noqa: E402
                                random_layouts, sensitivity_ranked)

NAN = np.nan


def hand():
    # A sees s0,s1   B sees s2,s3   C sees s1,s2   (same as the main-track placement test)
    D = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=bool)
    return D, ["A", "B", "C"]


def greedy_trap():
    # 6 scenarios. X sees {0,1,2,3}; Y sees {0,1,4}; Z sees {2,3,5}.  k=2.
    # Greedy takes X (4) then adds only 1 more -> 5.  Optimum is Y+Z -> 6.
    sets = {"X": [0, 1, 2, 3], "Y": [0, 1, 4], "Z": [2, 3, 5]}
    D = np.zeros((6, 3), dtype=bool)
    for j, name in enumerate("XYZ"):
        D[sets[name], j] = True
    return D, ["X", "Y", "Z"]


class TestKnownOptimum(unittest.TestCase):
    def test_k1_and_k2_hand_checked(self):
        D, c = hand()
        r1 = maximise_coverage_mip(D, None, 1, 10, candidates=c)
        self.assertEqual(r1.status, "optimal")
        self.assertEqual(r1.n_detected, 2)
        r2 = maximise_coverage_mip(D, None, 2, 10, candidates=c)
        self.assertEqual(r2.status, "optimal")
        self.assertEqual(r2.nodes, ["A", "B"])
        self.assertAlmostEqual(r2.coverage, 1.0)

    def test_weights_change_the_optimum(self):
        D, c = hand()
        w = np.array([1, 10, 1, 1], dtype=float)          # s1 is heavy: A or C needed
        r = maximise_coverage_mip(D, w, 1, 10, candidates=c)
        self.assertAlmostEqual(r.objective, 11.0)
        self.assertEqual(r.nodes, ["A"])                    # A: s0+s1 = 11 ; C: s1+s2 = 11 -> A first by node order

    def test_undetectable_scenarios_stay_in_denominator(self):
        D, c = hand()
        D[3, :] = False
        r = maximise_coverage_mip(D, None, 3, 10, candidates=c)
        self.assertAlmostEqual(r.coverage, 0.75)


class TestGreedyIsWorse(unittest.TestCase):
    def test_greedy_5_vs_mip_6(self):
        D, c = greedy_trap()
        g = greedy_coverage(D, None, None, 2, c)
        m = maximise_coverage_mip(D, None, 2, 10, candidates=c)
        e = exhaustive_coverage(D, None, None, 2, c)
        self.assertEqual(g.n_detected, 5)
        self.assertEqual(m.n_detected, 6)
        self.assertEqual(e.n_detected, 6)
        self.assertEqual(m.nodes, ["Y", "Z"])
        self.assertEqual(g.status, "heuristic")            # greedy never claims optimality


class TestMipEqualsExhaustive(unittest.TestCase):
    def test_50_random_matrices(self):
        rng = np.random.default_rng(2024)
        for trial in range(50):
            S, n = int(rng.integers(4, 12)), int(rng.integers(3, 9))
            D = rng.random((S, n)) < 0.3
            w = rng.integers(1, 6, S).astype(float)
            k = int(rng.integers(1, min(n, 4) + 1))
            m = maximise_coverage_mip(D, w, k, 30, tie_break_enum_limit=0)    # raw solver layout
            e = exhaustive_coverage(D, None, w, k)
            self.assertEqual(m.status, "optimal", f"trial {trial}")
            self.assertAlmostEqual(m.objective, e.objective, places=6, msg=f"trial {trial}")
            self.assertAlmostEqual(m.solver_objective, e.objective, places=6, msg=f"trial {trial}")
            self.assertLessEqual(len(m.indexes), k)


class TestEdgeCases(unittest.TestCase):
    def test_k_zero(self):
        D, c = hand()
        for r in (maximise_coverage_mip(D, None, 0, 5, candidates=c), greedy_coverage(D, None, None, 0, c),
                  exhaustive_coverage(D, None, None, 0, c), sensitivity_ranked(D, None, None, 0, c)):
            self.assertEqual(r.indexes, [])
            self.assertEqual(r.coverage, 0.0)
        self.assertEqual(maximise_coverage_mip(D, None, 0, 5).status, "trivial")   # not "optimal": solver not run

    def test_k_larger_than_candidates(self):
        D, c = hand()
        r = maximise_coverage_mip(D, None, 10, 10, candidates=c)
        self.assertEqual(r.nodes, ["A", "B", "C"])
        self.assertAlmostEqual(r.coverage, 1.0)
        self.assertEqual(greedy_coverage(D, None, None, 10, c).nodes, ["A", "B", "C"])
        self.assertEqual(random_layouts(D, None, 10, 3, 1)["layouts"].shape, (3, 3))

    def test_empty_detections(self):
        D = np.zeros((4, 3), dtype=bool)
        r = maximise_coverage_mip(D, None, 2, 10)
        self.assertEqual(r.status, "optimal")
        self.assertEqual(r.coverage, 0.0)
        self.assertEqual(r.n_detected, 0)
        self.assertEqual(len(r.indexes), 2)
        self.assertEqual(r.mean_delay_min, float("inf"))

    def test_bad_inputs(self):
        D, c = hand()
        with self.assertRaises(ValueError):
            maximise_coverage_mip(D, None, -1, 5)
        with self.assertRaises(ValueError):
            maximise_coverage_mip(D, np.ones(3), 1, 5)
        with self.assertRaises(ValueError):
            maximise_coverage_mip(D, None, 1, 0)
        with self.assertRaises(ValueError):
            maximise_coverage_mip(D, None, 1, 5, T=np.full(D.shape, NAN))   # NaN where detected


class TestTieBreaks(unittest.TestCase):
    def test_shortest_delay_wins(self):
        D = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool)
        T = np.array([[NAN, 10, NAN], [NAN, 10, NAN], [NAN, NAN, 5], [NAN, NAN, 5]], dtype=float)
        for r in (exhaustive_coverage(D, T, None, 1, ["A", "B", "C"]),
                  maximise_coverage_mip(D, None, 1, 10, T=T, candidates=["A", "B", "C"]),
                  greedy_coverage(D, T, None, 1, ["A", "B", "C"])):
            self.assertEqual(r.nodes, ["C"])

    def test_natural_node_order(self):
        D, T = np.ones((2, 2), dtype=bool), np.full((2, 2), 5.0)
        for r in (exhaustive_coverage(D, T, None, 1, ["J10", "J2"]),
                  maximise_coverage_mip(D, None, 1, 10, T=T, candidates=["J10", "J2"]),
                  greedy_coverage(D, T, None, 1, ["J10", "J2"]),
                  sensitivity_ranked(D, T, None, 1, ["J10", "J2"])):
            self.assertEqual(r.nodes, ["J2"])

    def test_sensitivity_ranked_ignores_overlap(self):
        # X and Y see the same 3 scenarios, Z sees a different 2: ranking picks X+Y (3), the optimum is X+Z (5)
        D = np.zeros((5, 3), dtype=bool)
        D[[0, 1, 2], 0] = D[[0, 1, 2], 1] = True
        D[[3, 4], 2] = True
        self.assertEqual(sensitivity_ranked(D, None, None, 2, ["X", "Y", "Z"]).n_detected, 3)
        self.assertEqual(maximise_coverage_mip(D, None, 2, 10, candidates=["X", "Y", "Z"]).n_detected, 5)


class TestDeterminism(unittest.TestCase):
    def test_random_layouts_same_seed_same_output(self):
        D = np.random.default_rng(1).random((10, 8)) < 0.4
        a, b, c = random_layouts(D, None, 3, 25, 7), random_layouts(D, None, 3, 25, 7), random_layouts(D, None, 3, 25, 8)
        np.testing.assert_array_equal(a["layouts"], b["layouts"])
        np.testing.assert_array_equal(a["coverage"], b["coverage"])
        self.assertFalse(np.array_equal(a["layouts"], c["layouts"]))

    def test_mip_repeatable(self):
        D = np.random.default_rng(3).random((12, 9)) < 0.3
        runs = [maximise_coverage_mip(D, None, 3, 10).indexes for _ in range(3)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])


class TestTimeLimit(unittest.TestCase):
    D = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)

    def test_time_limit_with_incumbent_is_not_called_optimal(self):
        fake = SimpleNamespace(status=1, x=np.array([1, 0, 1, 0, 1.0]), fun=-2.0, mip_gap=0.25, message="time limit")
        with mock.patch.object(co, "milp", return_value=fake):
            r = maximise_coverage_mip(self.D, None, 1, 0.01)
        self.assertEqual(r.status, "time_limit_feasible")
        self.assertNotEqual(r.status, "optimal")
        self.assertEqual(r.gap, 0.25)
        self.assertFalse(r.tie_break_applied)

    def test_time_limit_without_incumbent(self):
        fake = SimpleNamespace(status=1, x=None, fun=None, mip_gap=None, message="time limit, no solution")
        with mock.patch.object(co, "milp", return_value=fake):
            r = maximise_coverage_mip(self.D, None, 1, 0.01)
        self.assertEqual(r.status, "time_limit_no_solution")
        self.assertEqual(r.indexes, [])

    def test_solver_exception_is_recorded(self):
        with mock.patch.object(co, "milp", side_effect=RuntimeError("boom")):
            r = maximise_coverage_mip(self.D, None, 1, 5)
        self.assertEqual(r.status, "solver_error")

    def test_real_solver_passes_time_limit_option(self):
        real = co.milp
        with mock.patch.object(co, "milp", wraps=real) as spy:
            maximise_coverage_mip(self.D, None, 1, 12.5)
        self.assertEqual(spy.call_args.kwargs["options"]["time_limit"], 12.5)


if __name__ == "__main__":
    unittest.main()
