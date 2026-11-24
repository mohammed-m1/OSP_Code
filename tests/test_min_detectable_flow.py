import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import min_detectable_flow as m  # noqa: E402

try:
    import wntr
except ImportError:                                           # pragma: no cover
    wntr = None

L, D, C, H0, Q1 = 1000.0, 0.3, 100.0, 50.0, 0.02             # toy network: reservoir -> J1 (base demand 0.02) -> J2


def toy():
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R", base_head=H0)
    wn.add_junction("J1", base_demand=Q1, elevation=0)
    wn.add_junction("J2", base_demand=0.0, elevation=0)
    wn.add_pipe("P1", "R", "J1", length=L, diameter=D, roughness=C)
    wn.add_pipe("P2", "J1", "J2", length=1.0, diameter=D, roughness=C)    # J2 is a (nearly) dead end: no flow
    return wn


def hl(q):                                                    # Hazen-Williams, SI
    return 10.67 * L * q ** 1.852 / (C ** 1.852 * D ** 4.871)


def analytic_flow_for_drop(thr):                              # drop at J1 when q is added at J1
    target = hl(Q1) + thr
    q_total = (target * C ** 1.852 * D ** 4.871 / (10.67 * L)) ** (1 / 1.852)
    return q_total - Q1


class TestLogic(unittest.TestCase):
    flows = m.geometric_flows(0.1, 10.0, 21)

    def test_resolved_monotone(self):
        r = m.find_min_flow(lambda q: 2 * q, 5.0, self.flows)       # crosses at q = 2.5
        self.assertEqual(r.status, m.RESOLVED)
        self.assertGreater(r.min_flow, 2.5)
        self.assertLess(r.min_flow, 2.5 * (self.flows[1] / self.flows[0]))

    def test_bisection_tightens_answer(self):
        r = m.find_min_flow(lambda q: 2 * q, 5.0, self.flows, refine_steps=20)
        self.assertAlmostEqual(r.min_flow, 2.5, places=4)

    def test_non_monotone_is_flagged_and_not_bisected(self):
        drops = lambda q: 10 - abs(q - 3)                           # rises then falls
        r = m.find_min_flow(drops, 7.5, self.flows, refine_steps=20)
        self.assertEqual(r.status, m.NONMONO)
        self.assertTrue(math.isnan(r.min_flow))
        self.assertFalse(math.isnan(r.first_crossing))
        self.assertEqual(r.n_runs, len(self.flows))                 # no extra runs: no bisection

    def test_search_limit_failure_is_unresolved_not_finite(self):
        r = m.find_min_flow(lambda q: 0.01 * q, 5.0, self.flows)
        self.assertEqual(r.status, m.ABOVE)
        self.assertTrue(math.isnan(r.min_flow))
        r2 = m.find_min_flow(lambda q: 100 * q, 5.0, self.flows)      # already exceeded at the smallest flow
        self.assertEqual(r2.status, m.BELOW)
        self.assertTrue(math.isnan(r2.min_flow))

    def test_failed_sim_is_unresolved(self):
        r = m.find_min_flow(lambda q: None if q > 1 else 0.0, 5.0, self.flows)
        self.assertEqual(r.status, m.FAILED)

    def test_summary_excludes_unresolved_and_reports_shares(self):
        mat = pd.DataFrame(dict(status=[m.RESOLVED, m.RESOLVED, m.ABOVE, m.NONMONO],
                                min_flow_m3s=[1.0, 3.0, math.nan, math.nan]))
        s = m.summarise(mat)
        self.assertAlmostEqual(s["mean_minimum_detectable_flow_m3s"], 2.0)
        self.assertAlmostEqual(s["unresolved_share"], 0.25)
        self.assertAlmostEqual(s["non_monotonic_share"], 0.25)
        self.assertNotIn("ADT", s["metric_name"].replace("not Du et al.'s ADT", ""))

    def test_bad_grid_rejected(self):
        with self.assertRaises(ValueError):
            m.find_min_flow(lambda q: q, 1.0, [2.0, 1.0])
        with self.assertRaises(ValueError):
            m.geometric_flows(0, 1, 5)


@unittest.skipIf(wntr is None, "wntr not installed")
class TestToyHydraulics(unittest.TestCase):
    def test_matches_analytic_head_loss(self):
        thr = 0.5
        flows = m.geometric_flows(1e-4, 0.05, 25)
        mat = m.build_matrix(toy, ["J1"], ["J1"], thr, flows, refine_steps=25)
        r = mat.iloc[0]
        self.assertEqual(r.status, m.RESOLVED)
        self.assertAlmostEqual(r.min_flow_m3s / analytic_flow_for_drop(thr), 1.0, delta=0.01)

    def test_dead_end_sensor_sees_the_same_drop(self):
        flows = m.geometric_flows(1e-4, 0.05, 25)
        mat = m.build_matrix(toy, ["J1"], ["J1", "J2"], 0.5, flows, refine_steps=15).set_index("sensor")
        self.assertAlmostEqual(mat.loc["J1", "min_flow_m3s"], mat.loc["J2", "min_flow_m3s"], delta=1e-4)

    def test_unreachable_threshold_is_unresolved_on_real_network(self):
        flows = m.geometric_flows(1e-4, 0.01, 10)
        mat = m.build_matrix(toy, ["J1"], ["J1"], 50.0, flows)
        self.assertEqual(mat.iloc[0].status, m.ABOVE)
        self.assertTrue(math.isnan(mat.iloc[0].min_flow_m3s))


if __name__ == "__main__":
    unittest.main()
