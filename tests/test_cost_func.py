import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cost_func import PressureLoggerCostFunction, load_device_catalogue  # noqa: E402


class TestPurchaseCost(unittest.TestCase):
    def setUp(self):
        self.device = load_device_catalogue(ROOT / "data" / "devices" / "pressure_logger_catalog.csv")["EFENTO_BT_689KPA"]
        self.cost = PressureLoggerCostFunction(self.device)

    def test_catalogue_matches_published_listing(self):
        self.assertEqual(self.device.price_aud_ex_gst, 726.36)
        self.assertEqual(self.device.range_max_kpa, 689)
        self.assertAlmostEqual(self.device.accuracy_kpa, 1.7225, places=4)

    def test_zero_loggers_cost_nothing(self):
        r = self.cost.evaluate([])
        self.assertEqual(r["new_logger_count"], 0)
        self.assertEqual(r["hardware_cost_aud_ex_gst"], 0.0)

    def test_one_logger(self):
        self.assertAlmostEqual(self.cost.evaluate(["J1"])["hardware_cost_aud_ex_gst"], 726.36, places=2)

    def test_two_loggers(self):
        r = self.cost.evaluate(["J1", "J2"])
        self.assertEqual(r["new_logger_count"], 2)
        self.assertAlmostEqual(r["hardware_cost_aud_ex_gst"], 1452.72, places=2)

    def test_duplicate_locations_rejected(self):
        with self.assertRaises(ValueError):
            self.cost.evaluate(["J1", "J1"])

    def test_existing_loggers_are_free_and_cannot_be_rebought(self):
        c = PressureLoggerCostFunction(self.device, existing_nodes=["J9"])
        self.assertEqual(c.evaluate(["J1"])["hardware_cost_aud_ex_gst"], 726.36)
        self.assertEqual(c.evaluate(["J1"])["existing_logger_count"], 1)
        with self.assertRaises(ValueError):
            c.evaluate(["J9"])

    def test_bad_price_rejected(self):
        with self.assertRaises(ValueError):
            PressureLoggerCostFunction(replace(self.device, price_aud_ex_gst=0))

    def test_lifecycle_items_are_declared_excluded(self):
        self.assertIn("installation and fittings", self.cost.evaluate(["J1"])["excluded_items"])


if __name__ == "__main__":
    unittest.main()
