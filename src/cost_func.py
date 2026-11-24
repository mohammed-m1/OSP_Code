"""
PURCHASE-COST MODEL FOR PRESSURE LOGGERS (hardware only)
=======================================================
Cost of a layout S of newly purchased loggers:   C(S) = unit_price * |S|

INCLUDED : hardware price of each new logger (AUD, excluding GST), read from the device catalogue.
EXCLUDED : installation and fittings, travel and labour, gateways and cloud subscriptions,
           calibration certificates, maintenance, replacement, discounting.
The base Bluetooth logger is a LOCAL logging device. Remote alarming needs a gateway and cloud.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

KPA_PER_M_HEAD = 9.80665      # 1 m of water head = 9.80665 kPa

EXCLUDED_COST_ITEMS = (
    "installation and fittings", "travel and labour", "gateway", "cloud subscription",
    "calibration certificate", "maintenance", "replacement", "lifecycle discounting",
)


@dataclass(frozen=True)
class DeviceSpec:
    """One pressure logger, exactly as published by the supplier."""
    device_id: str
    supplier: str
    product: str
    price_aud_ex_gst: float
    range_min_kpa: float
    range_max_kpa: float
    pressure_type: str
    accuracy_pct_fso: float
    interval_min_s: int
    interval_max_s: int
    memory_samples: int
    source_url: str
    accessed: str
    notes: str

    @property
    def range_max_m_head(self) -> float:
        return self.range_max_kpa / KPA_PER_M_HEAD

    @property
    def accuracy_kpa(self) -> float:
        return self.range_max_kpa * self.accuracy_pct_fso / 100.0

    @property
    def accuracy_m_head(self) -> float:
        return self.accuracy_kpa / KPA_PER_M_HEAD


def load_device_catalogue(path) -> dict:
    """Read data/devices/pressure_logger_catalog.csv into {device_id: DeviceSpec}."""
    catalogue = {}
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spec = DeviceSpec(
                device_id=row["device_id"], supplier=row["supplier"], product=row["product"],
                price_aud_ex_gst=float(row["price_aud_ex_gst"]),
                range_min_kpa=float(row["range_min_kpa"]), range_max_kpa=float(row["range_max_kpa"]),
                pressure_type=row["pressure_type"], accuracy_pct_fso=float(row["accuracy_pct_fso"]),
                interval_min_s=int(row["interval_min_s"]), interval_max_s=int(row["interval_max_s"]),
                memory_samples=int(row["memory_samples"]), source_url=row["source_url"],
                accessed=row["accessed"], notes=row["notes"],
            )
            catalogue[spec.device_id] = spec
    return catalogue


class PressureLoggerCostFunction:
    """Hardware purchase cost of a layout. Existing loggers (default: none) are free."""

    def __init__(self, device: DeviceSpec, existing_nodes=()):
        if device.price_aud_ex_gst <= 0:
            raise ValueError("Device price must be greater than zero.")
        self.device = device
        self.existing_nodes = tuple(existing_nodes)

    def evaluate(self, selected_nodes) -> dict:
        """Cost of NEWLY purchased loggers at `selected_nodes`."""
        selected = list(selected_nodes)
        if len(selected) != len(set(selected)):
            raise ValueError("selected_nodes contains duplicate locations.")
        already = sorted(set(selected) & set(self.existing_nodes))
        if already:
            raise ValueError(f"Locations already have a logger: {already}")
        n = len(selected)
        return {
            "new_logger_count": n,
            "existing_logger_count": len(self.existing_nodes),
            "unit_price_aud_ex_gst": self.device.price_aud_ex_gst,
            "hardware_cost_aud_ex_gst": round(n * self.device.price_aud_ex_gst, 2),
            "excluded_items": EXCLUDED_COST_ITEMS,
        }
