"""Synthetic data generator — makes this repository runnable without real data.

Chainguard was built for a challenge whose dataset belongs to the organiser and
is not redistributed here. Rather than leave the repo dead on arrival, this
module fabricates a workbook with the *same five-sheet schema*, the same column
names, and comparable dimensions and distributions.

Everything it emits is invented. Hub IDs, material numbers, forwarders, lot
numbers and shipment IDs are generated from a seeded RNG. City names and
coordinates are real, publicly known logistics locations chosen so the map view
is legible — they carry no information about anyone's actual network.

The generator is *constructive about feasibility*: for every shipment it emits,
it guarantees at least one route option that clears every hard gate. That makes
the synthetic workbook a valid regression fixture — if the optimiser reports an
infeasible shipment on synthetic data, that is a bug in the optimiser, not in
the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXTERNAL_SHEET, HUB_SHEET, INTERNAL_SHEET, MATERIAL_SHEET, ROUTE_SHEET

# ---------------------------------------------------------------------------
# Fabrication vocabulary
# ---------------------------------------------------------------------------

CITIES: list[tuple[str, str, float, float, str]] = [
    ("Manila", "Philippines", 14.5995, 120.9842, "Southeast Asia"),
    ("Cebu", "Philippines", 10.3157, 123.8854, "Southeast Asia"),
    ("Penang", "Malaysia", 5.4164, 100.3327, "Southeast Asia"),
    ("Melaka", "Malaysia", 2.1896, 102.2501, "Southeast Asia"),
    ("Kuala Lumpur", "Malaysia", 3.1390, 101.6869, "Southeast Asia"),
    ("Singapore", "Singapore", 1.3521, 103.8198, "Southeast Asia"),
    ("Bangkok", "Thailand", 13.7563, 100.5018, "Southeast Asia"),
    ("Seoul", "South Korea", 37.5665, 126.9780, "East Asia"),
    ("Wuxi", "China", 31.4912, 120.3119, "East Asia"),
    ("Shanghai", "China", 31.2304, 121.4737, "East Asia"),
    ("Hsinchu", "Taiwan", 24.8138, 120.9675, "East Asia"),
    ("Kaohsiung", "Taiwan", 22.6273, 120.3014, "East Asia"),
    ("Tokyo", "Japan", 35.6762, 139.6503, "East Asia"),
    ("Dresden", "Germany", 51.0504, 13.7373, "Europe"),
    ("Regensburg", "Germany", 49.0134, 12.1016, "Europe"),
    ("Frankfurt", "Germany", 50.1109, 8.6821, "Europe"),
    ("Villach", "Austria", 46.6103, 13.8558, "Europe"),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041, "Europe"),
    ("Austin", "United States", 30.2672, -97.7431, "North America"),
    ("Los Angeles", "United States", 34.0522, -118.2437, "North America"),
    ("Dubai", "United Arab Emirates", 25.2048, 55.2708, "Middle East"),
]

STAGES: tuple[str, ...] = ("FE", "SIFO", "Backend", "OSAT")
STAGE_PREFIX = {"FE": "FE", "SIFO": "SIFO", "Backend": "BE", "OSAT": "OSAT"}

LANES: tuple[tuple[str, str], ...] = (
    ("FE", "SIFO"),
    ("SIFO", "Backend"),
    ("Backend", "OSAT"),
    ("OSAT", "Backend"),
    ("Backend", "SIFO"),
    ("SIFO", "FE"),
    ("FE", "Backend"),
    ("SIFO", "OSAT"),
    ("Backend", "Backend"),
    ("OSAT", "OSAT"),
)

PRODUCT_GROUPS = (
    "SenseLink", "SecureConnect", "DriveLogic", "EnergyEdge",
    "AutoControl", "SignalHub", "MemoryFlex", "PowerCore",
)
DIVISIONS = ("ATV", "PSS", "GIP", "CSS")
HAZARD_CLASSES = ("ESD Sensitive", "Moisture Sensitive", "Lithium Handling")
TEMP_REQUIREMENTS = ("Ambient", "Cold Chain")
PRIORITY_CLASSES = ("Standard", "Expedite", "Critical")
INCOTERMS = ("DAP", "FCA", "EXW", "DDP")
STATUSES = ("Planned", "In Transit", "Delivered", "Delayed", "Quality Hold")
MODES = ("Air", "Ocean", "Road", "Courier")
HUB_DISRUPTIONS = ("None", "Port congestion", "Labor shortage", "Weather disruption")
ROUTE_SCENARIOS = ("Normal", "PrimaryHubDown", "AirCapacityReduced")

# Per-mode cost / speed / risk / carbon character, matching the economics of
# real freight: courier is fast and dear, ocean is slow, cheap and dirty-per-day
# but clean per tonne-km, air is fast and carbon-heavy.
MODE_PROFILE: dict[str, dict[str, tuple[float, float]]] = {
    #            lead time (days)    cost (EUR)        risk (0-5)     CO2 (kg)
    "Courier": {"lead": (1, 4), "cost": (620, 890), "risk": (0.6, 2.6), "co2": (140, 205)},
    "Air": {"lead": (2, 5), "cost": (470, 800), "risk": (1.2, 3.4), "co2": (180, 244)},
    "Road": {"lead": (3, 8), "cost": (240, 520), "risk": (1.4, 3.8), "co2": (104, 150)},
    "Ocean": {"lead": (8, 12), "cost": (145, 450), "risk": (2.4, 4.7), "co2": (110, 165)},
}

DEST_COUNTRIES = ("DE", "US", "NL", "IN", "FR", "SG", "JP", "CN", "MY", "KR")
DEST_AIRPORTS = ("FRA", "LAX", "AMS", "BLR", "CDG", "SIN", "NRT", "PVG", "KUL", "ICN")


@dataclass(frozen=True)
class SynthConfig:
    """Dimensions of the fabricated workbook. Defaults mirror the challenge scale."""

    n_hubs: int = 488
    n_materials: int = 240
    n_families: int = 99
    n_internal: int = 240
    n_external: int = 225
    alternatives_per_lane: int = 8
    seed: int = 42
    # Fraction of hubs with cold-chain handling. Below ~0.3 the cold-chain
    # scenario starts producing genuinely infeasible shipments, which is a
    # useful stress test but a poor default.
    cold_chain_share: float = 0.45
    # Fraction of route rows flagged unavailable, simulating live blackouts.
    unavailable_share: float = 0.03


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------


def _build_hubs(rng: np.random.Generator, cfg: SynthConfig) -> pd.DataFrame:
    per_stage = {
        "FE": int(cfg.n_hubs * 0.22),
        "SIFO": int(cfg.n_hubs * 0.23),
        "Backend": int(cfg.n_hubs * 0.28),
    }
    per_stage["OSAT"] = cfg.n_hubs - sum(per_stage.values())

    rows: list[dict] = []
    for stage, count in per_stage.items():
        for i in range(1, count + 1):
            city, country, lat, lon, cluster = CITIES[rng.integers(len(CITIES))]
            weekly = int(rng.choice([12_000, 15_000, 18_000, 22_000, 29_000, 43_000, 93_000],
                                    p=[0.28, 0.22, 0.18, 0.14, 0.10, 0.06, 0.02]))
            disruption = str(rng.choice(HUB_DISRUPTIONS, p=[0.845, 0.085, 0.06, 0.01]))
            reduction = 0.0 if disruption == "None" else float(np.round(rng.uniform(0.15, 0.35), 2))
            hazards = ["None", "ESD Sensitive"]
            if rng.random() < 0.85:
                hazards.append("Moisture Sensitive")
            if rng.random() < 0.70:
                hazards.append("Lithium Handling")
            rows.append(
                {
                    "HubID": f"{STAGE_PREFIX[stage]}_LOC_{i:03d}",
                    "Stage": stage,
                    "WeeklyCapacityUnits": weekly,
                    "CurrentUtilizationPct": float(np.round(rng.uniform(0.35, 0.77), 2)),
                    "MaxUtilizationPct": 0.90,
                    "ColdChainAvailable": "Yes" if rng.random() < cfg.cold_chain_share else "No",
                    "DisruptionScenario": disruption,
                    "CapacityReductionPct": reduction,
                    "FixedHandlingCost_EUR": int(rng.integers(800, 3400)),
                    "VariableHandlingCostPerUnit_EUR": float(np.round(rng.uniform(0.025, 0.101), 3)),
                    "City": city,
                    "Country": country,
                    # Jitter coordinates so co-located hubs are separable on a map
                    # without moving anything to a different city.
                    "Latitude": float(np.round(lat + rng.normal(0, 0.04), 4)),
                    "Longitude": float(np.round(lon + rng.normal(0, 0.04), 4)),
                    "GeoCluster": cluster,
                    "ESDHandlingAvailable": "Yes",
                    "MoistureControlAvailable": "Yes" if "Moisture Sensitive" in hazards else "No",
                    "LithiumHandlingAvailable": "Yes" if "Lithium Handling" in hazards else "No",
                    "SupportedHazardClasses": "; ".join(hazards),
                }
            )
    return pd.DataFrame(rows)


def _build_materials(rng: np.random.Generator, cfg: SynthConfig) -> pd.DataFrame:
    families = []
    while len(families) < cfg.n_families:
        fam = f"{rng.choice(DIVISIONS)}-{rng.integers(10, 60)}-{rng.choice(PRODUCT_GROUPS)}"
        if fam not in families:
            families.append(fam)

    rows = []
    for i in range(cfg.n_materials):
        fam = families[i % len(families)]
        rows.append(
            {
                "MaterialNo_Anon": f"MAT-{10_000 + i * 13}",
                "MaterialFamily": fam,
                "ProductGroup": fam.rsplit("-", 1)[-1],
                "HazardClass": HAZARD_CLASSES[i % len(HAZARD_CLASSES)],
                "ShelfLifeDays": int(rng.choice([90, 120, 150, 180, 240, 300, 365, 420])),
                "TempRequirement": TEMP_REQUIREMENTS[0] if rng.random() > 0.20 else TEMP_REQUIREMENTS[1],
                "PriorityClass": PRIORITY_CLASSES[i % len(PRIORITY_CLASSES)],
                "SubstitutionGroup": f"SUB-{fam.rsplit('-', 1)[0]}",
            }
        )
    return pd.DataFrame(rows)


def _hub_pool(hubs: pd.DataFrame, stage: str, hazard: str, cold: bool) -> pd.DataFrame:
    """Hubs at ``stage`` that can physically handle this material."""
    pool = hubs[hubs["Stage"] == stage]
    pool = pool[pool["SupportedHazardClasses"].str.contains(hazard, regex=False)]
    if cold:
        pool = pool[pool["ColdChainAvailable"].str.lower() == "yes"]
    return pool


def _build_routes(
    rng: np.random.Generator,
    cfg: SynthConfig,
    hubs: pd.DataFrame,
    materials: pd.DataFrame,
) -> pd.DataFrame:
    """One primary + N alternatives per (family, lane) actually in use.

    The primary and the first two alternatives are constructed through hubs that
    satisfy the family's hazard and temperature requirements, which is what
    guarantees every shipment has a feasible option.
    """
    fam_profile = (
        materials.groupby("MaterialFamily")
        .agg(
            hazard=("HazardClass", "first"),
            cold=("TempRequirement", lambda s: (s == "Cold Chain").any()),
        )
        .reset_index()
    )

    rows: list[dict] = []
    rid = 0
    for _, fam in fam_profile.iterrows():
        family, hazard, cold = fam["MaterialFamily"], fam["hazard"], bool(fam["cold"])
        # Each family uses a random subset of lanes; every family gets the three
        # forward lanes so the end-to-end network graph is always connected.
        lanes = list(LANES[:3]) + [
            LANES[i] for i in rng.choice(range(3, len(LANES)), size=3, replace=False)
        ]

        # A material family flows through a *defined set of facilities*, not a
        # fresh random hub on every leg. Fixing a small per-stage hub pool per
        # family is what makes consecutive legs actually join up (leg 1's
        # destination is a legal origin for leg 2), so the end-to-end network
        # graph is traversable rather than a pile of disconnected edges.
        safe_pool: dict[str, pd.DataFrame] = {}
        wide_pool: dict[str, pd.DataFrame] = {}
        for stage in STAGES:
            eligible = _hub_pool(hubs, stage, hazard, cold)
            at_stage = hubs[hubs["Stage"] == stage]
            if eligible.empty:
                eligible = at_stage
            safe = eligible.sample(min(3, len(eligible)), random_state=int(rng.integers(1e9)))
            extra = at_stage.sample(min(3, len(at_stage)), random_state=int(rng.integers(1e9)))
            safe_pool[stage] = safe
            wide_pool[stage] = pd.concat([safe, extra]).drop_duplicates("HubID")

        for stage_from, stage_to in lanes:
            for k in range(cfg.alternatives_per_lane):
                rid += 1
                guaranteed = k < 3  # constructive feasibility for the first three
                pool_from = safe_pool[stage_from] if guaranteed else wide_pool[stage_from]
                pool_to = safe_pool[stage_to] if guaranteed else wide_pool[stage_to]
                src = pool_from.sample(1, random_state=int(rng.integers(1e9)))
                dst = pool_to.sample(1, random_state=int(rng.integers(1e9)))
                # Intra-stage lanes (Backend -> Backend) must still move between
                # two different facilities; a leg to itself is not a shipment.
                tries = 0
                while dst["HubID"].iloc[0] == src["HubID"].iloc[0] and len(pool_to) > 1 and tries < 8:
                    dst = pool_to.sample(1, random_state=int(rng.integers(1e9)))
                    tries += 1
                if dst["HubID"].iloc[0] == src["HubID"].iloc[0]:
                    continue
                mode = MODES[k % len(MODES)]
                prof = MODE_PROFILE[mode]
                lead = int(rng.integers(prof["lead"][0], prof["lead"][1] + 1))
                cost = int(rng.uniform(*prof["cost"]))
                risk = float(np.round(rng.uniform(*prof["risk"]), 1))
                co2 = int(rng.uniform(*prof["co2"]))
                scenario = "Normal" if guaranteed else str(rng.choice(ROUTE_SCENARIOS, p=[0.3, 0.35, 0.35]))
                available = "Yes" if (guaranteed or rng.random() > cfg.unavailable_share) else "No"
                rows.append(
                    {
                        "RouteOptionID": f"RO-{rid:05d}",
                        "MaterialFamily": family,
                        "StageFrom": stage_from,
                        "StageTo": stage_to,
                        "FromHub": src["HubID"].iloc[0],
                        "ToHub": dst["HubID"].iloc[0],
                        "TransportMode": mode,
                        "Forwarder_Anon": f"FWD-{rng.integers(1, 40):03d}",
                        "BaseLeadTimeDays": lead,
                        "BaseCostEUR": cost,
                        "CapacityUnitsPerWeek": int(rng.choice([3_000, 5_500, 8_600, 12_000, 18_000, 25_000])),
                        "RiskScore": risk,
                        "CO2Kg": co2,
                        "IsPrimary": "Yes" if k == 0 else "No",
                        "DisruptionScenario": scenario,
                        "AvailableFlag": available,
                        "Notes": "Primary planned route" if k == 0 else "Alternative route option",
                    }
                )
    return pd.DataFrame(rows)


def _build_internal(
    rng: np.random.Generator,
    cfg: SynthConfig,
    routes: pd.DataFrame,
    materials: pd.DataFrame,
) -> pd.DataFrame:
    """Shipments are sampled *from the lanes that have primary routes*.

    That is what makes the fixture honest: every shipment is placeable, so a
    reported infeasibility is a real modelling failure rather than a data gap.
    """
    primaries = routes[(routes["IsPrimary"] == "Yes") & (routes["AvailableFlag"] == "Yes")]
    mat_by_family = materials.groupby("MaterialFamily")["MaterialNo_Anon"].apply(list).to_dict()

    rows: list[dict] = []
    picks = primaries.sample(n=cfg.n_internal, replace=True, random_state=cfg.seed)
    for i, (_, r) in enumerate(picks.iterrows(), start=1):
        family = r["MaterialFamily"]
        candidates = mat_by_family.get(family, [])
        material = candidates[int(rng.integers(len(candidates)))] if candidates else "MAT-10000"
        qty = int(rng.choice([100, 300, 700, 1_500, 4_000, 12_000, 36_000], p=[0.16, 0.18, 0.18, 0.18, 0.14, 0.10, 0.06]))
        lead = int(rng.integers(1, 15))
        delay = int(rng.choice([0, 0, 0, 1, 2, 3], p=[0.6, 0.12, 0.08, 0.1, 0.06, 0.04]))
        ship_day = int(rng.integers(1, 28))
        ship = pd.Timestamp("2026-01-01") + pd.Timedelta(days=ship_day)
        temp_req = materials.loc[materials["MaterialNo_Anon"] == material, "TempRequirement"]
        is_cold = (not temp_req.empty) and temp_req.iloc[0] == "Cold Chain"
        rows.append(
            {
                "ShipmentID": f"SIM-{i:05d}",
                "Scenario": f"{r['StageFrom']}_to_{r['StageTo']}",
                "MaterialNo_Anon": material,
                "SPMaterialNo_Anon": f"SP-{10_000 + i * 7}",
                "DIV": family.split("-")[0],
                "PL": family.split("-")[1],
                "BatchLot_Anon": f"LOT-{i * 19:06d}",
                "StageFrom": r["StageFrom"],
                "StageTo": r["StageTo"],
                "ShipFrom_Alias": r["FromHub"],
                "ShipTo_Alias": r["ToHub"],
                "TransportMode": r["TransportMode"],
                "Forwarder_Anon": r["Forwarder_Anon"],
                "Incoterm": str(rng.choice(INCOTERMS)),
                "ShipDate": ship,
                "ExpectedArrival": ship + pd.Timedelta(days=lead),
                "ActualArrival": ship + pd.Timedelta(days=lead + delay),
                "LeadTimeDays": lead,
                "TransitDelayDays": delay,
                "Qty": qty,
                "UoM": "ST",
                "HandlingUnit_Anon": f"HU-{i * 23:06d}",
                "RouteRiskScore": float(np.round(rng.uniform(0, 9.9), 1)),
                "TemperatureControlled": "Yes" if is_cold else "No",
                "Status": str(rng.choice(STATUSES, p=[0.17, 0.18, 0.18, 0.29, 0.18])),
                "TracePath": f"{r['FromHub']} → {r['ToHub']}",
                "MaterialFamily": family,
            }
        )
    return pd.DataFrame(rows)


def _build_external(
    rng: np.random.Generator, cfg: SynthConfig, internal: pd.DataFrame
) -> pd.DataFrame:
    linked = internal.sample(n=min(cfg.n_external, len(internal)), replace=True, random_state=cfg.seed + 1)
    rows: list[dict] = []
    for i, (_, s) in enumerate(linked.iterrows(), start=1):
        weight = float(np.round(np.exp(rng.normal(3.2, 1.6)), 2))
        pieces = int(max(1, rng.integers(1, 120)))
        pickup = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(rng.integers(1, 28)))
        dest_idx = int(rng.integers(len(DEST_COUNTRIES)))
        rows.append(
            {
                "DeliveryNo": f"DEL-{i:05d}",
                "IncotermCode": s["Incoterm"],
                "Forwarder_Anon": s["Forwarder_Anon"],
                "ShipTo_CountryCodeISO": DEST_COUNTRIES[dest_idx],
                "ItemValueEuro": float(np.round(np.exp(rng.normal(8.2, 1.7)), 2)),
                "DelVolume_m3": float(np.round(weight / 1000 * rng.uniform(2, 9), 3)),
                "DelGrossWeight_KG": float(np.round(weight * rng.uniform(0.08, 0.15), 2)),
                "NoOfPackages": int(max(1, rng.integers(1, 8))),
                "ShipFromLocation": s["ShipFrom_Alias"],
                "Flight_No": f"{rng.choice(['SQ', 'LH', 'CX', 'EK'])} {rng.integers(100, 999)}",
                "Pieces": pieces,
                "PUP_Date": pickup,
                "Departure_Date_ETD": pickup + pd.Timedelta(days=1),
                "POD_Date": pickup + pd.Timedelta(days=int(rng.integers(2, 12))),
                "Airport_of_Destination": DEST_AIRPORTS[dest_idx],
                "Terms_of_Delivery": s["Incoterm"],
                "Gross_Weight": weight,
                "ChargeableWeight_KG": float(np.round(weight * rng.uniform(1.0, 1.15), 2)),
                "MaterialNo_Anon_Link": s["MaterialNo_Anon"],
                "InternalShipmentID_Link": s["ShipmentID"],
                "InternalScenario_Link": s["Scenario"],
                "InternalTracePath_Link": s["TracePath"],
                "InternalStageFrom_Link": s["StageFrom"],
                "InternalStageTo_Link": s["StageTo"],
                "MaterialFamily_Link": s["MaterialFamily"],
                "MaterialFamily": s["MaterialFamily"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(cfg: SynthConfig | None = None) -> dict[str, pd.DataFrame]:
    """Build the five synthetic sheets as in-memory frames."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)

    hubs = _build_hubs(rng, cfg)
    materials = _build_materials(rng, cfg)
    routes = _build_routes(rng, cfg, hubs, materials)
    internal = _build_internal(rng, cfg, routes, materials)
    external = _build_external(rng, cfg, internal)

    return {
        INTERNAL_SHEET: internal,
        EXTERNAL_SHEET: external,
        ROUTE_SHEET: routes,
        MATERIAL_SHEET: materials,
        HUB_SHEET: hubs,
    }


def write(path: str | Path, cfg: SynthConfig | None = None) -> Path:
    """Generate and write the synthetic workbook to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = generate(cfg)
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        for sheet, df in frames.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    return path
