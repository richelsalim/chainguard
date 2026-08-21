"""Column contracts for the five source sheets, plus a validating loader helper.

The point of this module is that a malformed workbook fails *immediately*, at
the boundary, with a message naming the sheet and the missing columns — instead
of surfacing as a silent NaN three layers down inside a solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (
    EXTERNAL_SHEET,
    HUB_SHEET,
    INTERNAL_SHEET,
    MATERIAL_SHEET,
    ROUTE_SHEET,
)


@dataclass(frozen=True)
class SheetContract:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    key: str | None = None

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Return a list of human-readable problems (empty means valid)."""
        problems: list[str] = []
        missing = [c for c in self.required if c not in df.columns]
        if missing:
            problems.append(
                f"[{self.name}] missing required column(s): {', '.join(missing)}"
            )
        if df.empty:
            problems.append(f"[{self.name}] contains no rows")
        if self.key and self.key in df.columns:
            dupes = df[self.key].dropna().duplicated().sum()
            if dupes:
                problems.append(f"[{self.name}] has {dupes} duplicate {self.key} value(s)")
        return problems


INTERNAL_CONTRACT = SheetContract(
    name=INTERNAL_SHEET,
    key="ShipmentID",
    required=(
        "ShipmentID",
        "MaterialNo_Anon",
        "StageFrom",
        "StageTo",
        "TransportMode",
        "Qty",
        "MaterialFamily",
    ),
    optional=(
        "Scenario",
        "ShipFrom_Alias",
        "ShipTo_Alias",
        "Forwarder_Anon",
        "Incoterm",
        "ShipDate",
        "ExpectedArrival",
        "ActualArrival",
        "LeadTimeDays",
        "TransitDelayDays",
        "UoM",
        "RouteRiskScore",
        "TemperatureControlled",
        "Status",
        "TracePath",
    ),
)

EXTERNAL_CONTRACT = SheetContract(
    name=EXTERNAL_SHEET,
    key="DeliveryNo",
    required=(
        "DeliveryNo",
        "MaterialNo_Anon_Link",
        "InternalShipmentID_Link",
        "ChargeableWeight_KG",
    ),
    optional=(
        "IncotermCode",
        "Forwarder_Anon",
        "ShipTo_CountryCodeISO",
        "ItemValueEuro",
        "DelVolume_m3",
        "DelGrossWeight_KG",
        "NoOfPackages",
        "ShipFromLocation",
        "Flight_No",
        "Pieces",
        "PUP_Date",
        "Departure_Date_ETD",
        "POD_Date",
        "Airport_of_Destination",
        "Gross_Weight",
        "MaterialFamily_Link",
    ),
)

ROUTE_CONTRACT = SheetContract(
    name=ROUTE_SHEET,
    key="RouteOptionID",
    required=(
        "RouteOptionID",
        "MaterialFamily",
        "StageFrom",
        "StageTo",
        "FromHub",
        "ToHub",
        "TransportMode",
        "BaseLeadTimeDays",
        "BaseCostEUR",
        "CapacityUnitsPerWeek",
        "RiskScore",
        "CO2Kg",
        "IsPrimary",
        "DisruptionScenario",
        "AvailableFlag",
    ),
    optional=("Forwarder_Anon", "Notes"),
)

MATERIAL_CONTRACT = SheetContract(
    name=MATERIAL_SHEET,
    key="MaterialNo_Anon",
    required=(
        "MaterialNo_Anon",
        "MaterialFamily",
        "HazardClass",
        "TempRequirement",
        "PriorityClass",
    ),
    optional=("ProductGroup", "ShelfLifeDays", "SubstitutionGroup"),
)

HUB_CONTRACT = SheetContract(
    name=HUB_SHEET,
    key="HubID",
    required=(
        "HubID",
        "Stage",
        "WeeklyCapacityUnits",
        "CurrentUtilizationPct",
        "MaxUtilizationPct",
        "ColdChainAvailable",
        "DisruptionScenario",
        "CapacityReductionPct",
        "SupportedHazardClasses",
    ),
    optional=(
        "FixedHandlingCost_EUR",
        "VariableHandlingCostPerUnit_EUR",
        "City",
        "Country",
        "Latitude",
        "Longitude",
        "GeoCluster",
        "ESDHandlingAvailable",
        "MoistureControlAvailable",
        "LithiumHandlingAvailable",
    ),
)

CONTRACTS: dict[str, SheetContract] = {
    INTERNAL_SHEET: INTERNAL_CONTRACT,
    EXTERNAL_SHEET: EXTERNAL_CONTRACT,
    ROUTE_SHEET: ROUTE_CONTRACT,
    MATERIAL_SHEET: MATERIAL_CONTRACT,
    HUB_SHEET: HUB_CONTRACT,
}


class SchemaError(ValueError):
    """Raised when a source workbook does not satisfy the sheet contracts."""


def validate_all(frames: dict[str, pd.DataFrame], strict: bool = True) -> list[str]:
    """Validate every loaded frame against its contract.

    Returns the list of problems. Raises :class:`SchemaError` when ``strict``
    and anything is wrong.
    """
    problems: list[str] = []
    for sheet, contract in CONTRACTS.items():
        if sheet not in frames:
            problems.append(f"[{sheet}] sheet not found in workbook")
            continue
        problems.extend(contract.validate(frames[sheet]))
    if problems and strict:
        raise SchemaError(
            "Source workbook failed validation:\n  - " + "\n  - ".join(problems)
        )
    return problems
