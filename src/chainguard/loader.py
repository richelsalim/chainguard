"""Load and normalise the five source sheets into typed, validated frames.

This is the only module that touches Excel. Everything downstream works on
clean pandas frames with guaranteed columns and coerced dtypes, so no solver or
simulation code ever has to defend itself against a stray string in a numeric
column or a trailing "Unnamed: 33" artefact from an Excel export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    EXTERNAL_SHEET,
    HUB_SHEET,
    INTERNAL_SHEET,
    MATERIAL_SHEET,
    REQUIRED_SHEETS,
    ROUTE_SHEET,
)
from .schema import CONTRACTS, validate_all

# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

_NUMERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    INTERNAL_SHEET: ("Qty", "LeadTimeDays", "TransitDelayDays", "RouteRiskScore"),
    EXTERNAL_SHEET: (
        "ItemValueEuro",
        "DelVolume_m3",
        "DelGrossWeight_KG",
        "NoOfPackages",
        "Pieces",
        "Gross_Weight",
        "ChargeableWeight_KG",
    ),
    ROUTE_SHEET: (
        "BaseLeadTimeDays",
        "BaseCostEUR",
        "CapacityUnitsPerWeek",
        "RiskScore",
        "CO2Kg",
    ),
    MATERIAL_SHEET: ("ShelfLifeDays",),
    HUB_SHEET: (
        "WeeklyCapacityUnits",
        "CurrentUtilizationPct",
        "MaxUtilizationPct",
        "CapacityReductionPct",
        "FixedHandlingCost_EUR",
        "VariableHandlingCostPerUnit_EUR",
        "Latitude",
        "Longitude",
    ),
}

_TEXT_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    INTERNAL_SHEET: (
        "ShipmentID",
        "MaterialNo_Anon",
        "StageFrom",
        "StageTo",
        "TransportMode",
        "MaterialFamily",
        "ShipFrom_Alias",
        "ShipTo_Alias",
    ),
    EXTERNAL_SHEET: ("DeliveryNo", "MaterialNo_Anon_Link", "InternalShipmentID_Link"),
    ROUTE_SHEET: (
        "RouteOptionID",
        "MaterialFamily",
        "StageFrom",
        "StageTo",
        "FromHub",
        "ToHub",
        "TransportMode",
        "IsPrimary",
        "DisruptionScenario",
        "AvailableFlag",
    ),
    MATERIAL_SHEET: (
        "MaterialNo_Anon",
        "MaterialFamily",
        "HazardClass",
        "TempRequirement",
        "PriorityClass",
    ),
    HUB_SHEET: (
        "HubID",
        "Stage",
        "ColdChainAvailable",
        "DisruptionScenario",
        "SupportedHazardClasses",
    ),
}


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Excel exports pad rows with empty ``Unnamed: N`` columns. Remove them."""
    keep = [c for c in df.columns if not str(c).startswith("Unnamed")]
    return df.loc[:, keep].copy()


def _coerce(df: pd.DataFrame, sheet: str) -> pd.DataFrame:
    df = _drop_unnamed(df)
    for col in _NUMERIC_COLUMNS.get(sheet, ()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _TEXT_KEY_COLUMNS.get(sheet, ()):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    # Percentage columns occasionally arrive as "42%" strings.
    for col in ("CurrentUtilizationPct", "MaxUtilizationPct", "CapacityReductionPct"):
        if col in df.columns and df[col].dtype == object:
            df[col] = (
                df[col].astype(str).str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
            )
            df.loc[df[col] > 1.0, col] = df.loc[df[col] > 1.0, col] / 100.0
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bronze container
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """The validated Bronze layer: five raw-but-typed frames plus provenance."""

    source: Path
    internal: pd.DataFrame
    external: pd.DataFrame
    routes: pd.DataFrame
    materials: pd.DataFrame
    hubs: pd.DataFrame
    warnings: list[str]

    def frame(self, sheet: str) -> pd.DataFrame:
        return {
            INTERNAL_SHEET: self.internal,
            EXTERNAL_SHEET: self.external,
            ROUTE_SHEET: self.routes,
            MATERIAL_SHEET: self.materials,
            HUB_SHEET: self.hubs,
        }[sheet]

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        return {sheet: self.frame(sheet) for sheet in REQUIRED_SHEETS}

    def profile(self) -> pd.DataFrame:
        """One row per sheet: dimensions and declared vs present columns."""
        rows: list[dict[str, Any]] = []
        for sheet in REQUIRED_SHEETS:
            df = self.frame(sheet)
            contract = CONTRACTS[sheet]
            declared = set(contract.required) | set(contract.optional)
            rows.append(
                {
                    "sheet": sheet,
                    "rows": len(df),
                    "columns": df.shape[1],
                    "required_present": sum(c in df.columns for c in contract.required),
                    "required_total": len(contract.required),
                    "extra_columns": len(set(df.columns) - declared),
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        dims = ", ".join(f"{s.split('_')[0]}={len(self.frame(s))}" for s in REQUIRED_SHEETS)
        return f"Dataset({self.source.name}: {dims})"


def load(path: str | Path, strict: bool = True) -> Dataset:
    """Read the workbook, coerce dtypes, validate against the sheet contracts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Workbook not found: {path}\n"
            "This repository ships no real data by design. Run `make synth` to "
            "generate a runnable synthetic workbook, or point --data at your own file."
        )

    raw = pd.read_excel(path, sheet_name=list(REQUIRED_SHEETS), engine="openpyxl")
    frames = {sheet: _coerce(df, sheet) for sheet, df in raw.items()}
    warnings = validate_all(frames, strict=strict)

    return Dataset(
        source=path,
        internal=frames[INTERNAL_SHEET],
        external=frames[EXTERNAL_SHEET],
        routes=frames[ROUTE_SHEET],
        materials=frames[MATERIAL_SHEET],
        hubs=frames[HUB_SHEET],
        warnings=warnings,
    )
