# Data schema

Chainguard reads one Excel workbook with five sheets. This document is the
contract; [`schema.py`](../src/chainguard/schema.py) enforces it at load time and
names the offending sheet and column rather than failing deep inside a solve.

**No real data ships with this repository.** `make synth` generates a fully
fabricated workbook against this exact schema — see [`data/README.md`](../data/README.md).

---

## Overview

| Sheet | Grain | Required columns | Role |
|---|---|---|---|
| `Internal_Shipments` | one internal movement leg | 7 | the demand to be routed |
| `External Shipments` | one customer delivery | 4 | links pieces to kilograms |
| `Route_Options` | one candidate route | 15 | the decision space |
| `Hub_Constraints` | one facility | 9 | the shared capacity that couples everything |
| `Material_Families` | one material | 5 | handling requirements and priority |

> The external sheet's name contains a **space**, not an underscore
> (`External Shipments`). That is how the source workbook spells it, so that is
> what the loader expects.

---

## `Internal_Shipments`

The shipments to be routed. One row per leg.

| Column | Type | Required | Notes |
|---|---|---|---|
| `ShipmentID` | text | ✅ | primary key; duplicates are reported |
| `MaterialNo_Anon` | text | ✅ | joins to `Material_Families` |
| `MaterialFamily` | text | ✅ | joins to `Route_Options` |
| `StageFrom` / `StageTo` | text | ✅ | `FE`, `SIFO`, `Backend`, `OSAT` |
| `TransportMode` | text | ✅ | as-planned mode |
| `Qty` | number | ✅ | **pieces**, not kilograms |
| `ShipFrom_Alias` / `ShipTo_Alias` | text | | as-planned hubs |
| `Scenario`, `Incoterm`, `Forwarder_Anon` | text | | descriptive |
| `ShipDate`, `ExpectedArrival`, `ActualArrival` | date | | |
| `LeadTimeDays`, `TransitDelayDays` | number | | realised history; the natural input for calibrating the risk model |
| `RouteRiskScore`, `TemperatureControlled`, `Status`, `TracePath` | | | descriptive |

## `External Shipments`

Customer deliveries. Used for exactly one purpose in the model: converting piece
counts to kilograms so `cost/kg` is well defined.

| Column | Type | Required | Notes |
|---|---|---|---|
| `DeliveryNo` | text | ✅ | primary key |
| `MaterialNo_Anon_Link` | text | ✅ | joins to materials |
| `InternalShipmentID_Link` | text | ✅ | joins to internal legs |
| `ChargeableWeight_KG` | number | ✅ | numerator of kg-per-unit |
| `Pieces` | number | | denominator of kg-per-unit |
| `MaterialFamily_Link` | text | | grouping key for the kg-per-unit median |
| `ItemValueEuro`, `DelVolume_m3`, `NoOfPackages` | number | | descriptive |
| `Flight_No`, `Airport_of_Destination`, `ShipTo_CountryCodeISO` | text | | descriptive |
| `PUP_Date`, `Departure_Date_ETD`, `POD_Date` | date | | descriptive |

Trailing `Unnamed: N` columns from Excel padding are dropped automatically.

## `Route_Options`

The decision space — every candidate the optimiser may choose from.

| Column | Type | Required | Notes |
|---|---|---|---|
| `RouteOptionID` | text | ✅ | primary key |
| `MaterialFamily` | text | ✅ | which family may use this route |
| `StageFrom` / `StageTo` | text | ✅ | the lane served |
| `FromHub` / `ToHub` | text | ✅ | join to `Hub_Constraints.HubID` |
| `TransportMode` | text | ✅ | `Air`, `Ocean`, `Road`, `Courier` |
| `BaseLeadTimeDays` | number | ✅ | objective term 1 |
| `BaseCostEUR` | number | ✅ | objective term 2 (before handling) |
| `CapacityUnitsPerWeek` | number | ✅ | route-level capacity gate |
| `RiskScore` | number | ✅ | 0–5; drives the simulation |
| `CO2Kg` | number | ✅ | sustainability objective term |
| `IsPrimary` | Yes/No | ✅ | the planned route |
| `DisruptionScenario` | text | ✅ | `Normal`, `PrimaryHubDown`, `AirCapacityReduced` |
| `AvailableFlag` | Yes/No | ✅ | live blackout flag |
| `Forwarder_Anon`, `Notes` | text | | descriptive |

## `Hub_Constraints`

Facilities. **The only table that couples shipments to each other**, via shared
weekly capacity — which is what makes this an optimisation problem rather than a
sort.

| Column | Type | Required | Notes |
|---|---|---|---|
| `HubID` | text | ✅ | primary key |
| `Stage` | text | ✅ | `FE`, `SIFO`, `Backend`, `OSAT` |
| `WeeklyCapacityUnits` | number | ✅ | $C_h$ |
| `CurrentUtilizationPct` | fraction | ✅ | $u^{\text{cur}}_h$; `"42%"` strings are coerced |
| `MaxUtilizationPct` | fraction | ✅ | $u^{\max}_h$ |
| `CapacityReductionPct` | fraction | ✅ | $r_h$, applied only when disrupted |
| `DisruptionScenario` | text | ✅ | `None`, `Port congestion`, `Labor shortage`, `Weather disruption` |
| `ColdChainAvailable` | Yes/No | ✅ | cold-chain gate |
| `SupportedHazardClasses` | text | ✅ | `;`-separated list; hazard gate |
| `FixedHandlingCost_EUR` | number | | added to route cost |
| `VariableHandlingCostPerUnit_EUR` | number | | multiplied by quantity |
| `Latitude` / `Longitude` | number | | map view and great-circle distances |
| `City`, `Country`, `GeoCluster` | text | | map labels |
| `ESDHandlingAvailable`, `MoistureControlAvailable`, `LithiumHandlingAvailable` | Yes/No | | descriptive; the gate reads `SupportedHazardClasses` |

## `Material_Families`

| Column | Type | Required | Notes |
|---|---|---|---|
| `MaterialNo_Anon` | text | ✅ | primary key |
| `MaterialFamily` | text | ✅ | joins to routes |
| `HazardClass` | text | ✅ | `ESD Sensitive`, `Moisture Sensitive`, `Lithium Handling`, or `None` |
| `TempRequirement` | text | ✅ | `Ambient` or `Cold Chain` |
| `PriorityClass` | text | ✅ | `Standard`, `Expedite`, `Critical` |
| `ShelfLifeDays` | number | | shelf-life gate; missing values default to unconstrained |
| `ProductGroup`, `SubstitutionGroup` | text | | descriptive |

---

## Referential integrity

The loader validates presence and types. These joins are the model's assumptions,
and the test suite asserts them on generated data:

```text
Internal_Shipments.MaterialNo_Anon      → Material_Families.MaterialNo_Anon
Internal_Shipments.MaterialFamily       → Route_Options.MaterialFamily
Route_Options.FromHub / ToHub           → Hub_Constraints.HubID
External Shipments.InternalShipmentID_Link → Internal_Shipments.ShipmentID
```

A shipment whose family has no published route on its lane is reported as
**unplaceable** with the reason, not silently dropped.

## Checking your own workbook

```bash
chainguard profile --data data/your_workbook.xlsx
```

Prints rows, columns and required-column coverage per sheet. Add `--lenient` to
report problems as warnings instead of raising.
