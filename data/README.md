# `data/` — bring your own workbook

**This repository ships no real supply-chain data.** The dataset used in the
Infineon SC Challenge 2026 is the organiser's property and is not redistributed
here, in anonymised form or otherwise. `data/` is gitignored except for this file.

## Option 1 — run on synthetic data (recommended, zero setup)

```bash
make synth          # or: chainguard synth --out data/synthetic.xlsx --seed 42
```

This generates `data/synthetic.xlsx`: a fully fictional workbook with the same
five-sheet schema, the same column names and the same order-of-magnitude
distributions as the challenge dataset. Every hub, material, forwarder and
shipment ID is fabricated. It is enough to exercise the entire pipeline —
feasibility gates, MILP, Monte Carlo, the network graph and the dashboard.

## Option 2 — supply your own workbook

Drop an `.xlsx` with these five sheets into `data/` and point the CLI at it:

```bash
chainguard optimize --data data/your_workbook.xlsx
```

| Sheet | Purpose |
|---|---|
| `Internal_Shipments` | Internal movement legs (FE → SIFO → Backend → OSAT) |
| `External Shipments` | Customer deliveries linked back to internal legs |
| `Route_Options` | Candidate routes with lead time, cost, risk, CO₂, availability |
| `Hub_Constraints` | Hub capacity, utilisation, handling capabilities, geo coordinates |
| `Material_Families` | Material grouping with hazard class, temperature and priority |

The exact required columns, types and validation rules are in
[`docs/DATA_SCHEMA.md`](../docs/DATA_SCHEMA.md). The loader validates on read and
tells you precisely which column is missing rather than failing deep in a solve.
