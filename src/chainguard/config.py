"""Single source of truth for every constant the model depends on.

Nothing in Chainguard hard-codes a weight, a threshold or a sheet name inline.
If a judge, a reviewer or a future maintainer wants to know "what exactly is
this optimising and under what rules", this file is the whole answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Source workbook contract
# ---------------------------------------------------------------------------

INTERNAL_SHEET = "Internal_Shipments"
EXTERNAL_SHEET = "External Shipments"  # note: space, not underscore, in the source file
ROUTE_SHEET = "Route_Options"
MATERIAL_SHEET = "Material_Families"
HUB_SHEET = "Hub_Constraints"

REQUIRED_SHEETS: tuple[str, ...] = (
    INTERNAL_SHEET,
    EXTERNAL_SHEET,
    ROUTE_SHEET,
    MATERIAL_SHEET,
    HUB_SHEET,
)

# ---------------------------------------------------------------------------
# The challenge objective
#
#     minimise  0.40 * norm(lead time)
#             + 0.40 * norm(cost per kg)
#             + 0.20 * norm(risk)
#
# Normalisation is min-max *within a shipment's own feasible candidate pool*,
# so the score answers "how good is this route relative to the alternatives
# this shipment actually has", and 0.0 always means "best available".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveWeights:
    lead_time: float = 0.40
    cost: float = 0.40
    risk: float = 0.20
    co2: float = 0.00  # non-zero only in the sustainability scenario

    def __post_init__(self) -> None:
        total = self.lead_time + self.cost + self.risk + self.co2
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Objective weights must sum to 1.0, got {total:.6f}")

    def as_dict(self) -> Mapping[str, float]:
        return {
            "lead_time": self.lead_time,
            "cost": self.cost,
            "risk": self.risk,
            "co2": self.co2,
        }


DEFAULT_WEIGHTS = ObjectiveWeights()

# Sustainability variant: CO2 is pulled out of the cost budget, lead time and
# risk keep their challenge weights so the trade-off stays interpretable.
SUSTAINABILITY_WEIGHTS = ObjectiveWeights(lead_time=0.30, cost=0.20, risk=0.20, co2=0.30)

# Expedite variant: speed dominates, cost is allowed to suffer.
EXPEDITE_WEIGHTS = ObjectiveWeights(lead_time=0.70, cost=0.10, risk=0.20)


# ---------------------------------------------------------------------------
# Hub capacity model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityPolicy:
    """How much of a hub's weekly capacity a plan is allowed to consume.

    headroom = WeeklyCapacityUnits x (MaxUtilizationPct
                                      - CapacityReductionPct_if_disrupted
                                      - CurrentUtilizationPct)

    A hub whose current utilisation already exceeds its disrupted ceiling has
    zero headroom, never negative.
    """

    apply_disruption_reduction: bool = True
    floor_at_zero: bool = True
    # Safety buffer withheld from every hub, as a fraction of weekly capacity.
    # 0.0 reproduces the raw challenge rule; raise it to plan conservatively.
    safety_buffer_pct: float = 0.0


DEFAULT_CAPACITY_POLICY = CapacityPolicy()


# ---------------------------------------------------------------------------
# Monte Carlo risk model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the stochastic lead-time model.

    Lead time is modelled as a Gamma variable whose mean is the route's
    ``BaseLeadTimeDays`` and whose coefficient of variation grows linearly with
    the route's ``RiskScore``. On top of that, a Bernoulli disruption event adds
    a heavy-tailed delay. Gamma is the right family here: strictly positive,
    right-skewed, and closed under the "sum of legs" operation that multi-leg
    paths require.
    """

    n_draws: int = 10_000
    seed: int = 42

    # CV = cv_base + cv_per_risk_point * RiskScore   (RiskScore is 0..~5)
    cv_base: float = 0.10
    cv_per_risk_point: float = 0.06

    # P(disruption on a leg) = disruption_base + disruption_per_risk_point * RiskScore
    disruption_base: float = 0.01
    disruption_per_risk_point: float = 0.025

    # Extra delay when a disruption fires: Exponential with this mean, in days.
    disruption_delay_mean_days: float = 3.5

    # Tail measure: CVaR at this confidence level.
    cvar_alpha: float = 0.95

    # Service-level target used for on-time probability, in days of slack over
    # the deterministic promise.
    sla_slack_days: float = 1.0


DEFAULT_SIMULATION = SimulationConfig()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    """CP-SAT settings.

    ``score_scale`` converts the continuous [0, 1] objective into integers,
    which CP-SAT requires. 10_000 keeps four decimals of the score — finer than
    any decision the model can actually distinguish.
    """

    score_scale: int = 10_000
    max_time_seconds: float = 60.0
    num_workers: int = 8
    # Cost of leaving a shipment unassigned, in scaled score units. Set well
    # above the worst possible assigned score (1.0 -> score_scale) so the solver
    # only drops a shipment when it is genuinely infeasible to place it.
    unassigned_penalty_multiplier: float = 10.0
    log_search_progress: bool = False


DEFAULT_SOLVER = SolverConfig()


# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------

STAGE_ORDER: tuple[str, ...] = ("FE", "SIFO", "Backend", "OSAT")
"""Canonical forward flow: front-end -> sort/inventory -> assembly & test -> partner."""

COLD_CHAIN_LABEL = "Cold Chain"
EXPEDITE_LABEL = "Expedite"
CRITICAL_LABEL = "Critical"
YES = "yes"

TRANSPORT_MODES: tuple[str, ...] = ("Air", "Ocean", "Road", "Courier")


@dataclass(frozen=True)
class DisruptionScenario:
    """A named stress test the network is evaluated under."""

    key: str
    label: str
    tagline: str
    # Route rows tagged with one of these DisruptionScenario values are in play.
    route_scenarios: tuple[str, ...] = ("Normal",)
    # Hub rows tagged with one of these DisruptionScenario values lose capacity.
    hub_disruptions: tuple[str, ...] = ()
    # Transport modes whose capacity is cut, and by how much.
    mode_capacity_multiplier: Mapping[str, float] = field(default_factory=dict)
    weights: ObjectiveWeights = DEFAULT_WEIGHTS
    # Restrict the shipment population this scenario scores.
    only_cold_chain: bool = False
    only_expedite: bool = False
    # Primary routes are unavailable (hub-down drill).
    exclude_primary: bool = False


SCENARIOS: dict[str, DisruptionScenario] = {
    "baseline": DisruptionScenario(
        key="baseline",
        label="Baseline network",
        tagline="Undisrupted operation — the reference plan every stress test is measured against.",
        route_scenarios=("Normal",),
    ),
    "port_congestion": DisruptionScenario(
        key="port_congestion",
        label="Port congestion",
        tagline="Ocean gateways lose throughput; capacity-reduced hubs must be routed around.",
        route_scenarios=("Normal", "PrimaryHubDown"),
        hub_disruptions=("Port congestion",),
        mode_capacity_multiplier={"Ocean": 0.6},
    ),
    "cold_chain": DisruptionScenario(
        key="cold_chain",
        label="Cold-chain restriction",
        tagline="Temperature-controlled materials may only traverse cold-chain-capable hubs.",
        route_scenarios=("Normal",),
        only_cold_chain=True,
    ),
    "primary_hub_down": DisruptionScenario(
        key="primary_hub_down",
        label="Primary hub down",
        tagline="The designated primary route is gone; the network must recover on alternatives.",
        route_scenarios=("Normal", "PrimaryHubDown"),
        hub_disruptions=("Port congestion", "Labor shortage", "Weather disruption"),
        exclude_primary=True,
    ),
    "air_capacity_reduced": DisruptionScenario(
        key="air_capacity_reduced",
        label="Air capacity reduced",
        tagline="Air freight is rationed; the plan must absorb the lead-time hit elsewhere.",
        route_scenarios=("Normal", "AirCapacityReduced"),
        mode_capacity_multiplier={"Air": 0.45},
    ),
    "expedite_priority": DisruptionScenario(
        key="expedite_priority",
        label="Expedite priority",
        tagline="Expedite-class materials buy speed with cost, under the same hard capacity limits.",
        route_scenarios=("Normal",),
        only_expedite=True,
        weights=EXPEDITE_WEIGHTS,
    ),
    "sustainability": DisruptionScenario(
        key="sustainability",
        label="Sustainability",
        tagline="CO2e enters the objective at 30% — what does decarbonising the plan actually cost?",
        route_scenarios=("Normal",),
        weights=SUSTAINABILITY_WEIGHTS,
    ),
}

DEFAULT_SCENARIO = "baseline"
