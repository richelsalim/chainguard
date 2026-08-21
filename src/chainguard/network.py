"""End-to-end multi-leg network model.

The per-leg optimisers answer "what is the best way to move this shipment from
SIFO to Backend". That is a local question. The question a supply-chain planner
actually asks is **"what is the best way to get this material from front-end all
the way to the partner hand-off"** — and the answers differ, because the cheapest
first leg routinely lands the material at a hub whose onward options are terrible.

This module builds the network as a directed graph over hubs, with one edge per
feasible route option, and optimises the *whole path*. Two things fall out of it
that a per-leg model cannot produce:

* **Path-level optima.** ``best_path`` returns the minimum-cost end-to-end chain;
  ``k_best_paths`` enumerates the k cheapest, which is what you show a planner
  who wants options rather than an oracle.
* **Structural risk.** Betweenness centrality over the feasible graph identifies
  the hubs that the largest share of viable paths must traverse. Those are the
  single points of failure — and they are invisible to any per-leg model, because
  no individual leg looks unusual.

Edges are weighted by the same 40/40/20 objective the optimisers use, normalised
globally across the graph (rather than per shipment) so that summing weights
along a path is meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from .config import DEFAULT_WEIGHTS, STAGE_ORDER, DisruptionScenario, ObjectiveWeights
from .loader import Dataset

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two coordinates."""
    if any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)), (lat1, lon1, lat2, lon2))):
        return float("nan")
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class RouteNetwork:
    """A hub-level directed graph with scored, geo-enriched edges."""

    graph: nx.MultiDiGraph
    scenario: DisruptionScenario
    weights: ObjectiveWeights

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def stats(self) -> dict:
        simple = nx.DiGraph(self.graph)
        return {
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "density": round(nx.density(simple), 6),
            "weakly_connected_components": nx.number_weakly_connected_components(simple),
            "largest_component": len(max(nx.weakly_connected_components(simple), key=len))
            if self.n_nodes
            else 0,
            "mean_out_degree": round(
                sum(d for _, d in simple.out_degree()) / max(self.n_nodes, 1), 3
            ),
        }

    # -- Path optimisation --------------------------------------------------

    def best_path(self, source: str, target: str) -> dict | None:
        """Minimum-weight end-to-end path, or ``None`` if unreachable."""
        simple = self._simple_view()
        if source not in simple or target not in simple:
            return None
        try:
            nodes = nx.shortest_path(simple, source, target, weight="weight")
        except nx.NetworkXNoPath:
            return None
        return self._describe_path(simple, nodes)

    def k_best_paths(self, source: str, target: str, k: int = 5) -> list[dict]:
        """The k lowest-weight loopless paths, cheapest first (Yen's algorithm)."""
        simple = self._simple_view()
        if source not in simple or target not in simple:
            return []
        out: list[dict] = []
        try:
            generator = nx.shortest_simple_paths(simple, source, target, weight="weight")
            for nodes in generator:
                out.append(self._describe_path(simple, nodes))
                if len(out) >= k:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return out
        return out

    def best_stage_path(self, family: str, stages: tuple[str, ...] = STAGE_ORDER) -> dict | None:
        """Best path that visits the canonical stage chain for one material family.

        Rather than fixing arbitrary endpoint hubs, this searches over every hub
        at the first stage and every hub at the last, which is the question a
        planner means by "how should this family flow through the network".
        """
        simple = self._simple_view(family=family)
        sources = [n for n, d in simple.nodes(data=True) if d.get("stage") == stages[0]]
        targets = [n for n, d in simple.nodes(data=True) if d.get("stage") == stages[-1]]
        if not sources or not targets:
            return None

        # Virtual super-source/sink with zero-weight arcs turns "any source to
        # any target" into a single shortest-path call instead of |S|x|T| calls.
        work = simple.copy()
        work.add_node("__src__", stage="virtual")
        work.add_node("__dst__", stage="virtual")
        for s in sources:
            work.add_edge("__src__", s, weight=0.0)
        for t in targets:
            work.add_edge(t, "__dst__", weight=0.0)
        try:
            nodes = nx.shortest_path(work, "__src__", "__dst__", weight="weight")
        except nx.NetworkXNoPath:
            return None
        return self._describe_path(simple, nodes[1:-1])

    # -- Structural risk ----------------------------------------------------

    def critical_hubs(self, top_n: int = 15) -> pd.DataFrame:
        """Hubs ranked by betweenness centrality — the network's chokepoints.

        Betweenness counts the share of shortest paths that must pass through a
        node. A hub with high betweenness and low headroom is the network's most
        dangerous asset: cheap to overlook, expensive to lose.
        """
        simple = self._simple_view()
        if simple.number_of_nodes() == 0:
            return pd.DataFrame(columns=["hub", "stage", "betweenness", "in_degree", "out_degree"])

        # Sampling keeps this tractable on large graphs; k=None is exact.
        k = None if simple.number_of_nodes() <= 300 else 300
        bc = nx.betweenness_centrality(simple, k=k, weight="weight", seed=42, normalized=True)
        rows = [
            {
                "hub": node,
                "stage": simple.nodes[node].get("stage", ""),
                "country": simple.nodes[node].get("country", ""),
                "betweenness": round(score, 6),
                "in_degree": simple.in_degree(node),
                "out_degree": simple.out_degree(node),
            }
            for node, score in bc.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("betweenness", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def edge_table(self) -> pd.DataFrame:
        """Flatten every edge with its attributes — the map layer's data source."""
        rows = []
        for u, v, data in self.graph.edges(data=True):
            rows.append(
                {
                    "from_hub": u,
                    "to_hub": v,
                    **{k: val for k, val in data.items()},
                    "from_lat": self.graph.nodes[u].get("lat"),
                    "from_lon": self.graph.nodes[u].get("lon"),
                    "to_lat": self.graph.nodes[v].get("lat"),
                    "to_lon": self.graph.nodes[v].get("lon"),
                }
            )
        return pd.DataFrame(rows)

    # -- Internals ----------------------------------------------------------

    def _simple_view(self, family: str | None = None) -> nx.DiGraph:
        """Collapse parallel edges to the single best one, optionally per family.

        ``shortest_path`` and ``shortest_simple_paths`` need a simple graph; when
        several route options connect the same hub pair, only the best-scoring
        one can ever be part of an optimal path, so collapsing loses nothing.
        """
        simple = nx.DiGraph()
        simple.add_nodes_from(self.graph.nodes(data=True))
        for u, v, data in self.graph.edges(data=True):
            if family is not None and data.get("material_family") != family:
                continue
            existing = simple.get_edge_data(u, v)
            if existing is None or data["weight"] < existing["weight"]:
                simple.add_edge(u, v, **data)
        return simple

    @staticmethod
    def _describe_path(simple: nx.DiGraph, nodes: list[str]) -> dict:
        legs = []
        for u, v in zip(nodes, nodes[1:], strict=False):
            d = simple.edges[u, v]
            legs.append(
                {
                    "from_hub": u,
                    "to_hub": v,
                    "route_id": d.get("route_id"),
                    "mode": d.get("mode"),
                    "lead_days": d.get("lead_days"),
                    "cost_eur": d.get("cost_eur"),
                    "risk": d.get("risk"),
                    "co2_kg": d.get("co2_kg"),
                    "weight": d.get("weight"),
                    "distance_km": d.get("distance_km"),
                }
            )
        return {
            "hubs": nodes,
            "legs": legs,
            "n_legs": len(legs),
            "total_weight": round(sum(leg["weight"] for leg in legs), 6),
            "total_lead_days": sum(leg["lead_days"] for leg in legs),
            "total_cost_eur": round(sum(leg["cost_eur"] for leg in legs), 2),
            "total_co2_kg": round(sum(leg["co2_kg"] for leg in legs), 1),
            # Risk compounds along a chain: the probability that *every* leg is
            # clean is the product of the per-leg clean probabilities. Summing
            # raw risk scores would understate a long path's true exposure.
            "path_risk": round(
                1.0 - math.prod(max(0.0, 1.0 - leg["risk"] / 5.0) for leg in legs), 4
            )
            if legs
            else 0.0,
            "total_distance_km": round(
                sum(leg["distance_km"] for leg in legs if leg["distance_km"] == leg["distance_km"]), 1
            ),
        }


def build_network(
    dataset: Dataset,
    scenario: DisruptionScenario,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> RouteNetwork:
    """Assemble the scenario-filtered hub graph with globally normalised weights."""
    routes = dataset.routes.copy()
    routes = routes[routes["AvailableFlag"].astype(str).str.casefold() == "yes"]
    routes = routes[routes["DisruptionScenario"].astype(str).isin(scenario.route_scenarios)]
    if scenario.exclude_primary:
        routes = routes[routes["IsPrimary"].astype(str).str.casefold() != "yes"]

    hubs = dataset.hubs.set_index("HubID")
    graph = nx.MultiDiGraph()

    for hub_id, row in hubs.iterrows():
        graph.add_node(
            hub_id,
            stage=row.get("Stage"),
            city=row.get("City"),
            country=row.get("Country"),
            cluster=row.get("GeoCluster"),
            lat=float(row.get("Latitude")) if pd.notna(row.get("Latitude")) else None,
            lon=float(row.get("Longitude")) if pd.notna(row.get("Longitude")) else None,
            cold_chain=str(row.get("ColdChainAvailable", "")).casefold() == "yes",
            weekly_capacity=float(row.get("WeeklyCapacityUnits", 0.0)),
        )

    if routes.empty:
        return RouteNetwork(graph=graph, scenario=scenario, weights=weights)

    # Global min-max normalisation: path weights are only additive if every edge
    # is measured on the same scale.
    def _norm(series: pd.Series) -> pd.Series:
        lo, hi = float(series.min()), float(series.max())
        if hi <= lo:
            return pd.Series(0.0, index=series.index)
        return (series.astype(float) - lo) / (hi - lo)

    n_lead = _norm(routes["BaseLeadTimeDays"])
    n_cost = _norm(routes["BaseCostEUR"])
    n_risk = _norm(routes["RiskScore"])
    n_co2 = _norm(routes["CO2Kg"])
    edge_weight = (
        weights.lead_time * n_lead
        + weights.cost * n_cost
        + weights.risk * n_risk
        + weights.co2 * n_co2
    )

    for pos, (_, r) in enumerate(routes.iterrows()):
        u, v = r["FromHub"], r["ToHub"]
        if u not in graph or v not in graph:
            continue
        graph.add_edge(
            u,
            v,
            key=r["RouteOptionID"],
            route_id=r["RouteOptionID"],
            material_family=r["MaterialFamily"],
            mode=r["TransportMode"],
            lead_days=float(r["BaseLeadTimeDays"]),
            cost_eur=float(r["BaseCostEUR"]),
            risk=float(r["RiskScore"]),
            co2_kg=float(r["CO2Kg"]),
            capacity=float(r["CapacityUnitsPerWeek"]),
            is_primary=str(r["IsPrimary"]).casefold() == "yes",
            # +1e-6 keeps every weight strictly positive, which Yen's algorithm
            # requires and which stops zero-cost cycles from confusing the search.
            weight=float(edge_weight.iloc[pos]) + 1e-6,
            distance_km=haversine_km(
                graph.nodes[u].get("lat"), graph.nodes[u].get("lon"),
                graph.nodes[v].get("lat"), graph.nodes[v].get("lon"),
            ),
        )

    return RouteNetwork(graph=graph, scenario=scenario, weights=weights)
