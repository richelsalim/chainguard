"""Chainguard control tower — a Dash app over the optimisation engine.

Run:  python app/dashboard.py --data data/synthetic.xlsx
Then: http://127.0.0.1:8050

The dashboard is a *view*, not a second implementation. Every number it renders
comes from the same library functions the CLI and the test suite call, so there
is exactly one place where the model lives and no chance of the screen and the
benchmark disagreeing.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import theme as th  # noqa: E402

from chainguard.benchmark import _penalised_objective  # noqa: E402
from chainguard.config import DEFAULT_SOLVER, SCENARIOS  # noqa: E402
from chainguard.feasibility import build_candidates  # noqa: E402
from chainguard.loader import load  # noqa: E402
from chainguard.network import build_network  # noqa: E402
from chainguard.optimize import capacity_violations, solve_greedy, solve_milp  # noqa: E402
from chainguard.optimize import repair as repair_mod  # noqa: E402
from chainguard.simulate import service_level_summary, simulate_plan  # noqa: E402

DATA_PATH = "data/synthetic.xlsx"
SLA_GRID = (0.0, 0.70, 0.80, 0.85, 0.90, 0.95)


# ---------------------------------------------------------------------------
# Model access (cached — a scenario is solved once per session)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def dataset():
    return load(DATA_PATH)


@lru_cache(maxsize=16)
def solve_scenario(key: str, max_splits: int, min_otd: float | None) -> dict:
    ds = dataset()
    scenario = SCENARIOS[key]
    cs = build_candidates(ds, scenario, max_splits=max_splits)

    greedy = solve_greedy(cs, scenario.weights)
    repaired = repair_mod.solve(cs, scenario.weights)
    optimal = solve_milp(
        cs, scenario.weights, max_splits=max_splits, min_on_time_probability=min_otd
    )

    penalty = DEFAULT_SOLVER.unassigned_penalty_multiplier
    plans = {"Greedy": greedy, "Greedy + repair": repaired, "MILP (global)": optimal}
    rows = []
    for label, plan in plans.items():
        summary = plan.summary()
        violations = capacity_violations(plan, cs)
        rows.append(
            {
                "method": label,
                "objective": _penalised_objective(summary, penalty),
                "mean_score": summary["mean_score"],
                "coverage": summary["coverage"],
                "assigned": summary["shipments"],
                "violations": len(violations),
                "excess": float(violations["excess_units"].sum()) if len(violations) else 0.0,
                "executable": len(violations) == 0,
                "seconds": plan.solve_seconds,
            }
        )

    per_ship = optimal.per_shipment()
    sim = simulate_plan(per_ship, keep_draws=True)

    return {
        "scenario": scenario,
        "candidates": cs,
        "comparison": pd.DataFrame(rows),
        "plan": optimal,
        "per_shipment": per_ship,
        "simulation": sim,
        "service": service_level_summary(sim),
        "network": build_network(ds, scenario, scenario.weights),
    }


@lru_cache(maxsize=8)
def service_frontier(key: str) -> pd.DataFrame:
    """Coverage vs achieved service level as the chance constraint tightens."""
    ds = dataset()
    scenario = SCENARIOS[key]
    cs = build_candidates(ds, scenario)
    rows = []
    for alpha in SLA_GRID:
        plan = solve_milp(cs, scenario.weights, min_on_time_probability=alpha or None)
        per = plan.per_shipment()
        svc = service_level_summary(simulate_plan(per))
        rows.append(
            {
                "target": alpha,
                "coverage": plan.summary()["coverage"],
                "achieved_otd": svc["mean_on_time_probability"],
                "assigned": plan.summary()["shipments"],
                "cvar": svc["mean_cvar95_days"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_method_comparison(comparison: pd.DataFrame, t: th.Theme) -> go.Figure:
    """Horizontal bars — one measure, colour reserved for executability.

    Lower is better, so greedy's short bar is the trap the chart exists to expose:
    the colour and the explicit label say it cannot be executed.
    """
    df = comparison.iloc[::-1]  # best method ends up on top
    colours = [t.status["good"] if e else t.status["critical"] for e in df["executable"]]
    labels = [
        f"{v:.3f}  {'✓ executable' if e else '✕ over capacity'}"
        for v, e in zip(df["objective"], df["executable"], strict=False)
    ]

    fig = go.Figure(
        go.Bar(
            x=df["objective"],
            y=df["method"],
            orientation="h",
            marker={"color": colours, "line": {"color": t.surface, "width": 2}},
            text=labels,
            textposition="outside",
            textfont={"color": t.text_primary, "size": 12},
            customdata=np.stack([df["coverage"], df["assigned"], df["excess"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>penalised objective %{x:.4f}"
                "<br>coverage %{customdata[0]:.1%}"
                "<br>%{customdata[1]} shipments assigned"
                "<br>%{customdata[2]:,.0f} units over capacity<extra></extra>"
            ),
        )
    )
    layout = t.plotly_layout(height=240)
    layout["title"]["text"] = "Penalised objective by method — lower is better"
    layout["xaxis"]["range"] = [0, max(df["objective"].max() * 1.42, 0.1)]
    layout["yaxis"]["automargin"] = True
    layout["showlegend"] = False
    layout["margin"] = {"l": 12, "r": 24, "t": 44, "b": 24}
    fig.update_layout(**layout)
    return fig


def fig_network_map(result: dict, t: th.Theme) -> go.Figure:
    """Hub network in geographic coordinates — sequential colour = headroom used.

    Deliberately *not* a basemap chart. Plotly's ``Scattergeo`` fetches its
    coastline topojson from a CDN at render time, so on a locked-down network the
    map silently renders as an empty rectangle. Since the question here is "which
    hubs is the plan loading, and how hard", the land outline was never carrying
    information — so this plots true longitude/latitude on a fixed aspect ratio,
    labels the regional clusters directly, and depends on nothing external.
    """
    net = result["network"]
    cs = result["candidates"]
    plan = result["plan"]

    hubs = pd.DataFrame(
        [
            {
                "hub": n,
                "lat": d.get("lat"),
                "lon": d.get("lon"),
                "stage": d.get("stage"),
                "city": d.get("city"),
                "country": d.get("country"),
                "cluster": d.get("cluster"),
            }
            for n, d in net.graph.nodes(data=True)
        ]
    ).dropna(subset=["lat", "lon"])
    if hubs.empty:
        return _empty(t, "No hub coordinates in this dataset")

    from chainguard.optimize.greedy import hub_load

    load_by_hub = hub_load(plan.assignments) if not plan.assignments.empty else pd.Series(dtype=float)
    hubs["committed"] = hubs["hub"].map(load_by_hub).fillna(0.0)
    hubs["headroom"] = hubs["hub"].map(cs.headroom).fillna(0.0)
    hubs["utilisation"] = (
        (hubs["committed"] / hubs["headroom"].replace(0.0, np.nan)).fillna(0.0).clip(0, 1)
    )

    used = hubs[hubs["committed"] > 0]
    idle = hubs[hubs["committed"] == 0]
    fig = go.Figure()

    # Planned legs first, so hub markers sit above them.
    if not plan.assignments.empty:
        coords = hubs.set_index("hub")[["lat", "lon"]]
        lats: list[float | None] = []
        lons: list[float | None] = []
        for _, row in plan.assignments.iterrows():
            if row["from_hub"] in coords.index and row["to_hub"] in coords.index:
                a, b = coords.loc[row["from_hub"]], coords.loc[row["to_hub"]]
                lons += [a["lon"], b["lon"], None]
                lats += [a["lat"], b["lat"], None]
        fig.add_trace(
            go.Scatter(
                x=lons, y=lats, mode="lines",
                line={"width": 1, "color": t.categorical[0]},
                opacity=0.25, hoverinfo="skip", name="Planned legs",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=idle["lon"], y=idle["lat"], mode="markers",
            marker={"size": 6, "color": t.muted, "opacity": 0.30, "line": {"width": 0}},
            name="Unused hubs",
            text=idle["hub"] + " · " + idle["city"].fillna(""),
            hovertemplate="<b>%{text}</b><br>not used by this plan<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=used["lon"], y=used["lat"], mode="markers",
            marker={
                "size": 11,
                "color": used["utilisation"],
                "colorscale": [
                    [i / (len(t.sequential) - 1), c] for i, c in enumerate(t.sequential)
                ],
                "cmin": 0, "cmax": 1,
                "line": {"width": 2, "color": t.surface},
                "colorbar": {
                    "title": {"text": "Headroom<br>used", "font": {"color": t.text_secondary, "size": 11}},
                    "tickformat": ".0%",
                    "tickfont": {"color": t.muted, "size": 11},
                    "outlinewidth": 0, "thickness": 10, "len": 0.65, "x": 1.01,
                },
            },
            name="Hubs carrying volume",
            customdata=np.stack([used["committed"], used["headroom"], used["utilisation"]], axis=-1),
            text=used["hub"] + " · " + used["city"].fillna("") + ", " + used["country"].fillna(""),
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} units"
                "<br>%{customdata[2]:.1%} of headroom<extra></extra>"
            ),
        )
    )

    layout = t.plotly_layout(height=430)
    layout["title"]["text"] = "Where the optimal plan puts volume"
    layout["xaxis"].update(
        {"title": {"text": "longitude", "font": {"color": t.muted, "size": 11}},
         "range": [-135, 160], "zeroline": False, "dtick": 30}
    )
    layout["yaxis"].update(
        {"title": {"text": "latitude", "font": {"color": t.muted, "size": 11}},
         "range": [-20, 65], "zeroline": False, "dtick": 20,
         "scaleanchor": "x", "scaleratio": 1}
    )
    layout["legend"] = {
        "orientation": "h", "yanchor": "top", "y": -0.14, "x": 0,
        "font": {"color": t.text_secondary, "size": 12}, "bgcolor": "rgba(0,0,0,0)",
    }
    layout["margin"] = {"l": 48, "r": 12, "t": 44, "b": 64}
    fig.update_layout(**layout)

    # Direct labels on the regional clusters replace the missing coastlines.
    for cluster, group in hubs.groupby("cluster"):
        if not cluster or len(group) < 3:
            continue
        fig.add_annotation(
            x=float(group["lon"].mean()), y=float(group["lat"].max()) + 7.0,
            text=str(cluster), showarrow=False,
            font={"color": t.muted, "size": 11},
        )
    return fig


def fig_lead_time_distribution(result: dict, t: th.Theme) -> go.Figure:
    """Simulated lead-time distribution with the planning percentiles marked."""
    sim = result["simulation"]
    if sim.draws is None or sim.metrics.empty:
        return _empty(t, "No simulated legs")

    draws = sim.draws.ravel()
    metrics = sim.metrics
    p50 = float(np.percentile(draws, 50))
    p90 = float(np.percentile(draws, 90))
    cvar = float(metrics["cvar95"].mean())

    fig = go.Figure(
        go.Histogram(
            x=draws,
            nbinsx=70,
            marker={"color": t.categorical[0], "line": {"width": 0}},
            hovertemplate="%{x:.1f} days<br>%{y:,} draws<extra></extra>",
            name="Simulated lead time",
        )
    )
    # These three markers routinely land within a day of each other, so their
    # labels go in paper space at distinct heights rather than stacked on the plot
    # line, where they overlap into an unreadable smear.
    for rank, (value, label, colour) in enumerate(
        (
            (p50, "P50", t.text_secondary),
            (p90, "P90", t.status["warning"]),
            (cvar, "CVaR₉₅", t.status["critical"]),
        )
    ):
        fig.add_vline(x=value, line={"color": colour, "width": 2, "dash": "dot"})
        fig.add_annotation(
            x=value, y=1.0 - rank * 0.14, xref="x", yref="paper",
            text=f"  {label} {value:.1f}d", showarrow=False,
            xanchor="left", yanchor="top",
            font={"color": colour, "size": 11},
        )

    layout = t.plotly_layout(height=300)
    layout["title"]["text"] = (
        f"Lead time — {sim.config.n_draws:,} draws × {len(metrics)} legs"
    )
    layout["xaxis"]["title"]["text"] = "days"
    layout["yaxis"]["title"]["text"] = "draws"
    # The Gamma-plus-shock tail runs past 40 days on a handful of draws; plotting
    # all of it compresses the actual mass into a sliver.
    layout["xaxis"]["range"] = [0, float(np.percentile(draws, 99.5)) * 1.2]
    layout["bargap"] = 0.02
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def fig_service_frontier(key: str, t: th.Theme) -> go.Figure:
    """The price of a service guarantee: achieved on-time rate vs coverage lost."""
    df = service_frontier(key)
    fig = go.Figure(
        go.Scatter(
            x=df["achieved_otd"],
            y=df["coverage"],
            mode="lines+markers+text",
            line={"color": t.categorical[1], "width": 2},
            marker={"size": 9, "color": t.categorical[1],
                    "line": {"width": 2, "color": t.surface}},
            text=[("none" if a == 0 else f"≥{a:.0%}") for a in df["target"]],
            textposition="bottom left",
            textfont={"color": t.text_secondary, "size": 11},
            hovertemplate=(
                "target %{text}<br>achieved on-time %{x:.1%}"
                "<br>coverage %{y:.1%}<extra></extra>"
            ),
            name="Service constraint",
        )
    )
    layout = t.plotly_layout(height=300)
    layout["title"]["text"] = "Price of a service guarantee"
    layout["xaxis"]["title"]["text"] = "achieved mean on-time probability"
    layout["xaxis"]["tickformat"] = ".0%"
    layout["yaxis"]["title"]["text"] = "shipment coverage"
    layout["yaxis"]["tickformat"] = ".0%"
    layout["showlegend"] = False
    layout["margin"] = {"l": 12, "r": 20, "t": 44, "b": 44}
    span = df["achieved_otd"].max() - df["achieved_otd"].min()
    layout["xaxis"]["range"] = [df["achieved_otd"].min() - span * 0.10, df["achieved_otd"].max() + span * 0.14]
    fig.update_layout(**layout)
    return fig


def _empty(t: th.Theme, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"color": t.muted, "size": 13})
    layout = t.plotly_layout(height=240)
    layout["xaxis"]["visible"] = False
    layout["yaxis"]["visible"] = False
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def stat_tile(label: str, value: str, note: str, tone: str = "neutral") -> html.Div:
    return html.Div(
        className=f"tile tone-{tone}",
        children=[
            html.Div(label, className="tile-label"),
            html.Div(value, className="tile-value"),
            html.Div(note, className="tile-note"),
        ],
    )


def data_table(df: pd.DataFrame, formats: dict | None = None) -> html.Table:
    formats = formats or {}
    header = html.Tr([html.Th(c.replace("_", " ")) for c in df.columns])
    rows = [
        html.Tr([html.Td(formats.get(c, str)(r[c])) for c in df.columns])
        for _, r in df.iterrows()
    ]
    return html.Table([html.Thead(header), html.Tbody(rows)], className="data-table")


STYLES = """
:root { color-scheme: light; }
*, *::before, *::after { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.app { min-height: 100vh; padding: 28px clamp(16px, 4vw, 48px) 64px; }
.app.light { background: #f9f9f7; color: #0b0b0b; color-scheme: light; }
.app.dark  { background: #0d0d0d; color: #ffffff; color-scheme: dark; }
.app.light .card { background: #fcfcfb; border-color: rgba(11,11,11,.10); }
.app.dark  .card { background: #1a1a19; border-color: rgba(255,255,255,.10); }
.app.light .tile-label, .app.light .tile-note, .app.light .data-table th { color: #52514e; }
.app.dark  .tile-label, .app.dark  .tile-note, .app.dark  .data-table th { color: #c3c2b7; }
.app.light .data-table td { border-top-color: #e1e0d9; }
.app.dark  .data-table td { border-top-color: #2c2c2a; }

header { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end;
         justify-content: space-between; margin-bottom: 22px; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -.01em; }
.sub { font-size: 13px; opacity: .72; margin: 0; max-width: 62ch; }
.controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.control { min-width: 190px; font-size: 13px; }
.control label { display: block; font-size: 11px; text-transform: uppercase;
                 letter-spacing: .06em; opacity: .6; margin-bottom: 4px; }
button.toggle { border-radius: 8px; padding: 8px 14px; font-size: 13px; cursor: pointer;
                background: transparent; color: inherit; border: 1px solid currentColor;
                opacity: .75; }
button.toggle:hover { opacity: 1; }

.card { border: 1px solid; border-radius: 14px; padding: 18px 20px; margin-bottom: 18px; }
.tiles { display: grid; gap: 14px; margin-bottom: 18px;
         grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); }
.tile { border: 1px solid; border-radius: 14px; padding: 16px 18px; }
.app.light .tile { background: #fcfcfb; border-color: rgba(11,11,11,.10); }
.app.dark  .tile { background: #1a1a19; border-color: rgba(255,255,255,.10); }
.tile-label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
.tile-value { font-size: 30px; font-weight: 600; margin: 6px 0 2px; letter-spacing: -.02em; }
.tile-note { font-size: 12px; }
.tone-good .tile-value { color: #0ca30c; }
.tone-critical .tile-value { color: #d03b3b; }
.tone-warning .tile-value { color: #fab219; }

.grid-2 { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(390px, 1fr)); }
.card h2 { font-size: 14px; margin: 0 0 4px; letter-spacing: -.005em; }
.card p.note { font-size: 12.5px; opacity: .72; margin: 0 0 14px; max-width: 68ch; line-height: 1.5; }

.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px;
              font-variant-numeric: tabular-nums; }
.data-table th { text-align: left; font-weight: 500; font-size: 11px;
                 text-transform: uppercase; letter-spacing: .05em; padding: 0 10px 8px 0; }
.data-table td { padding: 7px 10px 7px 0; border-top: 1px solid; }
footer { font-size: 12px; opacity: .6; margin-top: 26px; line-height: 1.6; }

/* This Dash version ships its own dropdown (a <button class="dash-dropdown">
   plus a body-level portal), not react-select. Without these rules the control
   renders white-on-white in dark mode. The menu is portalled outside .app, so
   the theme class is mirrored onto <body> by a clientside callback and the menu
   rules hang off that. */
.dash-dropdown, .dash-dropdown-content, .dash-dropdown-search {
  border-radius: 8px !important;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
  font-size: 13px !important;
}
body.light .dash-dropdown, body.light .dash-dropdown-content {
  background: #fcfcfb !important; color: #0b0b0b !important;
  border: 1px solid rgba(11,11,11,.18) !important;
}
body.dark .dash-dropdown, body.dark .dash-dropdown-content {
  background: #1a1a19 !important; color: #ffffff !important;
  border: 1px solid rgba(255,255,255,.20) !important;
}
body.dark .dash-dropdown-value, body.dark .dash-dropdown-value-item,
body.dark .dash-dropdown-option, body.dark .dash-dropdown-search {
  color: #ffffff !important; background: transparent !important;
}
body.light .dash-dropdown-value, body.light .dash-dropdown-value-item,
body.light .dash-dropdown-option, body.light .dash-dropdown-search {
  color: #0b0b0b !important; background: transparent !important;
}
.dash-dropdown-trigger-icon { color: currentColor !important; opacity: .6; }
body.dark .dash-dropdown-option:hover,
body.dark .dash-dropdown-option.selected { background: #2c2c2a !important; }
body.light .dash-dropdown-option:hover,
body.light .dash-dropdown-option.selected { background: #f0efec !important; }
"""


def build_app(data_path: str) -> Dash:
    global DATA_PATH
    DATA_PATH = data_path

    app = Dash(__name__, title="Chainguard control tower")
    app.index_string = f"""<!DOCTYPE html>
<html><head>{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<style>{STYLES}</style></head>
<body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"""

    app.layout = html.Div(
        id="root",
        className="app light",
        children=[
            dcc.Store(id="theme-store", data="light"),
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("Chainguard — resilient route control tower"),
                            html.P(
                                "Global MILP route assignment under shared hub capacity, with "
                                "Monte Carlo service levels. Every figure below is produced by "
                                "the same library the CLI and test suite call.",
                                className="sub",
                            ),
                        ]
                    ),
                    html.Div(
                        className="controls",
                        children=[
                            html.Div(
                                className="control",
                                children=[
                                    html.Label("Disruption scenario"),
                                    dcc.Dropdown(
                                        id="scenario",
                                        className="dash-dropdown",
                                        options=[
                                            {"label": s.label, "value": k} for k, s in SCENARIOS.items()
                                        ],
                                        value="baseline",
                                        clearable=False,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control",
                                children=[
                                    html.Label("Max lots per shipment"),
                                    dcc.Dropdown(
                                        id="splits",
                                        className="dash-dropdown",
                                        options=[{"label": f"{n}", "value": n} for n in (1, 2, 3)],
                                        value=1,
                                        clearable=False,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="control",
                                children=[
                                    html.Label("Min on-time probability"),
                                    dcc.Dropdown(
                                        id="min-otd",
                                        className="dash-dropdown",
                                        options=[{"label": "no constraint", "value": 0}]
                                        + [{"label": f"≥ {a:.0%}", "value": a} for a in (0.75, 0.85, 0.90)],
                                        value=0,
                                        clearable=False,
                                    ),
                                ],
                            ),
                            html.Button("◐ Theme", id="theme-toggle", className="toggle", n_clicks=0),
                        ],
                    ),
                ]
            ),
            html.Div(id="tiles", className="tiles"),
            html.Div(
                className="card",
                children=[
                    html.H2("Greedy is cheaper because it is illegal"),
                    html.P(
                        "Per-shipment greedy posts the best objective in this chart and produces "
                        "a plan that cannot be executed: it books shared hub capacity that does "
                        "not exist. The fair comparison is against greedy plus capacity repair — "
                        "what a planner does by hand — and that is the bar the MILP has to clear.",
                        className="note",
                    ),
                    dcc.Graph(id="comparison", config={"displayModeBar": False}),
                ],
            ),
            html.Div(
                className="card",
                children=[dcc.Graph(id="map", config={"displayModeBar": False})],
            ),
            html.Div(
                className="grid-2",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.H2("Lead time is a distribution, not a number"),
                            html.P(
                                "The route table promises a deterministic lead time. Simulating "
                                "each leg turns that promise into a service level: how often it "
                                "actually holds, and how bad the worst 5% of weeks are.",
                                className="note",
                            ),
                            dcc.Graph(id="distribution", config={"displayModeBar": False}),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.H2("What a service guarantee costs"),
                            html.P(
                                "Tightening the chance constraint raises the achieved on-time rate "
                                "and shrinks the set of shipments that can be placed at all. This "
                                "frontier is the trade-off, priced rather than argued about.",
                                className="note",
                            ),
                            dcc.Graph(id="frontier", config={"displayModeBar": False}),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="grid-2",
                children=[
                    html.Div(
                        className="card",
                        children=[
                            html.H2("Why routes were rejected"),
                            html.P(
                                "Each hard gate, counted independently — what is actually "
                                "constraining the network under this scenario.",
                                className="note",
                            ),
                            html.Div(id="ledger"),
                        ],
                    ),
                    html.Div(
                        className="card",
                        children=[
                            html.H2("Structural chokepoints"),
                            html.P(
                                "Hubs ranked by betweenness centrality — the share of viable "
                                "paths that must cross them. A per-leg model cannot see these.",
                                className="note",
                            ),
                            html.Div(id="chokepoints"),
                        ],
                    ),
                ],
            ),
            html.Footer(
                [
                    html.Div(f"Data source: {data_path} — synthetic unless you supplied your own."),
                    html.Div(
                        "This project ships no real supply-chain data. Run `make synth` to "
                        "regenerate the fabricated workbook."
                    ),
                ]
            ),
        ],
    )

    _register_callbacks(app)
    return app


def _register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("theme-store", "data"),
        Input("theme-toggle", "n_clicks"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_theme(_clicks, current):
        return "dark" if current == "light" else "light"

    @app.callback(Output("root", "className"), Input("theme-store", "data"))
    def set_theme_class(name):
        return f"app {name}"

    # Dropdown menus are portalled to <body>, outside the themed .app subtree,
    # so the theme has to be mirrored onto <body> for their CSS to apply.
    app.clientside_callback(
        "function(theme) { document.body.className = theme; return window.dash_clientside.no_update; }",
        Output("theme-store", "id"),
        Input("theme-store", "data"),
    )

    @app.callback(
        Output("tiles", "children"),
        Output("comparison", "figure"),
        Output("map", "figure"),
        Output("distribution", "figure"),
        Output("frontier", "figure"),
        Output("ledger", "children"),
        Output("chokepoints", "children"),
        Input("scenario", "value"),
        Input("splits", "value"),
        Input("min-otd", "value"),
        Input("theme-store", "data"),
    )
    def render(scenario_key, splits, min_otd, theme_name):
        t = th.get(theme_name)
        result = solve_scenario(scenario_key, int(splits), float(min_otd) or None)

        comparison = result["comparison"]
        milp = comparison[comparison["method"] == "MILP (global)"].iloc[0]
        greedy = comparison[comparison["method"] == "Greedy"].iloc[0]
        repaired = comparison[comparison["method"] == "Greedy + repair"].iloc[0]
        service = result["service"]

        gain = (
            (repaired["objective"] - milp["objective"]) / repaired["objective"] * 100
            if repaired["objective"]
            else 0.0
        )

        tiles = [
            stat_tile(
                "MILP objective",
                f"{milp['objective']:.3f}",
                f"{abs(gain):.1f}% better than repaired greedy" if gain >= 0 else f"{abs(gain):.1f}% worse than repaired greedy",
                "good" if gain >= 0 else "warning",
            ),
            stat_tile("Coverage", f"{milp['coverage']:.1%}", f"{int(milp['assigned'])} shipments placed"),
            stat_tile(
                "Greedy overload",
                f"{int(greedy['excess']):,}",
                f"units over capacity at {int(greedy['violations'])} hubs",
                "critical" if greedy["excess"] > 0 else "good",
            ),
            stat_tile(
                "Capacity violations",
                f"{int(milp['violations'])}",
                "MILP plan, by construction",
                "good" if milp["violations"] == 0 else "critical",
            ),
            stat_tile(
                "Mean on-time",
                f"{service['mean_on_time_probability']:.1%}",
                f"CVaR₉₅ {service['mean_cvar95_days']:.1f} days",
            ),
            stat_tile("Solve time", f"{milp['seconds']*1000:,.0f} ms", result["plan"].solver_status.lower()),
        ]

        ledger = result["candidates"].ledger
        ledger = ledger[ledger["rejected_pairs"] > 0].head(8)
        chokepoints = result["network"].critical_hubs(8)[
            ["hub", "stage", "country", "betweenness"]
        ]

        return (
            tiles,
            fig_method_comparison(comparison, t),
            fig_network_map(result, t),
            fig_lead_time_distribution(result, t),
            fig_service_frontier(scenario_key, t),
            data_table(ledger, {"rejected_pairs": lambda v: f"{int(v):,}"}),
            data_table(chokepoints, {"betweenness": lambda v: f"{v:.4f}"}),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Chainguard control tower")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = build_app(args.data)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
