"""Chart theme: one palette, two validated modes.

The hues below are a validated categorical palette — the slot *ordering* is the
colour-blind-safety mechanism, not decoration, so slots are assigned in fixed
order and never cycled. Adjacent-pair separation clears CVD ΔE ≥ 8 and
normal-vision ΔE ≥ 15 in both modes.

Two rules this module exists to enforce:

* **Status colours are reserved.** ``good``/``critical`` mean executable and
  not-executable. They never stand in for "series 3", and they always ship with a
  text label so meaning never rests on hue alone.
* **Dark mode is selected, not flipped.** The dark steps are chosen against the
  dark surface, not computed from the light ones by inversion.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fixed categorical order. Only the first three are safe for all-pairs forms
# (scatter, choropleth); beyond three, adjacent-only forms (bars, lines).
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]

# Sequential blue, light -> dark. Used for magnitude (hub utilisation).
SEQUENTIAL_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_DARK = ["#184f95", "#256abf", "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6", "#cde2fb"]

# Reserved status palette — never themed, never reused as a series colour.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    plane: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    border: str
    categorical: list[str]
    sequential: list[str]

    @property
    def status(self) -> dict[str, str]:
        return STATUS

    def plotly_layout(self, height: int | None = None) -> dict:
        """Shared layout: recessive chrome, no chart-junk, generous margins."""
        layout = {
            "paper_bgcolor": self.surface,
            "plot_bgcolor": self.surface,
            "font": {
                "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
                "color": self.text_secondary,
                "size": 13,
            },
            "margin": {"l": 12, "r": 12, "t": 40, "b": 12},
            "xaxis": {
                "gridcolor": self.grid,
                "linecolor": self.axis,
                "zerolinecolor": self.axis,
                "tickfont": {"color": self.muted, "size": 12},
                "title": {"font": {"color": self.text_secondary, "size": 12}},
            },
            "yaxis": {
                "gridcolor": self.grid,
                "linecolor": self.axis,
                "zerolinecolor": self.axis,
                "tickfont": {"color": self.muted, "size": 12},
                "title": {"font": {"color": self.text_secondary, "size": 12}},
            },
            "title": {"font": {"color": self.text_primary, "size": 15}, "x": 0, "xanchor": "left"},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
                "font": {"color": self.text_secondary, "size": 12},
                "bgcolor": "rgba(0,0,0,0)",
            },
            "hoverlabel": {
                "bgcolor": self.surface,
                "bordercolor": self.border,
                "font": {"color": self.text_primary, "size": 12},
            },
            "colorway": self.categorical,
        }
        if height:
            layout["height"] = height
        return layout


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    plane="#f9f9f7",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    categorical=CATEGORICAL_LIGHT,
    sequential=SEQUENTIAL_LIGHT,
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    plane="#0d0d0d",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    border="rgba(255,255,255,0.10)",
    categorical=CATEGORICAL_DARK,
    sequential=SEQUENTIAL_DARK,
)

THEMES = {"light": LIGHT, "dark": DARK}


def get(name: str) -> Theme:
    return THEMES.get(name, LIGHT)
