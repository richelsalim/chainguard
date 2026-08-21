# Methodology

Everything the model does, why it does it that way, and where it is wrong.

---

## 1. The objective

The challenge specifies a single scalarised objective:

> minimise `0.40 × lead time + 0.40 × cost/kg + 0.20 × risk`

Implemented in [`scoring.py`](../src/chainguard/scoring.py) as

$$
s_{ij} \;=\; w_L\,\hat L_{ij} \;+\; w_C\,\hat C_{ij} \;+\; w_R\,\hat R_{ij} \;+\; w_G\,\hat G_{ij}
$$

for shipment $i$ and candidate route $j$, where every $\hat{\cdot}$ is a min-max
normalisation **within shipment $i$'s own feasible candidate pool**:

$$
\hat X_{ij} \;=\; \frac{X_{ij} - \min_{k \in J_i} X_{ik}}{\max_{k \in J_i} X_{ik} - \min_{k \in J_i} X_{ik}}
$$

### Why normalise per shipment rather than globally

The score has to answer *"how good is this route for this shipment"*. A shipment
whose only options are all slow should be judged on whether the plan picked the
best of what it actually had — not penalised for the network's geography. Per-pool
normalisation also makes 0.0 mean "best available" and 1.0 "worst available" for
every shipment, which is what makes averaging scores across a heterogeneous
shipment population meaningful at all.

Degenerate pools (one candidate, or all candidates equal on an attribute)
normalise to 0.0 rather than `NaN`: with nothing to choose between, no option is
worse than another.

### What this does *not* imply

It does **not** follow that some route scores exactly 0. That would require one
route to be simultaneously best on lead time, cost *and* risk. Real candidate
pools rarely contain such a dominating option, so the blended minimum is
strictly positive whenever the attributes disagree — which is the normal case,
and the reason the problem is interesting. This is asserted in
`tests/test_scoring.py::test_blended_minimum_is_zero_only_when_one_route_dominates`.

### Cost per kilogram

The internal sheet counts **pieces**; the objective is denominated in **cost per
kilogram**. External deliveries carry both a chargeable weight and a piece count,
so they are the bridge:

$$
\text{kg/unit}_f = \operatorname{median}_{d \in D_f}\!\left(\frac{\text{ChargeableWeight\_KG}_d}{\text{Pieces}_d}\right)
$$

per material family $f$. Families never seen in the external sheet fall back to
the **global median**, not to 1.0 — the latter would silently inflate their cost
per kg by orders of magnitude and quietly distort 40% of the objective.

Route cost is freight plus handling at both ends:

$$
C_{ij} = \text{BaseCostEUR}_j + \sum_{h \in \{o_j, d_j\}}\!\left(\text{Fixed}_h + \text{Variable}_h \cdot q_i\right)
$$

---

## 2. Hard feasibility gates

Optimisation is only as trustworthy as its constraint set. A cheap route that
cannot legally carry the material is not a saving, it is a defect. All ten gates
in [`feasibility.py`](../src/chainguard/feasibility.py) are hard filters applied
*before* any score is computed, and every rejection is counted by reason — the
**gate ledger** is a first-class output, not a debug print.

| # | Gate | Rule |
|---|------|------|
| 1 | Lane | route must serve the shipment's `StageFrom → StageTo` |
| 2 | Family | route must be published for the shipment's `MaterialFamily` |
| 3 | Availability | `AvailableFlag = Yes` |
| 4 | Scenario | route's `DisruptionScenario` must be active in this run |
| 5 | Primary | excluded in a primary-hub-down drill |
| 6 | Cold chain | cold-chain materials need cold-capable hubs at **both** ends |
| 7 | Hazard | both hubs must declare support for the material's hazard class |
| 8 | Route capacity | route weekly capacity ≥ shipment (or lot) quantity |
| 9 | Hub headroom | both hubs must have residual weekly headroom |
| 10 | Shelf life | lead time ≤ the material's shelf life |

The ledger counts rejections attributable to each gate **in isolation**, which is
what a planner wants to know ("what is blocking me"), rather than as a sequential
funnel that hides overlapping causes behind whichever gate happens to run first.

### Hub headroom — and a corrected bug

$$
K_h \;=\; C_h \cdot \Bigl(u^{\max}_h - r_h\,\mathbb{1}[\text{disrupted}] - u^{\text{cur}}_h - b\Bigr)^{+}
$$

with weekly capacity $C_h$, utilisation ceiling $u^{\max}$, scenario capacity
reduction $r_h$, current utilisation $u^{\text{cur}}$ and planning buffer $b$.

This is **linear** in the utilisation ceiling. An earlier iteration of this model
applied the ceiling twice —

```text
headroom = C · (u_max − r) · u_max − C · u_cur      # wrong
```

— which understated every hub's headroom by roughly 10% and caused the capacity
gate to reject feasible routes. The concrete regression case is pinned in
`tests/test_feasibility.py::test_headroom_is_linear_in_the_utilisation_ceiling`:
with `C = 10 000`, `u_max = 0.9`, `u_cur = 0.5`, correct headroom is **4 000**;
the buggy form returns **3 100**.

### A second correctness fix: intra-stage legs

Intra-stage lanes (`Backend → Backend`) can have `FromHub == ToHub` in a source
workbook. The solver charges each **distinct** hub a leg touches, once. An
earlier audit routine charged origin and destination unconditionally, so it
double-counted those legs and reported violations the solver had correctly not
created — an audit that contradicted the constraint it was meant to verify. Both
sides now use the same `hub_load` function, so the audit cannot drift from the
model again.

---

## 3. Why greedy is not enough

Per-shipment greedy — score every feasible route, take the argmin — is the
standard answer, and it is per-shipment optimal. It is also **wrong at the
network level**, for one specific reason:

> **Hub capacity is a shared resource and greedy treats it as private.**

Every shipment independently picks the same handful of cheap, fast, low-risk
hubs, and nothing in the loop notices that their combined volume has blown
through those hubs' weekly headroom. The plan looks excellent on paper and cannot
be executed.

On the bundled synthetic network across all seven scenarios, greedy over-books
**100 hub-scenario pairs by 242,120 units**, and **not one** of its seven plans is
executable.

### The fair baseline

Comparing the MILP to *raw* greedy would be a rigged fight: greedy is cheaper
because it is illegal. So the benchmark's real baseline is
[`optimize/repair.py`](../src/chainguard/optimize/repair.py) — greedy plus
**min-regret capacity repair**, which is what a planner does by hand: find the
worst-overloaded hub, price the score penalty of moving each shipment off it to
its best legal alternative, execute the cheapest move, repeat; drop a shipment
only when it has no legal alternative.

That produces an executable plan from a legitimate heuristic, and it is the bar
the MILP has to clear.

---

## 4. The MILP

[`optimize/milp.py`](../src/chainguard/optimize/milp.py), solved with CP-SAT.

**Decision variables** — one boolean per (candidate, lot); $z$ marks a dropped lot:

$$
x_{ij\ell} \in \{0,1\}, \qquad z_{i\ell} \in \{0,1\}
$$

**Objective** — total normalised score plus a drop penalty:

$$
\min \; \sum_{i}\sum_{j \in J_i}\sum_{\ell} \frac{s_{ij}}{L}\, x_{ij\ell} \;+\; \frac{M}{L}\sum_{i}\sum_{\ell} z_{i\ell}
$$

**Constraints**

$$
\begin{aligned}
\sum_{j \in J_i} x_{ij\ell} + z_{i\ell} &= 1 && \forall i, \ell &&\text{(each lot placed or dropped)}\\
z_{i\ell} &\ge z_{i,\ell-1} && \forall i, \ell>1 &&\text{(symmetry breaking)}\\
\sum_{(i,j,\ell)\,:\,h \in \{o_j,d_j\}} q_{i\ell}\, x_{ij\ell} &\le K_h && \forall h \in H &&\textbf{(shared hub headroom)}\\
\sum_{i,\ell} q_{i\ell}\, x_{ij\ell} &\le U_j && \forall j \in R &&\text{(route weekly capacity)}
\end{aligned}
$$

### The hub constraint is the entire point

It is the only thing coupling shipments to each other. **Drop it and the program
decomposes into $|S|$ independent argmins — that is, it *becomes* greedy.** Keep
it and the solver must trade a slightly worse route for one shipment against a
much better one for another: a multi-dimensional generalised assignment problem,
NP-hard in general.

This is not an argument, it is a test.
`tests/test_optimize.py::test_milp_without_capacity_reproduces_greedy` runs the
MILP with `enforce_hub_capacity=False` and asserts the objective equals greedy's
to within 1e-3. If the two ever diverge, the difference is a bug in scoring, not
a benefit of optimisation.

### Why CP-SAT

Pure-integer model with knapsack-style side constraints and no continuous
relaxation of interest. CP-SAT's portfolio search suits that shape, ships in a
permissively licensed wheel needing no separate solver install, proves optimality
on instances of this size in milliseconds, and — decisive for a public repo —
anyone can `pip install ortools` and reproduce the numbers exactly.

### On the drop penalty $M$

Set to `10 × score_scale`, well above the worst achievable assigned score of 1.0,
so dropping is always worse than any placement. Modelling drops **explicitly**
rather than declaring the instance infeasible means an over-subscribed network
still returns the best partial plan *and names what it could not place* — the
answer a planner actually needs during a disruption.

**Caveat, stated plainly:** because $M \gg \max s_{ij}$, the penalised objective
is dominated by coverage. Two plans with the same coverage are separated by route
quality; plans with different coverage are separated mostly by the count of
dropped shipments. That is the intended ordering — an unplaced shipment really is
worse than a mediocre route — but it means the objective should always be read
next to `coverage` and `mean_score`, which is why the benchmark reports all three.

### Split-shipment relaxation

The single biggest cause of unplaceable shipments is a **large quantity meeting a
small residual headroom** — an all-or-nothing constraint no route choice can
satisfy. Setting `max_splits = L > 1` divides a shipment into $L$ balanced lots
that may travel separately; its score becomes the volume-weighted mean of the
routes used, so a split is only chosen when it genuinely helps.

This is operationally meaningful, not a modelling convenience: two trucks instead
of one is what a planner already does.

The gates must use the same $L$. They test one lot's worth of volume, otherwise
`build_candidates` discards routes the optimiser could legally have used and the
relaxation silently does nothing. `build_candidates(..., max_splits=L)` and
`solve(..., max_splits=L)` are meant to be passed the same value.

### Chance constraint

`min_on_time_probability = α` admits a route for a shipment only if its Monte
Carlo on-time probability is at least α.

**A service level is a constraint, not a preference.** Blended into the objective,
a 40% cost term can outvote it — see §5. As a constraint it holds by
construction, and its price, in both objective value and lost coverage, becomes
directly measurable.

---

## 5. Monte Carlo, and a negative result worth keeping

`RiskScore` is an ordinal number in [0, 5]. It ranks routes, which is useful, but
"route A has risk 3.2" has no operational meaning. "Route A delivers on time 78%
of the time, and in the worst 5% of weeks it is 6.1 days late" does.

For a route with deterministic lead time $\mu$ and risk $r$:

- **Baseline variability** — $T_0 \sim \mathrm{Gamma}(k,\theta)$ with
  $\mathbb{E}[T_0] = \mu$ and $\mathrm{CV}(r) = c_0 + c_1 r$. Gamma is the right
  family: strictly positive (transit time cannot be negative), right-skewed
  (delays have a long tail, early arrivals do not), and closed under addition, so
  summing legs of a multi-leg path stays in the family.
- **Disruption events** — Bernoulli$(p_0 + p_1 r)$ fires an extra
  $\mathrm{Exp}(\lambda)$ delay. This produces the heavy tail a pure Gamma
  understates, and is where the risk score does its real work.

Reported: P50 / P90 / P95, on-time probability against the promise plus SLA
slack, expected delay, and **CVaR₉₅** — the mean lead time *conditional on being
in the worst 5% of outcomes*. CVaR separates a route that is usually fine and
occasionally catastrophic from one that is reliably mediocre; expected value
cannot, and unlike a raw percentile CVaR is a coherent risk measure.

### The negative result

The obvious next step is to substitute CVaR for lead time in the objective and
call the plan risk-aware. **That does not work, and the failure is instructive.**

Both terms are min-max normalised inside each shipment's candidate pool, and CVaR
compresses differently from mean lead time at the low end. A route that is
second-best on lead time can sit *relatively* closer to the front on normalised
CVaR, and the 40% cost term then tips the choice toward it. Some decisions flip
to routes with a **worse** tail.

Measured on the bundled network: **~8% of routing decisions flip, and mean CVaR₉₅
moves the wrong way by about 0.27 days.**

The finding is kept, reproduced by `chainguard simulate`, and documented on
`simulate.risk_adjusted_ranking` rather than quietly deleted, because it is the
argument for where the service target belongs — the feasible region, not the
objective. The chance-constrained variant raises mean on-time probability from
**84.1% to 91.8%**, by construction, and reports exactly what that guarantee
costs in coverage.

---

## 6. Network graph

Per-leg optimisation answers "best way from SIFO to Backend". Planners ask "best
way from front-end to partner hand-off", and the answers differ — the cheapest
first leg routinely lands material at a hub whose onward options are terrible.

[`network.py`](../src/chainguard/network.py) builds the hub-level directed graph,
one edge per feasible route, weighted by the same objective but normalised
**globally** — path weights are only additive if every edge is on one scale.

- `best_path` / `k_best_paths` — minimum-weight and $k$-cheapest loopless paths
  (Yen's algorithm). Parallel edges collapse to the best one per hub pair, which
  loses nothing: only the best-scoring parallel edge can be on an optimal path.
- `best_stage_path` — best path over the canonical `FE → SIFO → Backend → OSAT`
  chain for a material family, using a virtual super-source/sink so "any origin to
  any destination" is one shortest-path call instead of $|S| \times |T|$.
- `critical_hubs` — betweenness centrality. A hub with high betweenness and low
  headroom is the network's most dangerous asset: cheap to overlook, expensive to
  lose. **No per-leg model can see these**, because no individual leg looks unusual.

Path risk **compounds** rather than averaging:

$$
R_{\text{path}} \;=\; 1 - \prod_{\ell \in \text{legs}} \left(1 - \tfrac{r_\ell}{5}\right)
$$

Summing or averaging raw risk scores would understate a long path's true exposure.

---

## 7. Known limitations

Stated because a model whose limits are undocumented is a model whose limits are
unknown.

1. **Independence in the simulation.** Legs are drawn independently. Real
   disruptions correlate — one typhoon delays every leg through a region. The
   plan-level "all legs on time" figure is therefore optimistic in shape (a
   correlated model would have a fatter joint tail) even though it already reads
   as pessimistically small, since independent 84% legs compound fast over ~190
   legs. A copula or a shared regional shock factor is the right fix.
2. **Weekly static capacity.** Hub headroom is a single weekly number with no time
   phasing. Real capacity is daily and lumpy; a plan feasible for the week can be
   infeasible on Tuesday. Fixing this means time-indexed capacity variables.
3. **Risk-score calibration is assumed, not fitted.** The mapping from
   `RiskScore` to CV and disruption probability is a plausible linear form with
   documented constants in `config.SimulationConfig` — it is not estimated from
   realised delays. The internal sheet has `TransitDelayDays`, so fitting it is
   possible and is the highest-value next step.
4. **Split lots are balanced.** The relaxation divides quantity into equal parts.
   Real splits are sized to the capacity actually available; a continuous
   volume-variable formulation would dominate, at the cost of a harder integer
   program.
5. **Solve time grows with lots.** `max_splits=3` is milliseconds to ~35s
   depending on scenario; beyond 4 the lot encoding gets expensive and CP-SAT
   starts returning `FEASIBLE` rather than `OPTIMAL` inside the time limit. The
   benchmark reports the status column so a truncated solve is never mistaken for
   a proven one.
6. **Single-period, no inventory.** No stock on hand, no backorders, no
   multi-period smoothing. This is a routing model, not a planning system.
