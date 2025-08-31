# soilcalc/solver.py
from __future__ import annotations
"""
Dependency-graph solver with *measurement-suggestion* pass
and **symbolic reachability** so we never recommend
measuring something algebra can already supply.

Stage 1  – Uniform-cost search (UCS) using only algebraic
           providers (exact providers cost = 0).
Stage 2a – Purely *symbolic* forward sweep that finds every
           parameter derivable with **no** measurements.
Stage 2b – If UCS still needs inputs, suggest laboratory /
           in-situ tests **excluding** the symbols from 2a.

The public API is unchanged; extra data appear on
`experimental_plans`, `measurement_choices`, and
`_no_measure_needed` after calling `.solve()`.
"""
from typing import Dict, List, Set, DefaultDict, Optional, Tuple
from collections import defaultdict
import heapq
from dataclasses import dataclass
from itertools import product

from .registry import REG, Registry
from .providers import EquationProvider
from .measurements import Measurement
from .params import Param, PARAM_UNITS
from .path_utils import prefer_direct_measurement, find_all_missing_paths
from . import provider_stats as stats          # local stats helper

# ──────────────────────────────────────────────────────────────────────
# Sentinel used to mark a parameter that is “assumed available but unknown”
# ──────────────────────────────────────────────────────────────────────
class _Dummy(float):
    """`float` subclass so the dummy can flow through SymPy without errors."""
    def __repr__(self) -> str:                 # ⇢ prints a nice ✓ in debug traces
        return "★dummy★"
DUMMY = _Dummy("nan")                          # `nan` so numeric ops survive

# ──────────────────────────────────────────────────────────────────────
# Helper: degrees‑of‑freedom based on current context
# ──────────────────────────────────────────────────────────────────────

def _degrees_of_freedom(
    ctx: Set[str],                       # user-supplied values so far
    inferable: Set[str],                 # algebraically derivable w/out tests
    registry: Registry,
    *,                                   # force keyword-only for extras
    verbose: bool = False,               # print the set for debugging
) -> Set[str]:
    """
    Return every parameter that is **not** already in *ctx* **and** cannot be
    derived purely algebraically (i.e. not in *inferable*), *yet* appears in at
    least one non-measurement provider (either as an input or an output).

    These symbols are the open “degrees of freedom” (DOF) that a measurement
    plan must resolve.
    """
    known = ctx | inferable
    dof: Set[str] = set()

    for prov in registry.providers():
        # Measurement providers don’t participate in algebraic chains
        if getattr(prov, "type", "") == "measurement":
            continue
        # union of .provides and .required picks up both forward & inverse roles
        for sym in prov.provides | prov.required():
            if sym not in known:
                dof.add(sym)

    print("[DEBUG] DOF symbols:", sorted(dof))

    return dof

# ──────────────────────────────────────────────────────────────────────
# NEW helper – purely symbolic reachability (no measurements)
# ──────────────────────────────────────────────────────────────────────
def _infer_no_measurement(
    reg: Registry,
    seeds: Set[str],
    *,
    verbose: bool = True,           # ← turn off to silence the trace
) -> Set[str]:
    """
    Return every symbol that can be produced without calling a Measurement.

    If *verbose* is True the function prints a running trace that shows
    • the starting seed set,
    • each provider that “fires”,
    • the symbols newly added on that pass,
    • and the final inferable set.
    """
    inferable: Set[str] = set(seeds)
    if verbose:
        print("▶ symbolic reachability")
        print("  seeds:", sorted(seeds))

    algebraic = list(reg.providers())      # Registry.providers() excludes Measurement
    pass_no = 0
    changed = True
    while changed:
        changed = False
        pass_no += 1
        if verbose:
            print(f"  ── sweep #{pass_no}")

        for prov in algebraic:
            if prov.required().issubset(inferable):
                new_syms = prov.provides - inferable
                if new_syms:
                    inferable |= new_syms
                    changed = True
                    if verbose:
                        out = ", ".join(sorted(new_syms))
                        print(f"    + {out:25s} via {prov.name}")

    if verbose:
        print("  inferable (no-test):", sorted(inferable))
        print("▲ end symbolic reachability\n")

    return inferable


@dataclass
class ExperimentalPlan:
    algebraic: List[tuple[str, EquationProvider]]      # skeleton path
    measurements: Dict[str, Measurement]               # test per param


# ──────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────
def _measurement_combos(choices: Dict[str, List[Measurement]]
                        ) -> List[Dict[str, Measurement]]:
    """
    Expand every possible selection (Cartesian product) of measurement options.
    """
    if not choices:
        return [{}]
    params, option_lists = zip(*choices.items())
    combos: List[Dict[str, Measurement]] = []
    for combo in product(*option_lists):
        combos.append({p: m for p, m in zip(params, combo)})
    return combos


def _fmt_with_unit(symbol: str, value, sig_figs: int = 4) -> str:
    try:
        unit = PARAM_UNITS[Param(symbol)]
    except ValueError:
        unit = ""
    if isinstance(value, (int, float)):
        v_str = f"{value:.{sig_figs}g}"
    else:
        v_str = str(value)
    return f"{v_str} {unit}".rstrip()


def _dedupe_path(path: List[tuple[str, EquationProvider]]
                 ) -> List[tuple[str, EquationProvider]]:
    seen: Set[str] = set()
    out: List[tuple[str, EquationProvider]] = []
    for sym, prov in path:
        if sym not in seen:
            out.append((sym, prov))
            seen.add(sym)
    return out


class NoPathError(RuntimeError):
    def __init__(self, message: str, missing: Optional[Dict[str, Set[str]]] = None):
        super().__init__(message)
        self.missing = missing or {}


def _suggest_measurements(reg: Registry, missing: Set[str]) -> Dict[str, List[Measurement]]:
    """Return a *param → [Measurement]* mapping ordered by (price, duration)."""
    out: Dict[str, List[Measurement]] = {}
    for p in missing:
        opts = sorted(reg.measurements(p), key=lambda m: (m.price, m.duration))
        out[p] = opts
    return out


# ──────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────
class DependencyGraphSolver:
    """Find a feasible algebraic path, then flag inputs needing tests."""

    def __init__(self, registry: Registry = REG):
        self.registry = registry
        self.trace: List[str] = []
        self.alternatives: List[List[tuple[str, EquationProvider]]] = []
        self.measurement_choices: Dict[str, List[Measurement]] = {}
        self.experimental_plans: List[ExperimentalPlan] = []
        self._no_measure_needed: Set[str] = set()      # new public-ish field

        # speed-up: symbol → providers able to supply it
        self._providers_for: DefaultDict[str, List[EquationProvider]] = defaultdict(list)
        for prov in self.registry.providers():
            for out in prov.provides:
                self._providers_for[out].append(prov)

    # ──────────────────────────────────────────────────────────────────
    # Uniform-cost search (uses prov.cost())
    # ──────────────────────────────────────────────────────────────────
    def find_execution_order(
        self,
        known: Set[str],
        want: Set[str]
    ) -> List[tuple[str, EquationProvider]]:
        start_have = frozenset(known)
        frontier: List[Tuple[float, frozenset[str], List[tuple[str, EquationProvider]]]] = [
            (0.0, start_have, [])
        ]
        best_cost: Dict[frozenset[str], float] = {start_have: 0.0}
        goal_cost: Optional[float] = None

        while frontier:
            cost, have_fs, plan = heapq.heappop(frontier)
            if cost > best_cost.get(have_fs, float("inf")):
                continue
            have = set(have_fs)

            if want <= have:
                if goal_cost is None:
                    goal_cost = cost
                    self.alternatives = [plan]
                elif cost == goal_cost:
                    self.alternatives.append(plan)
                else:
                    break
                continue

            if goal_cost is not None and cost > goal_cost:
                break

            for prov in self._providers_for_all(have):
                out = next(iter(prov.provides))
                reqs = prov.required()
                if not reqs <= have:
                    continue
                new_have = frozenset(have | {out})
                new_cost = cost + prov.cost()
                if new_cost < best_cost.get(new_have, float("inf")):
                    best_cost[new_have] = new_cost
                    heapq.heappush(
                        frontier,
                        (new_cost, new_have, plan + [(out, prov)])
                    )

        if not self.alternatives:
            raise NoPathError("UCS failed – no algebraic path")

        return self.alternatives[0]

    def _providers_for_all(self, have: Set[str]) -> List[EquationProvider]:
        return [
            p for p in self.registry.providers()
            if next(iter(p.provides)) not in have
        ]

    # ──────────────────────────────────────────────────────────────────
    # Greedy back-chaining (if UCS fails) – unchanged except for
    # “skip inferable symbols” injection later.
    # ──────────────────────────────────────────────────────────────────
    def _backchain_plan(
            self,
            target_syms: Set[str],
            known: Set[str]
    ) -> List[tuple[str, EquationProvider]]:
        plan: List[tuple[str, EquationProvider]] = []
        produced: Set[str] = set()
        visiting: Set[str] = set()

        def add(sym: str):
            if sym in known or sym in produced or sym in visiting:
                return
            visiting.add(sym)

            provs = [
                p for p in self._providers_for.get(sym, [])
                if sym not in p.required()
            ]
            if not provs:
                visiting.remove(sym)
                return

            prov = min(
                provs,
                key=lambda p: (p.cost(), len(p.required() - produced - known))
            )

            for req in prov.required():
                add(req)

            plan.append((sym, prov))
            produced.add(sym)
            visiting.remove(sym)

        for s in target_syms:
            add(s)

        return _dedupe_path(plan)

    # ──────────────────────────────────────────────────────────────────
    # Public API – `.solve()`
    # ──────────────────────────────────────────────────────────────────
    def solve(
        self,
        ctx: Dict[str, float],
        *targets: str,
        suggest_measurements: bool = True,
    ) -> Tuple[Dict[str, float], List[tuple[str, EquationProvider]], float]:

        # ─── Step 0 – apply any per-provider cost overrides hiding in ctx ───
        for key, val in list(ctx.items()):
            if key.startswith("delta_cost:"):
                prov_name = key.split(":", 1)[1]
                try:
                    override = float(val)
                except (TypeError, ValueError):
                    continue
                for prov in self.registry.providers():
                    if prov.name == prov_name:
                        prov.delta_cost = override
                        break
                del ctx[key]

        want = set(targets) - set(ctx)
        if not want:                                  # everything already supplied
            return ctx, [], 0.0

        # ─── Step ½ – purely symbolic reachability (no measurements) ───
        self._no_measure_needed = _infer_no_measurement(self.registry, set(ctx))

        # ─── Step 1 – attempt a fully algebraic execution plan (UCS) ───
        try:
            plan = self.find_execution_order(set(ctx), want)

            # prune unused zero-cost providers
            needed: Set[str] = set(want)
            pruned_rev: List[tuple[str, EquationProvider]] = []
            for sym, prov in reversed(plan):
                if sym in needed or prov.cost() > 0:
                    pruned_rev.append((sym, prov))
                    needed |= prov.required()
            plan = list(reversed(pruned_rev))

            # execute providers to fill ctx
            for sym, prov in plan:
                prov.apply(ctx)

            return ctx, plan, sum(p.cost() for _, p in plan)

        # ─── Step 2 – UCS failed → fall back to measurement planning ───
        except NoPathError:
            # 2a: forward-closure algebraic
            real_ctx = ctx.copy()
            closure_chain: List[tuple[str, EquationProvider]] = []
            changed = True
            while changed:
                changed = False
                for prov in self.registry.providers():
                    if prov.required() <= set(real_ctx):
                        before = set(real_ctx)
                        prov.apply(real_ctx)
                        new_syms = set(real_ctx) - before
                        if new_syms:
                            changed = True
                            closure_chain.extend((s, prov) for s in new_syms)

            # 2b: build DOF and find feasible DOFs via dummy-UCS
            dof = _degrees_of_freedom(
                set(real_ctx), self._no_measure_needed, self.registry
            )
            feasible: Set[str] = set()
            for meas in self.registry.measurements():
                sym_set = set(getattr(meas, "param", []))
                if sym_set.isdisjoint(dof):
                    continue
                trial_ctx = real_ctx.copy()
                trial_ctx.update({s: DUMMY for s in sym_set})
                try:
                    self.find_execution_order(set(trial_ctx), want)
                except NoPathError:
                    continue
                feasible |= sym_set
            dof &= feasible

            # Initialize missing_paths and experimental_plans
            self.missing_paths = []
            self.experimental_plans = []

            # 2c: enumerate measurement + algebraic-tail alternatives
            for meas in self.registry.measurements():
                sym_set = set(getattr(meas, "param", []))
                if sym_set.isdisjoint(dof):
                    continue

                trial_ctx = real_ctx.copy()
                trial_ctx.update({s: DUMMY for s in sym_set})

                try:
                    alg_chain = self.find_execution_order(set(trial_ctx), want)
                except NoPathError:
                    continue

                # Back-propagate needed symbols through alg_chain
                needed = set(want)
                for s, prov in reversed(alg_chain):
                    if s in needed:
                        needed.remove(s)
                        needed |= prov.required()
                if sym_set.isdisjoint(needed):
                    continue

                # Adapter for measurement as a zero-cost provider
                class _MeasAdapter:
                    def __init__(self, m, outs: Set[str]):
                        self._m = m
                        self.name = m.method
                        self.type = "measurement"
                        self.provides = outs
                    def cost(self) -> float:
                        return 0.0
                    def required(self) -> Set[str]:
                        return set()
                    def apply(self, ctx: Dict[str, float]) -> None:
                        for s in self.provides:
                            ctx[s] = DUMMY

                # Build full chain and record missing_paths
                meas_step = (next(iter(sym_set)), _MeasAdapter(meas, sym_set))
                full_chain = _dedupe_path([meas_step] + alg_chain)
                self.missing_paths.append((full_chain, sym_set))

                # Build ExperimentalPlan for graphing
                meas_dict = {s: meas for s in sym_set}
                alg_only = [(s, p) for s, p in full_chain if getattr(p, "type", "") != "measurement"]
                ep = ExperimentalPlan(algebraic=alg_only, measurements=meas_dict)
                self.experimental_plans.append(ep)

            return real_ctx, closure_chain, 0.0