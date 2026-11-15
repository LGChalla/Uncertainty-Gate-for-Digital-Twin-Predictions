"""
Governance gates (Phase 7, step 35).

Three checks that every agent output must pass. Each computes a verdict from
numbers rather than arguing for one, and a verdict cannot be overridden: the
orchestrator may revise and resubmit, but it cannot appeal.

The distinction this module exists to make
------------------------------------------
Existing deterministic validators for agent-generated studies check that a
study is well FORMED: that variables exist, that temporal relations cohere,
that the statistical task is well posed. These gates check that a claim is
SUPPORTABLE: that the simulated fields it rests on satisfy conservation of
mass, and that the data can identify the parameters it cites. A study can be
perfectly well formed and rest on a simulation that violates physics.

A language model appears nowhere in a verdict. Gates may be given a wrapper
that phrases a failure for the orchestrator, but the pass/fail and the measured
numbers come only from computation, so the same input always yields the same
verdict. That property is what the audit trail rests on, and it is what a
language-model judge cannot offer.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math


@dataclass(frozen=True)
class Verdict:
    """One gate's decision. passed is computed, never argued."""
    gate: str
    passed: bool
    reason: str
    measured: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.gate}: {'PASS' if self.passed else 'VETO'} - {self.reason}"


@dataclass
class Claim:
    """
    A structured agent output offered for checking.

    kind       what sort of claim: calibration, cohort, or effect
    payload    the numbers being asserted
    provenance what produced them: code version, config hash, seed, data hash,
               and the analysis specification hash the protocol gate compares
    """
    kind: str
    payload: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def digest(self):
        blob = json.dumps({"kind": self.kind, "payload": self.payload,
                           "provenance": self.provenance}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Gate:
    name = "gate"

    def check(self, claim) -> Verdict:
        raise NotImplementedError

    def _ok(self, reason, **measured):
        return Verdict(self.name, True, reason, measured)

    def _veto(self, reason, **measured):
        return Verdict(self.name, False, reason, measured)


class PhysicsGate(Gate):
    """
    The calibrated twin's fields must satisfy the equation that generated them.

    Checks, in order of how fundamental the violation is:
      1. every concentration is finite and non-negative
      2. every parameter lies inside its prior bounds
      3. mass balance: the change in total field matches production minus
         clearance, to tolerance
      4. the residual of the governing equation is below solver tolerance

    A claim carrying no field diagnostics passes with that stated, rather than
    silently: a gate that quietly approves what it cannot see is worse than no
    gate.
    """
    name = "physics"

    def __init__(self, residual_tol=1e-8, mass_tol=1e-6, bounds=None):
        self.residual_tol = residual_tol
        self.mass_tol = mass_tol
        self.bounds = bounds or {}

    def check(self, claim):
        p = claim.payload
        m = {}

        mins = p.get("field_minima", {})
        for name, v in mins.items():
            if v is None or not math.isfinite(v):
                return self._veto(f"{name} contains a non-finite value", field=name, value=v)
            if v < 0:
                return self._veto(f"{name} goes negative ({v:.3g}); concentrations cannot",
                                  field=name, value=v)
        m["worst_field_minimum"] = min(mins.values()) if mins else None

        for name, v in p.get("parameters", {}).items():
            if name not in self.bounds:
                continue
            lo, hi = self.bounds[name]
            if not (lo <= v <= hi):
                return self._veto(f"{name} = {v:.4g} is outside its prior bounds "
                                  f"[{lo:.4g}, {hi:.4g}]", parameter=name, value=v,
                                  low=lo, high=hi)

        mb = p.get("mass_balance_rel_error")
        if mb is not None:
            m["mass_balance_rel_error"] = mb
            if not math.isfinite(mb) or mb > self.mass_tol:
                return self._veto(f"mass balance off by {mb:.3g}, tolerance {self.mass_tol:.3g}",
                                  **m)

        res = p.get("pde_residual")
        if res is not None:
            m["pde_residual"] = res
            if not math.isfinite(res) or res > self.residual_tol:
                return self._veto(f"equation residual {res:.3g} exceeds tolerance "
                                  f"{self.residual_tol:.3g}", **m)

        if not mins and mb is None and res is None:
            return self._ok("no field diagnostics in this claim; physics not checked", **m)
        return self._ok("fields satisfy the governing equation within tolerance", **m)


class IdentifiabilityGate(Gate):
    """
    A claim may only cite parameters the data can actually pin down.

    Two conditions, both required:
      1. every parameter reported per patient is in the identifiable class of
         the decided partition (Sobol sensitivity plus posterior contraction)
      2. its posterior has contracted from its prior by at least min_contraction

    The second matters independently of the first. A parameter can be formally
    identifiable and still barely constrained in a given patient, and reporting
    it then asserts precision the data do not carry.
    """
    name = "identifiability"

    def __init__(self, partition, min_contraction=0.30):
        self.partition = dict(partition)          # name -> patient | cohort | latent
        self.min_contraction = min_contraction

    def check(self, claim):
        reported = claim.payload.get("reported_parameters", [])
        contraction = claim.payload.get("contraction", {})
        unknown, wrong_class, weak = [], [], []

        for name in reported:
            cls = self.partition.get(name)
            if cls is None:
                unknown.append(name)
            elif cls != "patient":
                wrong_class.append((name, cls))
            else:
                c = contraction.get(name)
                if c is None or c < self.min_contraction:
                    weak.append((name, c))

        m = {"reported": list(reported), "min_contraction": self.min_contraction}
        if unknown:
            return self._veto(f"{', '.join(unknown)} not in the identifiability partition; "
                              "no claim can rest on a parameter never classified", **m)
        if wrong_class:
            bad = ", ".join(f"{n} is {c}" for n, c in wrong_class)
            return self._veto(f"claim rests on a parameter that is not patient-identifiable "
                              f"({bad})", **m)
        if weak:
            bad = ", ".join(f"{n} contracted {0.0 if c is None else c:.2f}" for n, c in weak)
            return self._veto(f"posterior too close to the prior to report per patient "
                              f"({bad}, need {self.min_contraction:.2f})", **m)
        return self._ok("every reported parameter is identifiable and sufficiently "
                        "contracted", **m)


class ProtocolGate(Gate):
    """
    The analysis actually run must be the one pre-registered, and the cohort
    counts must add up.

    The specification hash is the whole point: it makes a silent change to the
    estimand, the eligibility rules, the outcome or the model impossible to
    miss, because any edit changes the hash. Flow reconciliation catches the
    separate failure of patients appearing or vanishing between filter steps.
    """
    name = "protocol"

    def __init__(self, registered_hash, flow_tolerance=0):
        self.registered_hash = registered_hash
        self.flow_tolerance = flow_tolerance

    def check(self, claim):
        got = claim.provenance.get("spec_hash")
        m = {"registered": self.registered_hash, "submitted": got}
        if got is None:
            return self._veto("claim carries no analysis specification hash", **m)
        if got != self.registered_hash:
            changed = claim.provenance.get("spec_changed", "unspecified")
            return self._veto(f"analysis specification does not match the pre-registered one "
                              f"(changed: {changed})", **m)

        flow = claim.payload.get("flow")
        if flow:
            start, excluded, final = flow.get("start"), flow.get("excluded"), flow.get("final")
            if None in (start, excluded, final):
                return self._veto("flow counts incomplete; cannot reconcile", flow=flow, **m)
            gap = abs((start - excluded) - final)
            m["flow_gap"] = gap
            if gap > self.flow_tolerance:
                return self._veto(f"flow counts do not reconcile: {start} - {excluded} "
                                  f"!= {final} (off by {gap})", flow=flow, **m)
        return self._ok("specification matches the pre-registration and counts reconcile", **m)


def run_gates(claim, gates):
    """
    Every gate sees every claim, and all verdicts are returned.

    We do not stop at the first veto. A claim can fail for several independent
    reasons, and an orchestrator that fixes only the first would resubmit
    something still broken. The run log keeps all of them.
    """
    verdicts = [g.check(claim) for g in gates]
    return verdicts, all(v.passed for v in verdicts)
