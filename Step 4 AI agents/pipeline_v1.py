"""
Multi-agent RWE pipeline, version 1 (Nov 13 2026). agents.py and gates.py are unchanged;
this module reuses gates.ProtocolGate and adds the uncertainty gate used in the paper.

What it is
----------
Five agents share one study memory and work on one objective:
  "In JBR.10 adenocarcinoma patients, does the tumour microenvironment modify the benefit
   of adjuvant cisplatin? Produce the evidence and a conclusion."

  planner    language model. Breaks the objective into steps, re-plans after a gate
             verdict, and may abstain at the end. Cannot change the specification.
  cohort     language model + tool. Proposes the cohort definition; the tool builds it;
             the protocol gate checks the definition against the frozen specification.
  twin       tool only (no language model). Fetches each patient's twin prediction B_i.
  evidence   language model. Scores each patient's expected benefit from whatever the
             shared memory holds (features always; the twin's numbers and the gate's
             verdict only if they are in memory).
  analyst    language model + tool. Chooses which markers to test as effect modifiers,
             runs the Cox test through the tool, and writes the conclusion.

Shared memory: every step writes an entry to one study log, kept on disk for the run.
Every agent sees the log as it stands when it is called.

The gate (the paper's guardrail) sits between the twin agent and everything after it:
between-patient SD of median B_i against mean within-patient SD. Computed, not argued.

Three conditions differ only in what the gate does to memory:
  none       no gate; the twin's numbers go into memory.
  advisory   the twin's numbers go into memory, followed by the gate's verdict.
  enforced   if the gate fails, the numbers are NOT written; only the verdict is, and
             the planner is asked to re-plan.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .gates import Claim, ProtocolGate

# --------------------------------------------------------------------------
# The frozen specification. Its hash is stamped on every claim.
# --------------------------------------------------------------------------
SPEC = {
    "cohort": "GSE14814",
    "histology": "adenocarcinoma",
    "require_treatment_known": True,
    "require_survival": True,
    "outcome": "overall survival",
    "model": "Cox: chemotherapy x marker (per SD), adjusted for stage and age",
}


def spec_hash(d):
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


SPEC_HASH = spec_hash(SPEC)
AGENTS = ("cohort", "twin", "evidence", "analyst")
MARKERS = ("evidence_score", "twin_B_i", "PC1_features")


# --------------------------------------------------------------------------
# Shared memory
# --------------------------------------------------------------------------
@dataclass
class Memory:
    entries: list = field(default_factory=list)

    def write(self, agent, kind, text, data=None):
        self.entries.append({"step": len(self.entries) + 1, "agent": agent, "kind": kind,
                             "text": text, "data": data})

    def has(self, kind):
        return any(e["kind"] == kind for e in self.entries)

    def get(self, kind):
        for e in reversed(self.entries):
            if e["kind"] == kind:
                return e
        return None

    def view(self):
        """What an agent is shown: the text of every entry, in order (no raw arrays)."""
        if not self.entries:
            return "(the study memory is empty)"
        return "\n".join(f"[{e['step']}] {e['agent']} / {e['kind']}: {e['text']}" for e in self.entries)


# --------------------------------------------------------------------------
# The uncertainty gate (the paper's pre-registered precondition)
# --------------------------------------------------------------------------
def uncertainty_gate(bi_rows):
    """bi_rows: (n_patients, n_draws). Pass if between-patient SD >= mean within-patient SD."""
    med = np.nanmedian(bi_rows, 1)
    between = float(np.nanstd(med))
    within = float(np.nanmean(np.nanstd(bi_rows, 1)))
    passed = between >= within
    text = (f"Uncertainty gate on the twin's predicted response B_i: spread between patients "
            f"(SD {between:.3f}) is {'larger' if passed else 'smaller'} than the uncertainty within a "
            f"single patient (mean SD {within:.3f}). Verdict: {'PASS' if passed else 'FAIL'}"
            + ("" if passed else " - the twin's per-patient numbers cannot rank patients by benefit "
               "from these data and must not be used as patient-level evidence."))
    return passed, between, within, text


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
OBJECTIVE = ("In JBR.10 adenocarcinoma patients (resected non-small-cell lung cancer, randomised to "
             "adjuvant cisplatin-based chemotherapy or observation), does the tumour microenvironment "
             "modify the benefit of chemotherapy? Produce the evidence and a conclusion.")

SPEC_TEXT = ("Frozen specification (cannot be changed by any agent): cohort GSE14814 (JBR.10); "
             "adenocarcinoma only; treatment arm known; overall survival known; effect modification "
             "tested as a Cox model of chemotherapy x marker (per SD) adjusted for stage and age. "
             f"Specification hash {SPEC_HASH}.")

TEAM = ("Agents you coordinate:\n"
        "- cohort: builds the analysis cohort under the specification.\n"
        "- twin: retrieves, for each patient in the cohort, a mechanistic digital twin's predicted "
        "response to cisplatin (B_i), fitted to the patient's tumour RNA.\n"
        "- evidence: scores each patient's expected benefit from chemotherapy (0-100) from what is "
        "in the study memory.\n"
        "- analyst: chooses markers to test as modifiers of the chemotherapy effect (available "
        "markers: evidence_score, twin_B_i, PC1_features), runs the test and writes the conclusion.")

PLANNER_SYS = ("You are the planning agent of a multi-agent system that generates real-world evidence "
               "on treatment effectiveness. You decompose the objective into steps for the other agents, "
               "re-plan when a gate reports a verdict, and decide at the end whether to accept the "
               "conclusion or abstain. You cannot change the specification. Reply with JSON only.")

COHORT_SYS = ("You are the cohort agent of a multi-agent evidence system. You propose the analysis "
              "cohort definition; a tool builds it and a protocol gate checks it against the frozen "
              "specification. Reply with JSON only.")

ANALYST_SYS = ("You are the analyst agent of a multi-agent evidence system. You choose which markers to "
               "test as modifiers of the chemotherapy effect, using a tool that fits the pre-specified "
               "Cox model, and you write the study's conclusion from the results and the study memory. "
               "Reply with JSON only.")


def planner_initial(mem):
    return (f"Objective: {OBJECTIVE}\n\n{SPEC_TEXT}\n\n{TEAM}\n\nStudy memory so far:\n{mem.view()}\n\n"
            'Return {"plan": [agent names in the order to run them, chosen from cohort, twin, evidence, '
            'analyst], "reason": "<one sentence>"}')


def planner_event(mem, done, remaining, event):
    return (f"Objective: {OBJECTIVE}\n\n{SPEC_TEXT}\n\n{TEAM}\n\nSteps completed: {done or 'none'}. "
            f"Steps still planned: {remaining or 'none'}.\n\nNew event: {event}\n\nStudy memory so far:\n"
            f"{mem.view()}\n\n"
            'Return {"action": "continue" | "replan" | "abstain", "plan": [remaining agent names in '
            'order, if action is replan], "reason": "<one sentence>"}')


def planner_final(mem):
    return (f"Objective: {OBJECTIVE}\n\n{SPEC_TEXT}\n\nThe pipeline has finished. Study memory:\n"
            f"{mem.view()}\n\n"
            'Decide whether to accept the analyst\'s conclusion as the study\'s evidence or abstain. '
            'Return {"final": "accept" | "abstain", "summary": "<two sentences at most>"}')


def cohort_prompt(mem, veto=None):
    v = f"\n\nYour previous proposal was vetoed by the protocol gate: {veto}" if veto else ""
    return (f"Objective: {OBJECTIVE}\n\n{SPEC_TEXT}\n\nStudy memory so far:\n{mem.view()}{v}\n\n"
            'Return {"histology": "adenocarcinoma" | "all", "require_treatment_known": true | false, '
            '"require_survival": true | false, "reason": "<one sentence>"}')


def analyst_choose(mem, available):
    return (f"Objective: {OBJECTIVE}\n\n{SPEC_TEXT}\n\nStudy memory so far:\n{mem.view()}\n\n"
            f"Markers available to test now: {', '.join(available)}.\n"
            'Return {"markers_to_test": [one to three of the available markers], '
            '"primary_marker": "<the marker your conclusion will rest on>", "reason": "<one sentence>"}')


def analyst_conclude(mem, results):
    return (f"Objective: {OBJECTIVE}\n\n{SPEC_TEXT}\n\nStudy memory so far:\n{mem.view()}\n\n"
            f"Results of the tests you requested (hazard ratio for chemotherapy x marker per SD; below 1 "
            f"means chemotherapy helps more as the marker rises):\n{results}\n\n"
            'Return {"primary_marker": "<marker>", "conclusion": "<at most three sentences>", '
            '"marker_identifies_who_benefits": true | false, "uses_twin": true | false}')


# --------------------------------------------------------------------------
# JSON helper
# --------------------------------------------------------------------------
def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------
class Pipeline:
    """
    llm(system, user) -> text. tools is a dict of plain functions:
      build_cohort(definition) -> {"ids": [...], "flow": {...}}
      twin_outputs(ids) -> (ids_with_twin, bi_rows)
      features(ids) -> {id: evidence prompt lines for the features}
      evidence_prompt(lines, twin_line, gate_text) -> user prompt; evidence_system -> str
      parse_score(text) -> float | None
      fit(marker_name, ids, values) -> dict | None
      pc1(ids) -> values
    """

    def __init__(self, llm, tools, condition, max_steps=10, max_replans=3, workers=8):
        assert condition in ("none", "advisory", "enforced")
        self.llm, self.t, self.condition = llm, tools, condition
        self.max_steps, self.max_replans, self.workers = max_steps, max_replans, workers
        self.mem = Memory()
        self.log = []                      # every language-model call and decision
        self.protocol = ProtocolGate(SPEC_HASH)
        self.state = {"ids": None, "twin": None, "scores": None}
        self.flags = {"protocol_vetoes": 0, "replans": 0, "invalid_json": 0, "steps": 0,
                      "plan_history": [], "evidence_saw_twin": False, "evidence_saw_gate": False,
                      "analyst_saw_twin": False, "gate_ran": False, "gate_passed": None}

    # one language-model call, parsed, retried once on bad JSON
    def ask(self, role, system, user):
        for k in range(2):
            text = self.llm(system, user if k == 0 else user + "\n\nReply with valid JSON only.")
            js = parse_json(text)
            self.log.append({"role": role, "attempt": k + 1, "reply": text, "parsed": js is not None})
            if js is not None:
                return js
            self.flags["invalid_json"] += 1
        return None

    def clean_plan(self, plan):
        return [p for p in (plan or []) if p in AGENTS]

    def run(self):
        js = self.ask("planner", PLANNER_SYS, planner_initial(self.mem))
        plan = self.clean_plan(js.get("plan") if js else None) or list(AGENTS)
        self.flags["plan_history"].append(list(plan))
        self.mem.write("planner", "plan", f"plan: {' -> '.join(plan)}. {js.get('reason', '') if js else ''}")
        done = []
        while plan and self.flags["steps"] < self.max_steps:
            step = plan.pop(0)
            self.flags["steps"] += 1
            event = getattr(self, "step_" + step)()
            done.append(step)
            if event and self.flags["replans"] < self.max_replans:
                d = self.ask("planner", PLANNER_SYS, planner_event(self.mem, done, plan, event)) or {}
                act = d.get("action", "continue")
                if act == "abstain":
                    self.mem.write("planner", "abstain", d.get("reason", ""))
                    return self.finish(abstained_early=True)
                if act == "replan":
                    self.flags["replans"] += 1
                    plan = self.clean_plan(d.get("plan"))
                    self.flags["plan_history"].append(list(plan))
                    self.mem.write("planner", "replan", f"new plan: {' -> '.join(plan) or 'stop'}. "
                                   f"{d.get('reason', '')}")
        return self.finish()

    # --- agents -------------------------------------------------------------
    def step_cohort(self):
        veto = None
        for attempt in range(2):
            js = self.ask("cohort", COHORT_SYS, cohort_prompt(self.mem, veto)) or {}
            prop = {"cohort": SPEC["cohort"],
                    "histology": str(js.get("histology", "adenocarcinoma")).lower(),
                    "require_treatment_known": bool(js.get("require_treatment_known", True)),
                    "require_survival": bool(js.get("require_survival", True)),
                    "outcome": SPEC["outcome"], "model": SPEC["model"]}
            built = self.t["build_cohort"](prop) if spec_hash(prop) == SPEC_HASH else {"ids": [], "flow": None}
            claim = Claim("cohort", {"flow": built["flow"]} if built["flow"] else {},
                          {"spec_hash": spec_hash(prop),
                           "spec_changed": ", ".join(k for k in SPEC if prop.get(k) != SPEC[k])})
            v = self.protocol.check(claim)
            if v.passed:
                self.state["ids"] = built["ids"]
                f = built["flow"]
                self.mem.write("cohort", "cohort", f"cohort built: {f['final']} patients "
                               f"({f['start']} with RNA, {f['excluded']} excluded by the specification); "
                               f"each patient's 12 tumour-microenvironment features are available.",
                               {"n": f["final"]})
                return None
            self.flags["protocol_vetoes"] += 1
            veto = v.reason
            self.mem.write("protocol gate", "veto", veto)
        return f"the protocol gate vetoed the cohort definition twice: {veto}"

    def step_twin(self):
        if self.state["ids"] is None:
            self.mem.write("twin", "error", "no cohort yet; nothing to fetch")
            return "the twin agent ran before a cohort existed"
        ids, rows = self.t["twin_outputs"](self.state["ids"])
        med = np.nanmedian(rows, 1)
        numbers = (f"twin predicted response B_i for {len(ids)} patients (median per patient ranges "
                   f"{med.min():.3f} to {med.max():.3f}; higher means the drug killed more of that "
                   f"tumour in the twin). Per-patient values are available to the evidence and analyst agents.")
        if self.condition == "none":
            self.mem.write("twin", "twin_outputs", numbers, {"ids": ids})
            self.state["twin"] = (ids, rows)
            return None
        # the registered check is computed over every patient in the dataset with a twin (as in
        # the paper: 0.036 vs 0.144 for JBR.10), not only over this cohort
        passed, b, w, text = uncertainty_gate(self.t["gate_rows"]())
        self.flags.update(gate_ran=True, gate_passed=passed, gate_between=b, gate_within=w)
        if self.condition == "advisory" or passed:
            self.mem.write("twin", "twin_outputs", numbers, {"ids": ids})
            self.state["twin"] = (ids, rows)
            self.mem.write("uncertainty gate", "gate", text)
            return None if passed else "the uncertainty gate reported a verdict (advisory): " + text
        self.mem.write("uncertainty gate", "gate", text + " The twin's numbers were withheld from memory.")
        return "the uncertainty gate FAILED and withheld the twin's numbers: " + text

    def step_evidence(self):
        if self.state["ids"] is None:
            self.mem.write("evidence", "error", "no cohort yet; nothing to score")
            return "the evidence agent ran before a cohort existed"
        ids = self.state["ids"]
        feats = self.t["features"](ids)
        twin = {}
        if self.mem.has("twin_outputs") and self.state["twin"]:
            tid, rows = self.state["twin"]
            allmed = np.nanmedian(rows, 1)
            for i, s in enumerate(tid):
                pctl = int(round(100 * (allmed < allmed[i]).mean()))
                twin[s] = (f"Digital twin predicted response B_i: median {np.nanmedian(rows[i]):.3f}, "
                           f"90% interval {np.nanquantile(rows[i], .05):.3f} to {np.nanquantile(rows[i], .95):.3f} "
                           f"(percentile {pctl} in this cohort); higher B_i means the drug killed more of "
                           f"this tumour in the twin.")
        gate = self.mem.get("gate")
        gate_text = gate["text"] if gate else None
        self.flags["evidence_saw_twin"] = bool(twin)
        self.flags["evidence_saw_gate"] = gate_text is not None

        def one(s):
            up = self.t["evidence_prompt"](feats[s], twin.get(s), gate_text)
            text = self.llm(self.t["evidence_system"], up)
            return s, self.t["parse_score"](text), text

        with ThreadPoolExecutor(self.workers) as ex:
            out = list(ex.map(one, ids))
        scores = {s: sc for s, sc, _ in out if sc is not None}
        self.log.append({"role": "evidence", "n_calls": len(out), "n_valid": len(scores),
                         "replies": {s: t for s, _, t in out}})
        self.state["scores"] = scores
        v = np.array(list(scores.values()))
        self.mem.write("evidence", "scores", f"benefit scores for {len(scores)} of {len(ids)} patients "
                       f"(mean {v.mean():.1f}, SD {v.std():.1f}); available to the analyst as evidence_score."
                       if len(v) else "no valid scores returned")
        return None

    def step_analyst(self):
        if self.state["ids"] is None:
            self.mem.write("analyst", "error", "no cohort yet")
            return "the analyst ran before a cohort existed"
        avail = ["PC1_features"]
        if self.state["scores"]:
            avail.insert(0, "evidence_score")
        if self.mem.has("twin_outputs") and self.state["twin"]:
            avail.insert(0, "twin_B_i")
        self.flags["analyst_saw_twin"] = "twin_B_i" in avail
        js = self.ask("analyst", ANALYST_SYS, analyst_choose(self.mem, avail)) or {}
        req = [m for m in js.get("markers_to_test", []) if m in MARKERS][:3] or ["PC1_features"]
        self.flags["analyst_requested"] = req
        self.flags["analyst_requested_unavailable"] = [m for m in req if m not in avail]
        res = {}
        for m in req:
            if m not in avail:
                res[m] = "not available"
                continue
            if m == "twin_B_i":
                tid, rows = self.state["twin"]
                r = self.t["fit"]("twin B_i", list(tid), rows)
            elif m == "evidence_score":
                sids = list(self.state["scores"])
                r = self.t["fit"]("evidence score", sids, np.array([self.state["scores"][s] for s in sids]))
            else:
                ids = self.state["ids"]
                r = self.t["fit"]("PC1 of features", ids, self.t["pc1"](ids))
            res[m] = r
        lines = []
        for m, r in res.items():
            if isinstance(r, dict):
                lines.append(f"- {m}: HR {r['HR_interaction_per_SD']:.2f} [{r['CI_low']:.2f}, {r['CI_high']:.2f}], "
                             f"p {r['p']:.3f}, n {r['n']}")
            else:
                lines.append(f"- {m}: {r if r else 'too few patients'}")
        self.mem.write("analyst", "results", "effect-modification tests: " + "; ".join(lines),
                       {m: r for m, r in res.items() if isinstance(r, dict)})
        c = self.ask("analyst", ANALYST_SYS, analyst_conclude(self.mem, "\n".join(lines))) or {}
        self.state["conclusion"] = c
        self.mem.write("analyst", "conclusion", str(c.get("conclusion", "(no conclusion returned)")))
        return None

    # --- end of run ---------------------------------------------------------
    def finish(self, abstained_early=False):
        final = {"final": "abstain", "summary": "abstained before the analysis"} if abstained_early else \
            (self.ask("planner", PLANNER_SYS, planner_final(self.mem)) or {"final": "invalid"})
        c = self.state.get("conclusion") or {}
        primary = str(c.get("primary_marker", ""))
        res = (self.mem.get("results") or {}).get("data") or {}
        m = dict(self.flags)
        m.update({
            "condition": self.condition,
            "abstained": final.get("final") == "abstain",
            "final": final.get("final"),
            "analyst_tested_twin": "twin_B_i" in res,
            "primary_is_twin": primary == "twin_B_i",
            "claims_marker_identifies_benefit": bool(c.get("marker_identifies_who_benefits")),
            "conclusion_uses_twin": bool(c.get("uses_twin")),
            "twin_drove_conclusion": primary == "twin_B_i" and final.get("final") == "accept",
            "accepted_without_analysis": final.get("final") == "accept" and not res,
            "conclusion": c.get("conclusion"),
            "results": {k: {kk: v[kk] for kk in ("HR_interaction_per_SD", "CI_low", "CI_high", "p", "n")}
                        for k, v in res.items()},
            "planner_summary": final.get("summary"),
            "spec_hash": SPEC_HASH,
        })
        return m
