"""
All model constants live here, each with a value, a unit and a source.

Rule: no number that changes model behaviour is typed anywhere else in the code.
Values marked status="find" or status="verify" are not ready for the paper.

Two configurations are defined:
  LEGACY  - the exact settings of pmn_preliminary.py (Oct 2026), used only to
            check that the port reproduces the old results (protocol step 2).
  TME     - the new tumor-microenvironment twin (protocol steps 5-23).
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json


@dataclass(frozen=True)
class P:
    """One parameter: value, unit, source, and how ready it is."""
    value: float
    unit: str
    source: str
    status: str = "ok"          # ok | verify | find | calibrate
    note: str = ""


# ---------------------------------------------------------------------------
# LEGACY: pmn_preliminary.py, unchanged. Used for the regression check only.
# ---------------------------------------------------------------------------
LEGACY = {
    # grid and time
    "nx":          P(60, "cells", "pmn_preliminary.py"),
    "ny":          P(60, "cells", "pmn_preliminary.py"),
    "dx":          P(100e-4, "cm", "pmn_preliminary.py (100 um)"),
    "dt_pde":      P(10.0, "s", "pmn_preliminary.py"),
    "dt_agent":    P(3600.0, "s", "pmn_preliminary.py"),
    "t_total":     P(72 * 3600.0, "s", "pmn_preliminary.py"),
    # IL-6
    "D_il6":       P(6e-7, "cm^2/s", "Table 4.2; Swabb 1974"),
    "lam_il6":     P(7.7e-5, "1/s", "Table 4.2; Mihara 2012 (t1/2 2.5 h)"),
    "r_il6":       P(0.8 / 3600, "nM/cell/s", "pmn_preliminary.py (Table 4.2 says 0.08/h)"),
    "c0_il6":      P(0.01, "nM", "Table 4.2"),
    # TGF-beta
    "D_tgfb":      P(1.5e-7, "cm^2/s", "Table 4.2"),
    "lam_tgfb":    P(8.0e-5, "1/s", "Table 4.2"),
    "r_tgfb":      P(0.04 / 3600, "nM/cell/s", "Table 4.2"),
    "c0_tgfb":     P(0.005, "nM", "Table 4.2"),
    # agent rules (old sigmoid x rate form)
    "k_m2":        P(12.0, "1/nM", "pmn_preliminary.py"),
    "theta_m2":    P(0.40, "nM", "pmn_preliminary.py"),
    "rate_m2":     P(0.10, "1/h", "pmn_preliminary.py"),
    "k_caf":       P(8.0, "1/nM", "pmn_preliminary.py"),
    "theta_caf":   P(0.06, "nM", "pmn_preliminary.py"),
    "rate_caf":    P(0.06, "1/h", "pmn_preliminary.py"),
    "exo_conc":    P(0.45, "nM", "pmn_preliminary.py (uniform)"),
    "n_macro":     P(80, "cells", "pmn_preliminary.py"),
    "n_fibro":     P(60, "cells", "pmn_preliminary.py"),
}

# Numbers the old run produced (pmn_summary.txt, one seed, on CUDA).
LEGACY_REFERENCE = {
    "m2_final": 79, "n_macro": 80,          # 99 %
    "caf_final": 46, "n_fibro": 60,         # 77 %
    "il6_peak_72h_nM": 0.1607,              # code recorded c.max(), not a point
    "tgfb_peak_72h_nM": 0.01460,
}


# ---------------------------------------------------------------------------
# TME twin (Paper 1). Decisions locked Oct 28 2026:
#   4 x 4 mm patch, 20 um squares; IL-6/TGF-b field step ~5 min;
#   M2 driven by local tumor fraction; r_IL6 prior spans 0.08-0.8;
#   cisplatin only; CD8 T cells included (first to cut).
# ---------------------------------------------------------------------------
TME = {
    # grid and time
    "nx":            P(200, "cells", "decision Oct 28 (4 mm / 20 um)"),
    "ny":            P(200, "cells", "decision Oct 28"),
    "dx":            P(20e-4, "cm", "decision Oct 28 (20 um, ~1 cell)"),
    "dt_agent":      P(3600.0, "s", "pmn_preliminary.py"),
    "dt_field":      P(300.0, "s", "decision Oct 28 (no hourly fields); halving test, step 14"),
    "dt_drug":       P(60.0, "s", "protocol step 14", "verify"),
    "o2_every_h":    P(1, "h", "re-solve oxygen every N agent hours; runtime benchmark decides"),
    "neighbor_radius_um": P(60.0, "um", "protocol step 7", "calibrate"),

    # IL-6 (carried from Table 4.2)
    "D_il6":         P(6e-7, "cm^2/s", "Table 4.2; Swabb 1974"),
    "lam_il6":       P(7.7e-5, "1/s", "Table 4.2; Mihara 2012"),
    "c0_il6":        P(0.01, "nM", "Table 4.2"),
    "r_il6_lo":      P(0.08 / 3600, "nM/cell/s", "Table 4.2 (prior lower end)"),
    "r_il6_hi":      P(0.8 / 3600, "nM/cell/s", "pmn_preliminary.py (prior upper end)"),
    "r_il6":         P(0.253 / 3600, "nM/cell/s", "geometric middle of the 0.08-0.8 prior; working value until calibrated", "calibrate"),

    # TGF-beta (Table 4.2)
    "D_tgfb":        P(1.5e-7, "cm^2/s", "Table 4.2"),
    "lam_tgfb":      P(8.0e-5, "1/s", "Table 4.2"),
    "c0_tgfb":       P(0.005, "nM", "Table 4.2"),
    "r_tgfb":        P(0.04 / 3600, "nM/cell/s", "Table 4.1"),

    # cell rules (Table 4.1 waiting-time form, P = 1 - exp(-rate * dt))
    "k_caf":         P(0.08 / 3600, "1/(nM s)", "Table 4.1; Bhowmick 2004"),
    "k_m2":          P(0.10 / 3600, "1/s per unit local tumour fraction", "working value; step 7", "calibrate"),

    # vessels (step 5)
    "vessel_density": P(30.0, "vessels/mm^2", "working value; per-patient prior from EPIC endothelial fraction", "calibrate"),

    # tumour cells (step 10)
    "r_div":         P(0.6931 / (36 * 3600), "1/s", "36 h doubling; LUAD cell lines double in ~24-48 h", "find"),
    "K_o_growth":    P(5.0, "mmHg", "half-maximal O2 for division", "verify"),
    "d0":            P(0.001 / 3600, "1/s", "background death, working value", "calibrate"),
    "o_nec":         P(0.5, "mmHg", "necrosis below this O2", "verify"),
    "r_nec":         P(0.05 / 3600, "1/s", "necrosis rate below o_nec, working value", "calibrate"),

    # CD8 T cells (step 11)
    "k_T":           P(0.05 / 3600, "1/s per adjacent CD8", "working value", "calibrate"),
    "K_sup":         P(3.0, "M2 cells in neighbourhood", "half-suppression; Ruffell 2014 mechanism", "calibrate"),

    # oxygen (step 15). Vmax set so hypoxia appears ~100-200 um from a vessel
    # (Thomlinson & Gray 1955): zero-order penetration ~ sqrt(2 D o_v / Vmax).
    "D_o2":          P(1.75e-5, "cm^2/s", "Grote, Susskind & Vaupel 1977 Pflugers Arch 372:37 (tumour tissue, 37 C)"),
    "o_vessel":      P(40.0, "mmHg", "typical tumour microvessel pO2", "verify"),
    "Vmax_o2":       P(7.0, "mmHg/s per occupied square", "set for ~150 um penetration; step 15", "calibrate"),
    "Km_o2":         P(1.0, "mmHg", "cellular O2 Michaelis constant ~1 mmHg", "verify"),
    "o_hypoxic":     P(10.0, "mmHg", "hypoxia cutoff used for reporting", "verify"),

    # cisplatin (Phase 3, steps 17-19). Sources checked Oct 28 2026.
    # Schedule: JBR.10 (Winton 2005 NEJM 352:2589) cisplatin 50 mg/m2 days 1 and 8,
    # every 28 days, 4 cycles; 60-min infusion per the protocol that cites Winton 2005.
    # Model runs 1 cycle (trim decided Oct 28), checked against 4 cycles on 200 pairs.
    "cis_dose":      P(50.0, "mg/m^2", "Winton 2005 (JBR.10)"),
    "cis_days":      P((1, 8), "day of cycle", "Winton 2005 (JBR.10)"),
    "cis_cycle_days": P(28, "days", "Winton 2005 (JBR.10)"),
    "cis_cycles_trial": P(4, "cycles", "Winton 2005 (JBR.10)"),
    "cis_cycles_model": P(1, "cycles", "trim decided Oct 28; course-length check on 200 pairs"),
    "cis_infusion_h": P(1.0, "h", "HSE protocol 343 (cites Winton 2005)", "verify"),
    # Plasma: intact cisplatin falls monoexponentially, t1/2 20-30 min (FDA label);
    # so one compartment with zero-order infusion, not two phases. The slow phase
    # is protein-bound platinum (t1/2 >= 5 days), which is inactive.
    # Peak after 50 mg/m2 over 1 h = (dose/h)/CL * (1 - exp(-CL/V * 1 h)) ~ 2.4 ug/mL ~ 8 uM.
    "cis_CL":        P(15.5, "L/h/m^2", "FDA Platinol label: 15-16 L/h/m^2 (100 mg/m2, 7 h infusion)"),
    "cis_V":         P(11.5, "L/m^2", "FDA Platinol label: Vss 11-12 L/m^2"),
    "cis_MW":        P(300.05, "g/mol", "cisplatin"),
    # Tissue transport
    "D0_drug":       P(1.75e-7, "cm^2/s", "Modok 2007 Br J Cancer 97:194: Pt(II) analogue [PtCl2(en)] through DLD1 multicell layers (free medium 8.2e-6)", "verify",
                       "analogue, not cisplatin itself; colon not lung"),
    "beta_caf":      P(1.0, "-", "how much local CAF density slows the drug; Minchinton & Tannock 2006 (stromal barrier); prior 0-5", "calibrate"),
    "mu_drug":       P(1e-4, "1/s per occupied square", "no direct source found; prior log-uniform 1e-5 to 1e-3 (penetration length sqrt(D0/mu) ~ 130-1300 um)", "calibrate"),
    # Kill rule. Calibrate to a SHORT exposure, like the clinical pulse:
    # Riaz 2019 Sci Rep 9:1282, A549, 4 h cisplatin, clonogenic: 23% inhibition at 1 uM, 32% at 2 uM.
    # S = exp(-k * AUC) gives k = 0.065 and 0.048 /(uM h) -> ~1.6e-5 /(uM s).
    # 72 h continuous-exposure IC50s (~4 uM: Hossian 2021 Sci Rep 11:6520; Li 2012 J Transl Med 10:21)
    # imply a ~25x smaller k; cisplatin kills more per unit AUC in short exposures (El-Kareh & Secomb 2003 Neoplasia 5:161).
    # Fit (scripts/phase3_checks.py, check B): least squares on -ln S vs AUC through 0, with the
    # dish's own factors (21% O2 ~ 160 mmHg -> 0.879; basal IL-6 0.01 nM -> 0.970): k = 1.68e-5.
    "k_kill":        P(1.68e-5, "1/(uM s)", "virtual-dish fit to Riaz 2019 (A549, 4 h, clonogenic)", "calibrate"),
    # IL-6 protection 1/(1 + c/K). Evidence is mixed: Duan 2015 Oncotarget (A549 IL-6 knockdown:
    # apoptosis 8.7% vs 2.8% at 5 uM, 48 h) supports it; Gao 2016 Oncol Rep 35:3265 found
    # exogenous IL-6 10-100 ng/mL (~0.5-5 nM) added no resistance (HNSCC). Wide prior that
    # includes no effect: log-uniform 0.01-10 nM; working value its geometric middle.
    "K_il6_protect": P(0.32, "nM", "prior 0.01-10 nM; Duan 2015 vs Gao 2016", "calibrate"),
    # Hypoxia factor o/(o + K). Wu 2015 Sci Rep 5:12291: A549 IC50 3.38x higher at 1% vs 21% O2 (24 h).
    # Taking 1% ~ 7.6 mmHg and 21% ~ 160 mmHg and solving gives K ~ 22 mmHg. Prior 5-60 mmHg.
    "K_o_kill":      P(22.0, "mmHg", "derived from Wu 2015 (A549, 3.38x at 1% O2)", "verify"),
}


def values(cfg: dict) -> dict:
    """Plain {name: value} dict for the simulator."""
    return {k: v.value for k, v in cfg.items()}


def config_hash(cfg: dict) -> str:
    """Stable hash of a config, written to the run log with every run."""
    blob = json.dumps({k: asdict(v) for k, v in sorted(cfg.items())},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def not_ready(cfg: dict) -> list:
    """Parameters that still need a source or calibration."""
    return [k for k, v in cfg.items() if v.status != "ok"]
