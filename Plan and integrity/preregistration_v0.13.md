# Pre-registration: TME twin as a mechanistic guardrail for agentic RWE

v0.13 — Nov 10, 2026. Status: **draft for LC to freeze**, after the absolute-response
check (Section 6, A2) has been run. Supersedes v0.12, v0.11, v0.10, v0.9, v0.8, v0.7, v0.6, v0.5, v0.4, v0.3, v0.2 and v0.1, all kept
unchanged in `docs/`.

Freezing: copy this file unchanged to `docs/preregistration_frozen.md`, record its
SHA-256 in `runs/runlog.jsonl`, and pass it to `scripts/phase8_outcomes_v1.py --freeze`.
After that the file is never edited; amendments go in Section 10 with a date and reason.

Everything in this version was decided without any treatment-by-survival or
twin-by-survival data having been examined (Section 4).

---

## 1. Questions

1. Can a mechanistic tumour-microenvironment twin, built from literature biology, be
   checked before use and fitted to real patients from one bulk-RNA sample? (A1)
2. Does a single fitted twin respond to cisplatin at a realistic magnitude? (A2)
3. What can bulk RNA tell the twin about a patient, and what drives the twin's
   chemotherapy response? (A3)
4. Which additional measurement would let the twin tell patients' responses apart? (A4)
5. Does twin-predicted response B_i modify the effect of adjuvant chemotherapy?
   (Exp 4, subject to the precondition in Section 6; Exp 5 supporting)
6. Does an AI agent predict chemotherapy benefit from the same data, with and without
   the twin's outputs? (Exp 7)

## 2. The twin and its output

Model as run: `tissue_v3` (fibroblast sensing 120 µm), `config_v3`, `priors_v4`,
observation `observe_v2`, plan `epic+gsva_m2`. Campaign: 20,000 draws,
`runs/phase5_v3_fib120_sims` (seeds 7000, shards a, b, c [stopped at 512 draws], c1,
c2, d; combined Nov 7).

- **B_i** = log10( max(N_no chemo, 1) / max(N_chemo, 1) ) at hour 192 (day 9 of cycle 1,
  24 h after the day-8 dose), paired arms with the same random numbers.
  *Change from v0.4:* both arms floored at one cell. A tumour that disappears without
  chemo gave log10(0); with both floored it gives B_i = 0 (nothing left to remove).
  Occurred in 29 of 1,330 JBR.10 tissues. Decided Nov 8 before any outcome was seen.
- **Shared worlds.** For the B_i used against outcomes, draw d of every patient uses the
  same values of the 24 shared and latent parameters (one "world" per draw).
  *Change from v0.4:* v0.4 already stated latent parameters are shared across patients
  within a draw; the first implementation did not, and is corrected
  (`phase6_patient_chemo_v3 --shared-world`). Reason found from the outcome-free
  sensitivity ranking (A3).
- **Source of B_i: direct simulation**, not the emulator. *Change from v0.4:* the
  emulator's held-out R² was 0.75 against the pre-registered 0.95 (Nov 8). The
  value-of-information analysis (A4) shows the ceiling for any predictor is R² 0.76,
  because about a quarter of B_i variance is run-to-run chance (cell placement). The 0.95
  bar was not attainable for this model; the emulator is not used.
- JBR.10 (GSE14814): 133 patients × 10 shared-world draws, one cycle
  (`runs/phase6_bi_direct_v2_sharedworld_floorboth/bi_GSE14814.npz`).
- Standardisation: B_i enters survival models per SD of posterior medians within the
  cohort, computed before outcome merge.

## 3. Observation choices (made from the outcome-free coverage check, Nov 6)

- f_endothelial: linked within cohort (primary); direct comparison as sensitivity
  (`runs/phase5_v3_fib120_npe_logithyp_endodirect`).
- Hypoxia through a logit link; the other linked features through a log link.
- Coverage on all 20,000 draws: joint test passes in all four cohorts (1.5%, 6.7%,
  1.7%, 1.9% beyond the simulations' 95th percentile; limit 10%). GSE42127 fibroblasts
  13–16% below the twin's range: reported limitation.

## 4. Outcome firewall

Unchanged from v0.4. Survival and treatment are read only by
`scripts/phase8_outcomes_v1.py`, which refuses to run without this frozen file.
**Declaration (LC):** as of the freeze, no survival outcome has been examined by
treatment arm or against any twin output or AI-agent score, in any cohort.

## 5. Data and cohorts

As v0.4. GSE14814 (JBR.10): adenocarcinoma among the 133 arrayed patients (71 per Zhu
2010) for Exp 4; all 133 are fitted and simulated, the analysis restricts to
adenocarcinoma. TCGA-LUAD, GSE68465, GSE42127 for Exp 5. Outcome: overall survival.

## 6. Analyses

**A1. Checks before fitting (done; reported as results).** Steady state (40–60 days, all
populations within sampling noise); direction checks (vessel density z = +102,
Vmax_o2 z = −57, macrophage recruitment z = −4.0; CAF effect explained by crowding);
coverage (Section 3).

**A1b. Calibration of the fitting (SBC, added v0.6).** 300 simulated tissues held out,
the estimator retrained on the rest with the GSE14814 link and noise, 200 posterior draws
per held-out tissue; per parameter the rank of the truth (Talts 2018), chi-square test of
uniformity, 50% and 90% interval coverage (`phase5_sbc_v1`). Reported as is; a parameter
that fails is named and its between/within ratio read with that caveat.

**A1c. Cell-cell dependencies (added v0.6).** Spearman correlations between the 12
features in the campaign vs in each cohort (rank-based, so unaffected by the monotone
links). Reported: share of clearly correlated patient pairs (|rho| > 0.2, p < 0.01) whose
sign the twin matches, matrix correlation, and pairs the twin gets wrong in ≥ 2 cohorts
(`phase5_dependency_check_v1`). Outcome-free.

**A1d. The dependency equation (added v0.7).** For every non-tumour feature,
y = a + b·log(1 − f_tumour) + c·I + e, where I is one shared axis per patient (first
principal component after purity). Reported per cohort and for the twin: variance explained
by purity and by purity plus the axis, the b and c weights, and how alike c is across
cohorts (`phase5_dependency_equation_v1`). Outcome-free.

**A1e. Dependency gate (added v0.12).** A fixed pass/fail rule
(`phase5_dependency_gate_v1`): a cohort passes if the twin matches the sign of >= 75% of the
clearly correlated patient feature pairs (|rho| > 0.2, p < 0.01) AND the purity slopes of
the A1d equation agree in direction (cosine of the b vectors >= 0.70); the twin passes if
>= 3 of 4 cohorts pass. Positive control: 20 synthetic cohorts drawn from the twin itself,
with the fitting's observation noise, should pass. The gate establishes consistency with
two empirical constraints only; it is not evidence of biological correctness or of
identifiability. *Disclosure:* the thresholds were set on Nov 10 after the A1c/A1d results
had been seen (sign agreement 0.45–0.53); they are reported as such, and the positive
control, not the twin's result, is what tests the gate. TumorTwin is not scored (it does not
generate features).

**A5. Positive control for the identifiability guardrail (added v0.12).** Question: does
the Exp 4 precondition (between-patient SD of B_i >= mean within-patient SD) pass when
patients truly differ and the pipeline can see it, and fail when they do not?
(`phase6_guardrail_positive_control_v1`). Synthetic patients are held-out campaign tissues
with known parameters, observed with the fitting's link and noise; a posterior estimator is
trained on the remaining tissues. B_i is computed by a surrogate (gradient boosting from the
36 parameters to B_i on the 4,504 campaign tissues with a simulated B_i, plus run-to-run chance
from its held-out residuals), not by new simulations; the surrogate is first applied to the
real JBR.10 posterior draws and is reported beside the direct result (0.036 vs 0.144).
Four scenarios, 20 synthetic cohorts of 133 patients, 10 draws each: S1 measured (patient
parameters known to ±10%, shared parameters known), S2 RNA posterior with shared parameters
known, S3 RNA posterior with shared parameters drawn per world as in the study, S4 null (all
patients identical; features differ only by noise). Two forms of the check: primary, as
pre-registered; secondary, a reliability coefficient R = 1 − (π/2)·mean(within²)/draws/between²
(share of the between-patient spread of medians not explained by the noise of a median),
pass if R >= 0.5. *Criteria, fixed before running:* the check discriminates if S1 passes in
>= 90% of cohorts and S4 fails in >= 95%; it is informative about ranking if, over S1–S3
cohorts, "pass" agrees with "Spearman rho between estimated and true responses >= 0.5" in
>= 80%. Reported for both forms whatever the result. The Exp 4 verdict on real patients
stays the primary form's (it fails); the secondary form is reported beside it and would only
be proposed for future use. Described as a surrogate-based demonstration of how the check
behaves, not as new simulation evidence.

*Results of A1b–A1d and A2, computed Nov 8 before freezing, are reported as computed:*
SBC calibrated for 10 of 12 parameters (r_il6, Vmax_o2 slightly high; 90% coverage
0.87, 0.92). Dependency check: the twin matches the sign of about half of clear patient
correlations (25–28 of 53–56 pairs). Equation: purity explains 28–37% in patients and 10%
in the twin; purity plus the axis 47–54% vs 35%; the axis agrees across the three
microarray cohorts (cosine 0.84–0.93), not TCGA (0.23–0.41). A2: twin MPR 5.5%, pCR 4.5%
(CheckMate 816 chemo alone 12.7%, 2.2%).

**A2. Absolute response.** Each JBR.10 twin (133 patients × 3 draws) receives three
cycles of cisplatin on the JBR.10 schedule (2,016 h). Reported: median viable tumour
fraction after each cycle; tumour left relative to the same twin untreated; the share of
patients whose twin reaches ≤ 10% viable tumour (twin MPR) and no tumour (twin pCR).
Reference: chemotherapy-alone arm of CheckMate 816 (Forde 2022), MPR 12.7%, pCR 2.2%.
Reported as an order-of-magnitude comparison (single agent vs platinum doublet;
different schedule). No threshold for success; nothing is tuned to it.

**A3. What RNA tells the twin; what drives B_i (done).** Patient specificity per
parameter (between/within ratio of posterior medians); first-order Sobol indices of B_i
over the 20,000-draw campaign (`phase6_sensitivity_v1`); ranking stability across shared
worlds (mean correlation of patient B_i between worlds).

**A4. Value of information (done).** Held-out R² for B_i from the 12 RNA features alone
and with one added noisy measurement each (vessel density, Ki-67, hypoxia PET, tumour
burden, ex vivo cisplatin sensitivity), against the all-parameters ceiling
(`phase7_value_of_information_v1`).

**A3b/A4b. Prior sensitivity (added v0.6).** A3 and A4 repeated on the campaign subset
whose k_kill lies in the central 25% of its log prior range around the working value
(1.68e-5) and K_o_kill in the central 50% around 22 mmHg (`phase6_subset_sims_v1`). The
question: do the patient parameters gain weight once the shared drug parameters are
pinned down? Reported side by side with A3/A4.

**Exp 4 (JBR.10).** Cox: hazard ~ treatment + B_i + treatment × B_i + stage + age,
B_i per SD, fitted once per shared world (10) and pooled by Rubin's rules.

*Precondition (from v0.4, unchanged):* Exp 4 runs as a confirmatory test only if the SD
of posterior medians across patients is at least the mean within-patient SD.
**Result, computed before outcome merge (Nov 8; corrected Nov 9): 0.036 vs 0.144. The precondition
fails.** *Correction, v0.9:* v0.5–v0.8 gave 0.047, which was the SD of each patient's
*mean* over the 10 worlds (`phase6_bi_from_runs_v2`); the precondition is defined on
posterior *medians*, which gives 0.036. Found Nov 9 when the guardrail arm computed the
check from the B_i file; no outcome had been examined. Both values fail. Exp 4 is therefore reported as **not performed as a confirmatory test**, with
the reason. This is the guardrail outcome the design anticipated.

*Exploratory:* the same Cox model is still fitted and reported, labelled exploratory,
with its interval, and no conclusion is drawn from it about effect modification.
Comparators in the same model, also exploratory: TumorTwin B_TT
(`phase7_tumortwin_v1`, same posterior draws), the AI-agent scores (Exp 7), and PC1 of
the 12 features (oriented to correlate positively with B_i). The 15-gene signature of
Zhu 2010 is quoted, not refitted. Power statement as v0.4 (only HR ≲ 0.4 per SD
detectable with 71 patients).

**Exp 5 (supporting, exploratory).** TCGA-LUAD, GSE68465, GSE42127: the same interaction
model for TumorTwin, PC1 and (if computed) twin B_i, with stabilised IPTW on stage, age
and sex, a 180-day landmark, robust SEs; fixed-effect pooled estimate across cohorts.
Twin B_i for these cohorts only if direct simulation is completed before the outcome
merge; otherwise reported as not computed.

**Exp 7. AI agent with and without the twin, and with the twin's guardrail (JBR.10).**
Model `claude-opus-5-5` (Anthropic Messages API, a separate API key, no conversation
history or project context; temperature 0, at most 300 reply tokens; date and the model
id the API reports are recorded in every row). Script `phase7_llm_predict_v3.py`.
Fable 5.1 was considered and not used as primary because of reported refusal and
rerouting on biomedical prompts; it may be run on the 3-patient pilot only, reported
descriptively.

Three arms per patient:
(a) *features*: the 12 tumour features, each with its percentile in the cohort;
(b) *features+twin*: (a) plus the twin's B_i (median, 90% interval over the 10 shared
worlds, cohort percentile) and the 12 fitted parameters (median, 90% interval, readable
units);
(c) *features+twin+guardrail*: (b) plus one sentence stating the twin's pre-registered
identifiability check (between-patient SD of B_i vs mean within-patient SD, computed from
the B_i file by the script) and its verdict.
Prompts are fixed in the script (SYSTEM, TWIN_NOTE, guardrail_note); no identifiers, study
names, stage, age, sex, treatment or outcome in any prompt; patients in a shuffled order.
3 repeats per patient, averaged; at temperature 0 the repeats are expected to be near
identical and are reported as a reproducibility check, not as uncertainty.
*Deviation, recorded Nov 10 after the run and before any outcome was examined:* the API
rejects `temperature` for `claude-opus-5-5` ("temperature is deprecated for this model");
the script's fallback then sent every call without it, so all 1,197 calls used the model's
default sampling, not temperature 0. The model also reasoned internally by default
(thinking tokens per call, median 176, 109 and 159 in the three arms) inside the
300-token cap; 5 of 1,197 calls (4 features, 1 features+twin) spent the whole cap on
reasoning and returned no score, and are counted as invalid as planned (each patient keeps
at least two valid repeats). Consequence: the 3 repeats measure the agent's run-to-run
variation (mean within-patient SD 1.5–2.0 points of 100) rather than a near-deterministic
reproducibility check. No call was rerun; the arm comparisons are unchanged.

*Outcome-free comparisons between arms (pre-specified):* the SD of scores across
patients per arm; the Spearman correlation of scores with the twin's B_i percentile in
arms (b) and (c); the share of replies whose reason refers to the twin's uncertainty or
inability to rank (counted by keyword list fixed in the analysis script before reading
replies: "uncertain", "cannot", "unreliable", "not reliable", "rank", "identif",
"limited"); refusals and replies without a valid score. Expectation, stated before the
run: if the guardrail works as a guardrail, arm (c) has a narrower spread and a weaker
link to B_i than arm (b).
*Against outcomes:* each arm's score enters the Exp 4 Cox model as an exploratory
comparator.

*Exp 7b, added v0.11 (Nov 10, before any outcome was examined; budget raised to $300 by
LC's advisor). Robustness of the agent result, kept small on purpose: the paper is about
the twin and its guardrail, not about language models or prompt wording.*
- **Enforced guardrail arm** (`features+twin+enforced`): the twin is described, its check
  and verdict are stated, and its outputs (B_i and parameters) are withheld. Script
  `phase7_llm_predict_v4.py`; prompt text fixed in `enforced_note`.
- Run A: `claude-opus-5-5`, 300 tokens, enforced arm only (pairs with the primary three arms).
- Run B: `claude-opus-5-5`, 1,000 tokens, all four arms (does a larger reasoning budget
  change reliance on the twin; removes cap failures).
- Run C: `claude-sonnet-5-5`, 1,000 tokens, all four arms (a second model).
- All: 133 patients x 3 repeats, default sampling (temperature not accepted), cap $40 per
  arm file; stop reason and thinking tokens recorded per call.
- Analysis (`phase7_llm_arms_compare_v2.py`), per run, same measures as Exp 7, plus the
  Spearman correlation of the *features-only* arm with the B_i percentile as the baseline:
  B_i is itself a function of the features, so an agent can correlate with B_i without
  seeing it. Reliance on the twin's number is read as an arm's correlation minus the
  features-only baseline, not as the raw correlation.
- Expectation, stated before the run: the enforced arm's correlation with B_i is close to
  the features-only baseline, and its spread is no wider than the advisory arm's.
- Reported as robustness; the primary Exp 7 result remains Opus, 300 tokens, three arms.
- Not done (LC): prompt rewording. Fable 5.1 not used.

*Exp 7b results (Opus 300/1,000 tokens, Sonnet 1,000 tokens), computed Nov 10 before any
outcome was examined, are reported as computed:* reliance above the features-only baseline,
Opus: twin +0.57, advisory guardrail +0.19, enforced +0.01 (1,000 tokens: +0.56, +0.18, +0.01);
Sonnet: twin +0.71, advisory +0.64, enforced −0.06. At 300 tokens most enforced-arm replies
stopped at the token cap after the score, so reasons (keyword shares) at 300 tokens
undercount; reasons are read from the 1,000-token runs.

*Exp 7c, added v0.13 (Nov 10, before any outcome was examined; LC):* the same four arms with
an open-weight, medicine-trained model, `google/medgemma-27b-text-it`, served on the UNT GPU
server with vLLM (`scripts/serve_medgemma_v1.sh`; quantisation fp8 or 4-bit chosen by free
GPU memory and recorded with the vLLM version and the model's generation settings in
`runs/phase7_medgemma_serving_<quant>.json`), via `phase7_llm_predict_v5.py --provider local`,
1,000 tokens, 133 patients x 3 repeats, model default sampling (no temperature sent, as for
Claude). Nothing leaves the server. Same analysis (`phase7_llm_arms_compare_v2`), same
expectation as Exp 7b. Reported as robustness (open, domain-specific model). *Stopping rule:*
if the model is not serving by Nov 11, 14:00 CT, Exp 7c is not run and is reported as
planned but not done.

Budget: $100, may be raised to $150 (any raise recorded with the date). Cap
`--max-dollars 33` per arm (each arm keeps its own cost record). A 3-patient pilot per
arm first.

*Disclosure:* an instance of the same model family assisted in writing the analysis code;
it did not see outcomes, and every reported number comes from code run on the UNT server.
The agent arms test one model with one fixed prompt, not AI agents in general.

*Why the prompt is built this way (each choice with its basis):*
- No identifiers or study names: language models can memorise public tabular datasets
  and their labels (Bordt et al. 2024), so nothing that could identify JBR.10 or a patient
  is given.
- Features serialised as labelled text lines, one per feature: the standard way to give
  tabular rows to language models (Hegselmann et al., TabLLM, AISTATS 2023).
- Cohort percentile beside each raw value: the raw scales (EPIC fractions, ssGSEA scores)
  have no meaning outside their cohort; a percentile gives the reference a clinician
  would use. Design choice, stated as such.
- "Score relative benefit from chemotherapy, not prognosis": the predictive vs prognostic
  distinction (Ballman, JCO 2015); the outcome test is a treatment-by-score interaction.
- One fixed prompt and format, recorded verbatim: model outputs change with small
  formatting changes (Sclar et al., ICLR 2024), so the prompt is fixed before the run and
  reported in full (TRIPOD-LLM, Gallifant et al., Nat Med 2025).
- Temperature 0 and repeats: lowest-variance setting, but not fully deterministic
  (Atil et al. 2024), hence 3 repeats.
- 0-100 score in JSON with a one-sentence reason: a verbalised numeric score, the usual
  way to get a graded judgement from a model (Tian et al., EMNLP 2023); the reason is kept
  for the keyword analysis above. Scale and wording are design choices, stated as such.
*Change from v0.7:* model to Claude; third arm; readable units; per-arm cap; the
outcome-free arm comparisons; prompt rationale.

**Dropped from v0.4 (future work):** Exp 1 (layout stability), Exp 3 (virtual trials),
Exp 6 (gate ablations, LLM judge, isolation forest), Exp 2 (spatial data), the 200 × 200
patch check. (SBC re-added in v0.6 as A1b.)

## 7. Reported limitations (decided before outcomes)

- Cisplatin only; no partner drug; one cycle for B_i.
- 2D, 2 × 2 mm patch, 8–84 days.
- Drug kill does not depend on cell cycle (the twin's faster-growing tumours respond
  less, against the literature).
- CAFs do not secrete IL-6; the CAF chemoresistance effect in the literature is not
  reproduced (crowding dominates).
- Drug potency and hypoxia protection are shared parameters known as ranges; they
  dominate B_i.
- GSE42127 fibroblast coverage; NPE z-scoring warning on macrophage, CD8 and CD4
  features (heavy tails).
- JBR.10 adenocarcinoma n = 71.
- Cell-cell co-occurrence: the twin does not reproduce the patient pattern in which pathway
  scores (hypoxia, TGF-β, IL-6, M2 and CAF share) rise as tumour content falls, nor the
  shared fibrotic-hypoxic axis. Likely in part the observation step (bulk scores diluted
  by tumour content, absent from the link) and in part missing biology (one shared
  infiltration driver; IL-6 from CAFs and vessels). Building the full twin that addresses
  this is the next phase and out of scope for this paper.

## 8. Model as built

As v0.4 Section 8, plus: fibroblast contact inhibition over 120 µm (N = 112), K_fib prior
0.005–0.2 (Nov 6, from the coverage check).

## 9. Code and data trail

Campaign `runs/phase5_v3_fib120_sims`; fitting `runs/phase5_v3_fib120_npe_logithyp_endolinked`
(`phase5_train_npe_v5`); B_i `runs/phase6_bi_direct_v2_sharedworld_floorboth`;
absolute response `runs/phase6_chemo_v2_GSE14814_3cycles_3draws_p{0..3}`;
sensitivity `runs/phase6_sensitivity_v1_bi192`; value of information
`runs/phase7_value_of_information_v1_bi192`; TumorTwin `runs/phase7_tumortwin_v1`.

## 10. Version history and amendments

- v0.1 Oct 28; v0.2, v0.3, v0.4 Nov 5 (see v0.4).
- **v0.5, Nov 8, 2026.** Model v3 (fibroblast 120 µm); observation choices from coverage;
  emulator dropped (R² ceiling 0.76) for direct simulation with shared worlds and both
  arms floored; Exp 4 precondition evaluated and failed (0.047 vs 0.144), Exp 4
  exploratory only; added A2–A4 and Exp 7 (AI agent, `gpt-6-astra`); Exp 5 kept as
  exploratory; Exp 1, 2, 3, 6 dropped (LC, Nov 8).

- **v0.6, Nov 8, 2026.** Added A1b (SBC), A1c (cell-cell dependencies), A3b/A4b (prior
  sensitivity); Exp 7 model changed to `claude-opus-5-5` with disclosure. Reviewer items
  1, 3, 7 (LC, Nov 8).
- **v0.7, Nov 8, 2026.** Added A1d (dependency equation); recorded A1b–A1d and A2 results;
  added the co-occurrence limitation (LC, Nov 8).
- **v0.8, Nov 8, 2026.** Exp 7 rewritten: Claude Opus 5.5, third arm (twin + guardrail),
  readable units, per-arm cap, outcome-free arm comparisons, prompt rationale with
  sources (LC, Nov 8).
- **v0.9, Nov 9, 2026.** Exp 4 precondition value corrected to the pre-registered definition
  (SD of posterior medians, 0.036; the earlier 0.047 used means). Verdict unchanged.
  v0.8 had been frozen; it is kept as `docs/preregistration_frozen_v0.8.md` with its
  SHA-256 in the run log, and v0.9 is frozen in its place. No outcome had been examined (LC).
- **v0.10, Nov 10, 2026.** Exp 7 deviation recorded: temperature not accepted by the model
  (default sampling used), default internal reasoning, 5 calls invalid at the token cap.
  v0.9 frozen copy kept as `docs/preregistration_frozen_v0.9.md`; v0.10 frozen in its
  place. No outcome had been examined (LC).
- **v0.11, Nov 10, 2026.** Exp 7b added (enforced guardrail arm; 1,000-token run; second
  model); features-only correlation with B_i defined as the baseline for reliance. No
  outcome had been examined (LC).
- **v0.12, Nov 10, 2026.** Added A1e (dependency gate, thresholds disclosed as set after
  A1c/A1d) and A5 (positive control for the guardrail, surrogate-based, criteria fixed
  before running). No outcome had been examined (LC).
- **v0.13, Nov 10, 2026.** Recorded Exp 7b results; added Exp 7c (MedGemma 27B, open and
  medical, served locally) with a stopping rule. No outcome had been examined (LC).

Amendments after freeze: none yet.
