# Uncertainty Gate for Digital Twin Predictions

A mechanistic digital twin of the lung adenocarcinoma microenvironment is fitted to each
patient from one tumor RNA sample and given cisplatin. The twin reports how much each
patient's predicted response changes across what it cannot pin down: its uncertainty.
The uncertainty gate compares how much patients differ with that uncertainty, and decides
whether the twin's number may be passed on to AI agents that generate evidence from the
trial and observational cohorts.

---

## 1. How the project was built: by phase

The work was built and run in phases:

| Phase | What was built |
|---|---|
| Data pipeline | Download the four cohorts and turn each tumor's RNA into 12 features |
| Phase 4 | The model, and checks that each mechanism moves the tumor the right way |
| Phase 5 | 20,000 simulated tissues, coverage and dependency checks, fitting each patient |
| Phase 6 | Cisplatin in each patient's twin; what drives the response; the uncertainty gate tested on synthetic cohorts |
| Phase 7 | AI agents given the twin's number (single handoff); value of information; a well-mixed comparator |
| Phase 8 | Survival analysis in the randomized JBR.10 trial and three other cohorts |
| Phase 9 | The multi-agent pipeline |

Each change was saved as a new version (`_v2`, `_v3`, …). The final files build on some
earlier versions, which are included in `Earlier versions/`.

## 2. How this repository is organized: by step

To make the work easier to follow, this repository is organized by the steps of the study
(Figure 1 of the paper) rather than by phase:

```
Step 1 Data Pipeline/                 RNA → 12 features per patient
Step 2 building mechanistic twin/
    The model/                        the twin itself (final versions)
        Earlier versions/             earlier files the final versions build on
    Checking the model/               steady state, directions, coverage, dependency checks
    Fitting each patient/             neural posterior estimation and its calibration
    Cisplatin in each twin/           the response B, three cycles, drivers, value of information
Step 3 uncertainty gate/              the gate tested on synthetic cohorts; well-mixed comparator
Step 4 AI agents/                     single handoff (four arms) and the multi-agent pipeline
Step 5 evaluation against survival/   survival analysis
Plan and integrity/                   the frozen analysis plan, later amendments, run log
```

Each file's header states what it does and, under **Needs**, the other files it uses.

## 3. How to run it

### Build the run folder

The code was written and run in the phase layout described in Section 1, so to run it,
first rebuild that layout from the step folders. From the repository's top folder:

```bash
mkdir -p run/tme_twin run/scripts run/tests run/data_pipeline run/docs run/runs
cp "Step 1 Data Pipeline/"*                                                   run/data_pipeline/
cp "Step 2 building mechanistic twin/The model/"*.py                          run/tme_twin/
cp "Step 2 building mechanistic twin/The model/Earlier versions/"*.py         run/tme_twin/
cp "Step 4 AI agents/pipeline_v1.py" "Step 4 AI agents/gates.py"            run/tme_twin/
cp "Step 2 building mechanistic twin/Checking the model/test_population_v3.py" run/tests/
cp "Step 2 building mechanistic twin/Checking the model/"*.py \
   "Step 2 building mechanistic twin/Fitting each patient/"*.py \
   "Step 2 building mechanistic twin/Cisplatin in each twin/"*.py \
   "Step 3 uncertainty gate/"*.py \
   "Step 4 AI agents/"*.py "Step 4 AI agents/"*.sh \
   "Step 5 evaluation against survival/"*.py                                  run/scripts/
rm run/scripts/test_population_v3.py run/scripts/pipeline_v1.py run/scripts/gates.py
cp "Plan and integrity/preregistration_v0.13.md"                              run/docs/preregistration_frozen.md
cp "Plan and integrity/runlog.jsonl"                                          run/runs/
cp requirements.txt run/
cd run
```

All commands below run from inside `run/`.

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sbi==0.27 lifelines scikit-learn
```

- **Step 1** also needs R with `EPIC`, `GSVA`, `msigdbr`, `hgu133a.db`, `illuminaHumanv3.db` and `org.Hs.eg.db`.
- **Open models** in Step 4 are served with vLLM, in a separate environment.
- Step 4 reads `ANTHROPIC_API_KEY` (Claude models) and `HF_TOKEN` (open-model weights) from the environment.

Steps 2–4 need a GPU. Seeds are fixed.

### Commands, in order

Below, `<sims>` = `runs/phase5_v3_fib120_sims`,
`<npe>` = `runs/phase5_v3_fib120_npe_logithyp_endolinked`, and
`<bi>` = `runs/phase6_bi_direct_v2_sharedworld_floorboth`.

**Step 1. Data → features**
```bash
python data_pipeline/geo.py --gse GSE14814 GSE68465 GSE42127 --out data/raw
python data_pipeline/tcga.py --out data/raw/TCGA-LUAD
python data_pipeline/flow.py --raw data/raw --out data/processed
python data_pipeline/deconvolve.py --renv ~/renv
python data_pipeline/qc_features.py
python data_pipeline/features_v2.py
```

**Step 2. The twin**
```bash
# check the model
python tests/test_population_v3.py
python scripts/phase4_directions_v2.py --device cuda --param <parameter>
# 20,000 simulated tissues in four shards (a: 0–5000, b: 5000–10000, c: 10000–15000, d: 15000–20000), then combine
python scripts/phase5_simulate_v3.py --device cuda --n 20000 --start 0 --end 5000 --out runs/phase5_v3_fib120_sims_a
python scripts/phase5_simulate_v3.py --combine runs/phase5_v3_fib120_sims_{a,b,c,d} --out <sims>
python scripts/phase5_coverage_check_v5.py --sims <sims> --transform logit --endothelial linked --out runs/phase5_v3_fib120_coverage_full_logithyp_endolinked
python scripts/phase5_dependency_check_v1.py    --sims <sims> --cohorts GSE14814 GSE68465 GSE42127 TCGA-LUAD --out runs/phase5_dependency_check_v1_4cohorts
python scripts/phase5_dependency_equation_v1.py --sims <sims> --cohorts GSE14814 GSE68465 GSE42127 TCGA-LUAD --out runs/phase5_dependency_equation_v1_4cohorts
python scripts/phase5_dependency_gate_v1.py     --sims <sims> --cohorts GSE14814 GSE68465 GSE42127 TCGA-LUAD --out runs/phase5_dependency_gate_v1_4cohorts
# fit each patient
python scripts/phase5_train_npe_v5.py --device cuda --sims <sims> --out <npe>
python scripts/phase5_sbc_v1.py --device cuda --sims <sims> --cohort GSE14814 --n-test 300 --n-post 200 --out runs/phase5_sbc_v1_GSE14814_300test_200post
# cisplatin in each twin (10 shared worlds, 192 h), then collect B
python scripts/phase6_patient_chemo_v3.py --device cuda --shared-world --npe <npe> --cohort GSE14814 --draws-per-patient 10 --hours 192 --every 24 --out runs/phase6_chemo_v3_GSE14814_1cycle_10draws_sharedworld_p0
python scripts/phase6_bi_from_runs_v2.py --runs runs/phase6_chemo_v3_GSE14814_1cycle_10draws_sharedworld_p* --cohort GSE14814 --hour 192 --out <bi>
# three cycles; drivers; value of information
python scripts/phase6_patient_chemo_v2.py --device cuda --cycles 3 --hours 2016 --every 168 --npe <npe> --cohort GSE14814 --draws-per-patient 3 --out runs/phase6_chemo_v2_GSE14814_3cycles_3draws_p0
python scripts/phase6_absolute_response_v1.py --runs runs/phase6_chemo_v2_GSE14814_3cycles_3draws_p* --out runs/phase6_absolute_response_v1_GSE14814_3cycles_3draws
python scripts/phase6_sensitivity_v1.py --sims <sims> --npe <npe> --out runs/phase6_sensitivity_v1_bi192
python scripts/phase7_value_of_information_v1.py --sims <sims> --out runs/phase7_value_of_information_v1_bi192
```

**Step 3. The uncertainty gate**
```bash
python scripts/phase6_guardrail_positive_control_v1.py --device cuda --sims <sims> --cohort GSE14814 --real-draws <npe>/draws_GSE14814.npz --cohorts 20 --out runs/phase6_guardrail_positive_control_v1_GSE14814_20cohorts_133pt_10draws
python scripts/phase7_tumortwin_v1.py --npe <npe> --out runs/phase7_tumortwin_v1
```

**Step 4. AI agents**
```bash
# single handoff, Claude (four arms)
for ARM in features features+twin features+twin+guardrail features+twin+enforced; do
  python scripts/phase7_llm_predict_v4.py --provider anthropic --model claude-opus-5-5 --arm $ARM --cohort GSE14814 --max-tokens 1000 --npe <npe> --bi <bi> --max-dollars 40 --out runs/phase7_llm_v4_GSE14814_3rep
done
# repeat with --max-tokens 300 and with --model claude-sonnet-5-5; second prompt wording:
#   phase7_llm_predict_v6.py ... --prompt-variant B --out runs/phase7_llm_v6_promptB_GSE14814_3rep
# compare the arms of each run
python scripts/phase7_llm_arms_compare_v2.py --bi <bi> --cohort GSE14814 --arm features=<jsonl> --arm features+twin=<jsonl> --arm features+twin+guardrail=<jsonl> --arm features+twin+enforced=<jsonl> --out runs/phase7_llm_arms_compare_v2_claude-opus-5-5_tok1000_4arms

# single handoff, open models: serve one model, then run its four arms and the comparison
bash scripts/serve_medgemma_v1.sh <gpu> fp8 8000
bash scripts/run_open_model_arms_v1.sh google/medgemma-27b-text-it 8000 <npe> <bi> medgemma-27b
bash scripts/serve_gemma3_v2.sh <gpu> fp8 8000
bash scripts/run_open_model_arms_v1.sh google/gemma-3-27b-it 8000 <npe> <bi> gemma-3-27b
bash scripts/serve_llama33_v2.sh <gpu> fp8 8001
bash scripts/run_open_model_arms_v1.sh meta-llama/Llama-3.3-70B-Instruct 8001 <npe> <bi> llama-3.3-70b

# multi-agent pipeline (10 runs per condition: no gate, gate as warning, gate enforced)
bash scripts/serve_llama33_v3.sh <gpu A> <gpu B> 8001        # Llama 3.3 70B across two GPUs
python scripts/phase9_multiagent_v1.py --provider local --base-url http://localhost:8001/v1 --model meta-llama/Llama-3.3-70B-Instruct --runs 10 --bi <bi> --out runs/phase9_multiagent_v1_llama33_3cond_10runs
bash scripts/serve_gemma3_v2.sh <gpu> fp8 8000
python scripts/phase9_multiagent_v1.py --provider local --base-url http://localhost:8000/v1 --model google/gemma-3-27b-it --runs 10 --bi <bi> --out runs/phase9_multiagent_v1_gemma3_3cond_10runs
python scripts/phase9_summarise_v1.py --runs runs/phase9_multiagent_v1_*_3cond_10runs --out runs/phase9_summary_v1_llama33_gemma3
python scripts/phase9_reliance_v1.py  --runs runs/phase9_multiagent_v1_*_3cond_10runs --bi <bi> --out runs/phase9_reliance_v1_llama33_gemma3
```

**Step 5. Survival**
```bash
python scripts/phase8_outcomes_v3.py --freeze docs/preregistration_frozen.md --twin <bi> --tumortwin runs/phase7_tumortwin_v1 \
    --llm runs/phase7_llm_v3_claude-opus-5-5_GSE14814_3rep runs/phase7_llm_v4_GSE14814_3rep runs/phase7_llm_v5_GSE14814_3rep \
    --osf https://osf.io/vzqc5 --precondition-failed --out runs/phase8_outcomes_v3_exploratory_osf-vzqc5
# all 133 JBR.10 patients, histology as a covariate
python scripts/phase8_outcomes_v4.py <same arguments; --llm also includes runs/phase7_llm_v6_promptB_GSE14814_3rep> --all-histologies --out runs/phase8_outcomes_v4_POSTHOC_allhistologies_JBR10
```

## 4. Data

All data are public:
- **GEO:** GSE14814 (JBR.10 trial), GSE68465, GSE42127.
- **TCGA-LUAD:** expression from the UCSC Xena GDC hub; clinical and treatment records from the GDC API.

The Step 1 scripts download them.

## 5. Plan

The gate and the survival analysis were fixed before survival data were read. The plan is
on OSF: https://osf.io/vzqc5.
