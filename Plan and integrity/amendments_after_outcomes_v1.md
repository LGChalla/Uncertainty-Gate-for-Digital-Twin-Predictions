# Analyses added after the outcome analysis (post hoc)

The pre-registration (v0.13, frozen 10 Nov 2026 16:10 CT, SHA-256 5669f034…eff48c; OSF
https://osf.io/vzqc5) is not changed. The pre-registered outcome analysis
(`phase8_outcomes_v3`) was run once on 11 Nov 2026 and is reported as run. The analyses below
were decided on 11 Nov 2026 (LC) AFTER its results had been seen. They are labelled post hoc
everywhere they appear and are reported whatever they show.

1. **All JBR.10 histologies** (`phase8_outcomes_v4 --all-histologies`). The pre-registered
   analysis used the 71 adenocarcinoma patients. Every JBR.10 patient with RNA, treatment and
   survival (about 133) is analysed with the same model, adding histology (adenocarcinoma
   yes/no) as a covariate. Reason: the adenocarcinoma restriction leaves only very strong
   effects detectable; randomisation holds for all histologies, and the twin and all agent
   arms were already computed for all patients. Caveat: the twin's biology and the paper's
   question are about adenocarcinoma.
2. **Second prompt wording for the agent** (`phase7_llm_predict_v6 --prompt-variant B`).
   Same content and numbers, different wording, fixed in the script before running.
   `claude-opus-5-5`, 1,000 tokens, all four arms, 133 × 3 repeats, compared with the
   original wording at 1,000 tokens (`phase7_llm_arms_compare_v2`). Reason: reviewers will ask
   whether the arm results depend on wording. The arm comparisons themselves are outcome-free;
   the scores also enter the post hoc outcome model as exploratory comparators.

11 Nov (added after the freeze): Gemma 3 27B (open, general purpose; the base MedGemma 27B was tuned from) and Llama 3.3 70B (open, general purpose, Meta) run in the four agent arms with the same prompts, budget and serving settings as MedGemma 27B. Purpose: separate medical tuning from open weights, and test a third model family. Reliance analysis as pre-registered; scores enter the outcome model only as a post hoc addition.
