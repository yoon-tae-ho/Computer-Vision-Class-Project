# Step 4 Prompt: Orchestration and Calibration

You are editing `Project #3.ipynb`.

Before editing:

1. Read `talcplusplus_step_prompts/STEP_NOTES.md`.
2. Read `talcplusplus_step_prompts/00_design_review_and_execution_plan.md`.
3. Inspect the notebook after Steps 1-3.

Goal for this step:

Connect the implemented pieces into a bounded, validation-driven experiment runner and implement validation-gated calibration.

Required scenario support:

```python
SCENARIOS = [
    ("exp", 0.1),
    ("exp", 0.01),
    ("step", 0.1),
    ("step", 0.01),
]
```

Required orchestration functions:

```python
def run_single_candidate(config, final_test=False):
    ...

def run_scenario_model_search(imb_type, imb_factor):
    ...

def run_seed_confirmation(selected_config):
    ...

def run_final_frozen_evaluation(selected_configs):
    ...
```

Bounded candidate policy:

- In smoke mode:
  - one scenario,
  - `resnet32`,
  - tiny epochs,
  - at most CE and one TALC++ candidate.

- In full mode:
  - all four scenarios,
  - start with `resnet32`,
  - Stage 1 candidates:
    - `CE_StrongAug`
    - `LDAM_DRW`
    - `BalancedSoftmax`
  - choose top 2 Stage 1 checkpoints by validation score,
  - Stage 2 candidates only on selected Stage 1 checkpoints:
    - `none`
    - `cRT_no_reset`
    - `cRT_reset`
    - `cRT_no_reset_LabelAwareSmoothing`
  - calibration only on top Stage 2 candidate(s).

Required calibration implementation:

```python
def tune_calibration(model, val_loader, cls_num_list, device, groups):
    ...
```

Calibration dictionary:

```python
{
    "tau_norm": 0.0,
    "logit_adjust_tau": 0.0,
    "T_head": 1.0,
    "T_medium": 1.0,
    "T_tail": 1.0,
}
```

Calibration requirements:

- Include identity/no-calibration in every grid.
- Search by validation `selection_score`.
- If calibration does not improve over uncalibrated validation score, select identity calibration.
- Do not permanently modify classifier weights during tau-norm search unless the selected candidate is restored intentionally and saved.
- Save full calibration grid table.

Required result files:

- `results/validation_candidate_summary.csv`
- `results/validation_pareto_summary.csv`
- `results/selected_configs.json`
- `results/calibration_summary.csv`
- scenario-level candidate logs under `logs/scenario_<type>_<factor>/...`

Final test guard:

- Add a final cell named exactly or very close to:
  `FINAL FROZEN TEST EVALUATION - DO NOT RUN UNTIL CONFIGS ARE FIXED`
- Gate it behind `RUN_FINAL_TEST = False`.
- Final evaluation must read frozen selected configs.
- It must not perform any search, calibration tuning, seed selection, or config mutation.
- After final test evaluation, write `FROZEN_RESULTS_DO_NOT_TUNE.txt`.

Seed policy:

- Search with `SEARCH_SEEDS = [0]`.
- Confirm selected config with `CONFIRM_SEEDS = [0, 1, 2]` only when feasible.
- Pick final seed by validation score only.
- Do not pick seed by test accuracy.

Verification:

- Validate notebook JSON.
- Run `SMOKE_TEST=True` if data/environment permits.
- Confirm that smoke mode writes selected config and summary artifacts.
- Confirm final test cell does nothing unless `RUN_FINAL_TEST=True`.

Scope limits:

- Do not add BCL/RIDE-Lite here.
- Do not add large visualizations here unless already trivial.
- Do not run full 200-epoch training as part of this step.

Before finishing:

1. Append a note to `talcplusplus_step_prompts/STEP_NOTES.md` under "Step 4 Notes: Orchestration and Calibration".
2. Include changed files, verification commands/results, blockers, and the next recommended prompt:
   `talcplusplus_step_prompts/05_results_reporting_prompt.md`
