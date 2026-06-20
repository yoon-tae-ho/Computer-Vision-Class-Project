# TALC++ Step Notes

This file is the handoff log for the staged TALC++ notebook rewrite.

Every step must:

1. Read this file before editing.
2. Respect decisions already recorded here.
3. Append a note under the matching step section before finishing.
4. Include changed files, verification commands, unresolved issues, and the recommended next prompt.

## Global Decisions

- Target notebook: `Project #3.ipynb`
- Original high-level prompt: `codex_project3_talcplusplus_final_prompt.md`
- Execution plan: `talcplusplus_step_prompts/00_design_review_and_execution_plan.md`
- Preserve the provided `IMBALANCECIFAR10` and `IMBALANCECIFAR100` classes.
- Use CIFAR-100 for final runs.
- Keep `BATCH_SIZE = 128`.
- Keep final per-model budget at or below 200 total epochs.
- Use validation split derived only from the imbalanced training split.
- Do not use the test split for tuning.
- Default final track: `split_calibrated`.
- Core implementation first; optional BCL/RIDE/WRN enhancements only after smoke mode works.

## Current Status

- Step 5 results/reporting completed. Core staged rewrite is complete; optional next prompt is `talcplusplus_step_prompts/06_optional_boost_prompt.md` only after runtime smoke tests pass.

## Step 1 Notes: Foundation

Append notes here after running `01_foundation_prompt.md`.

Template:

```markdown
### YYYY-MM-DD HH:MM - Step 1
- Changed files:
- What was implemented:
- Verification:
- Issues/blockers:
- Handoff to Step 2:
```

### 2026-06-20 20:02 KST - Step 1
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
- What was implemented:
  - Reorganized the notebook into a TALC++ foundation with clear markdown/code sections.
  - Preserved the provided `IMBALANCECIFAR10` and `IMBALANCECIFAR100` dataset classes.
  - Added global config, `SMOKE_TEST`, all four required scenarios, epoch-budget safeguards, `RUN_FINAL_TEST`, and `FINAL_SUBMISSION_TRACK = "split_calibrated"`.
  - Added reproducibility helpers, environment logging, transform builders, custom `Cutout`, protected stratified train/validation split, natural/balanced loaders, evaluation metrics, selection score, params/MACs/FLOPs hooks, and JSON/CSV artifact helpers.
  - Added explicit placeholders for Step 2 models/losses, Step 3 training, Step 4 orchestration/calibration/final test, and Step 5 reporting.
- Verification:
  - `python3` JSON load of `Project #3.ipynb`: passed.
  - Extracted code-cell syntax compile with `compile(...)`: passed.
  - `nbformat` validation could not run because `nbformat` is not installed in the local shell environment.
  - Full extracted foundation-code execution could not run because local `python3` is missing `numpy`.
- Issues/blockers:
  - Local shell environment lacks notebook/runtime dependencies (`numpy`; `nbformat` also unavailable). No CIFAR download or full notebook execution was attempted in Step 1.
- Handoff to Step 2:
  - Run `talcplusplus_step_prompts/02_models_losses_prompt.md` next.
  - Step 2 should replace the model/loss placeholders with `CifarResNet32`, `CifarResNet56`, model registry, LDAM, Balanced Softmax, label-aware smoothing, and classifier/calibration helper foundations.

## Step 2 Notes: Models and Losses

Append notes here after running `02_models_losses_prompt.md`.

Template:

```markdown
### YYYY-MM-DD HH:MM - Step 2
- Changed files:
- What was implemented:
- Verification:
- Issues/blockers:
- Handoff to Step 3:
```

### 2026-06-20 20:35 KST - Step 2
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
- What was implemented:
  - Replaced the model placeholder with CIFAR-native `CifarResNet32` and `CifarResNet56`.
  - Added the common TALC++ model interface: `forward_features`, `forward_classifier`, `forward(..., return_features=False)`, `reset_classifier`, `freeze_backbone`, `unfreeze_all`, `get_classifier`, and `set_classifier`.
  - Added `MODEL_REGISTRY` and `build_model(...)` with `resnet32` and `resnet56` aliases.
  - Replaced the loss placeholder with CE builder, `LDAMLoss`, class-balanced effective-number weights, `BalancedSoftmaxLoss`, soft-label Balanced Softmax, training-time logit adjustment, and `LabelAwareSmoothingCE`.
  - Added optional Mixup/CutMix helpers and soft-target helper.
  - Added classifier adaptation helpers: classifier state clone/restore, MaxNorm projection, classifier weight norms, and `set_backbone_eval_classifier_train`.
  - Added calibration foundations: group index tensors, default calibration dict, group-wise temperature vector, group-wise logit calibration, tau-normalization with state restore support, and calibration strength.
  - Added a non-automatic `run_step2_sanity_checks(...)` helper for random-tensor model/loss/FLOP checks once the notebook runtime has dependencies installed.
- Verification:
  - `python3` JSON load of `Project #3.ipynb`: passed.
  - Extracted code-cell syntax compile with `compile(...)`: passed.
  - `rg` confirmed required Step 2 classes/helpers are present.
  - `nbformat` validation could not run because `nbformat` is not installed in the local shell environment.
  - Full extracted-code execution and random-tensor sanity checks could not run because local `python3` is missing `numpy`.
- Issues/blockers:
  - Local shell environment still lacks runtime dependencies (`numpy`; `nbformat` also unavailable). No tensor smoke test was executed locally.
  - `WideResNet16_2` was intentionally deferred to avoid expanding scope before the core pipeline is stable.
- Handoff to Step 3:
  - Run `talcplusplus_step_prompts/03_training_pipeline_prompt.md` next.
  - Step 3 should implement Stage 1 representation training and BN-safe Stage 2 classifier adaptation using the implemented model/loss/helper functions.

## Step 3 Notes: Training Pipeline

Append notes here after running `03_training_pipeline_prompt.md`.

Template:

```markdown
### YYYY-MM-DD HH:MM - Step 3
- Changed files:
- What was implemented:
- Verification:
- Issues/blockers:
- Handoff to Step 4:
```

### 2026-06-20 21:32 KST - Step 3
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
- What was implemented:
  - Replaced the Stage 1/Stage 2 training placeholder with the core TALC++ training pipeline.
  - Added shared training utilities: epoch-budget assertion, run-dir resolution, scalar metric logging, checkpoint payload helpers, warmup-cosine LR, SGD builder, AMP scaler/autocast helpers, and `ModelEMA`.
  - Implemented `train_stage1_representation(...)` with `CE_StrongAug`, `LDAM_DRW`, `BalancedSoftmax`, and `LogitAdjustedCE`.
  - Implemented LDAM deferred re-weighting using effective-number class-balanced weights after `drw_start_epoch`.
  - Added optional Balanced Softmax Mixup/CutMix support when the corresponding alpha is positive; no-mixup remains the default.
  - Added Stage 1 EMA support and validation comparison between raw and EMA checkpoints.
  - Implemented `train_stage2_classifier(...)` with `none`, `cRT_no_reset`, `cRT_reset`, `cRT_no_reset_LabelAwareSmoothing`, and `cRT_no_reset_MaxNorm`.
  - Enforced BN-safe Stage 2 behavior by freezing the backbone, keeping frozen BN layers in eval mode, and training only the classifier.
  - Added best checkpoint saving for `best_stage1.pth` and `best_stage2.pth`, stage-specific CSV logs, and validation metrics JSON files.
  - Added a non-automatic `run_step3_smoke_training(...)` helper for a tiny split-trained run when dependencies/data are available.
- Verification:
  - `python3` JSON load of `Project #3.ipynb`: passed.
  - Extracted code-cell syntax compile with `compile(...)`: passed.
  - `rg` confirmed required Step 3 classes/functions and checkpoint/log paths are present.
  - `nbformat` validation could not run because `nbformat` is not installed in the local shell environment.
  - Full extracted-code execution and smoke training could not run because local `python3` is missing `numpy`.
- Issues/blockers:
  - Local shell environment still lacks runtime dependencies (`numpy`; `nbformat` also unavailable), so no tensor or CIFAR smoke training was executed locally.
  - Full scenario orchestration, calibration grid search, final frozen test evaluation, figures, and report generation remain intentionally deferred.
- Handoff to Step 4:
  - Run `talcplusplus_step_prompts/04_orchestration_calibration_prompt.md` next.
  - Step 4 should connect dataset/model/training pieces into bounded scenario search, validation-gated calibration, seed confirmation, selected config saving, and guarded final frozen evaluation.

## Step 4 Notes: Orchestration and Calibration

Append notes here after running `04_orchestration_calibration_prompt.md`.

Template:

```markdown
### YYYY-MM-DD HH:MM - Step 4
- Changed files:
- What was implemented:
- Verification:
- Issues/blockers:
- Handoff to Step 5:
```

### 2026-06-20 21:37 KST - Step 4
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
- What was implemented:
  - Replaced the calibration/orchestration placeholder with validation-gated calibration and bounded experiment orchestration.
  - Added calibration grids for `tau_norm`, post-hoc logit adjustment, and head/medium/tail temperatures, with identity/no-calibration always included.
  - Implemented `tune_calibration(...)` with validation-only selection, no-calibration fallback when calibration does not improve validation score, full grid CSV saving, and safe classifier state restore during tau-normalization search.
  - Implemented candidate policies for smoke and full modes: Stage 1 candidate search, top Stage 1 selection, Stage 2 candidate expansion, and calibration only after Stage 2.
  - Implemented `run_single_candidate(...)`, `run_scenario_model_search(...)`, `run_all_scenario_model_searches(...)`, and `run_seed_confirmation(...)`.
  - Added global summary output handling for `results/validation_candidate_summary.csv`, `results/validation_pareto_summary.csv`, `results/selected_configs.json`, `results/calibration_summary.csv`, and scenario-level selected config files.
  - Implemented guarded `run_final_frozen_evaluation(...)`, which reads frozen selected configs, loads saved final checkpoints, evaluates test only when `RUN_FINAL_TEST=True`, writes final test CSV/JSON, and keeps `FROZEN_RESULTS_DO_NOT_TUNE.txt` creation in the guarded final cell.
  - Added per-class test output helper for `per_class_test.csv` and `confusion_test.npy`.
- Verification:
  - `python3` JSON load of `Project #3.ipynb`: passed.
  - Extracted code-cell syntax compile with `compile(...)`: passed.
  - `rg` confirmed required Step 4 functions and output paths are present.
  - Confirmed `RUN_FINAL_TEST = False` remains the default and the final test function asserts the guard.
  - `nbformat` validation could not run because `nbformat` is not installed in the local shell environment.
  - Full extracted-code execution and smoke-mode training/search could not run because local `python3` is missing `numpy`.
- Issues/blockers:
  - Local shell environment still lacks runtime dependencies (`numpy`; `nbformat` also unavailable), so no CIFAR smoke search was executed locally.
  - Reporting, figures, report notes, and final notebook polish remain intentionally deferred to Step 5.
- Handoff to Step 5:
  - Run `talcplusplus_step_prompts/05_results_reporting_prompt.md` next.
  - Step 5 should add result-table generation, diagnostic/qualitative figure helpers, `report_notes.md` generation, final artifact polish, and final smoke-verification instructions.

## Step 5 Notes: Results and Reporting

Append notes here after running `05_results_reporting_prompt.md`.

Template:

```markdown
### YYYY-MM-DD HH:MM - Step 5
- Changed files:
- What was implemented:
- Verification:
- Issues/blockers:
- Final handoff:
```

### 2026-06-20 21:41 KST - Step 5
- Changed files:
  - `Project #3.ipynb`
  - `report_notes.md`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
- What was implemented:
  - Polished the notebook opening markdown so TALC++ is presented as a coherent three-stage proposed method.
  - Replaced the reporting placeholder with result-table generation helpers.
  - Added `generate_result_tables(...)` support for final summary normalization, ablation summary, per-class scenario aggregation, seed/calibration placeholder CSVs, final weight copying, and `report_artifact_summary.json`.
  - Added final weight artifact support that copies selected checkpoints to `logs/scenario_<type>_<factor>/selected/final_model.pth` and writes a neighboring metadata JSON sidecar.
  - Added matplotlib-optional diagnostic figure helpers for class distribution, per-class accuracy vs frequency, tail-to-head confusion, calibration heatmap, qualitative tail examples, and feature PCA using NumPy SVD.
  - Added graceful `.txt` figure placeholders when inputs or matplotlib are unavailable.
  - Added `generate_report_notes(...)`, `prepare_report_artifacts(...)`, and `run_static_smoke_verification(...)`.
  - Added an initial `report_notes.md` scaffold in the workspace; the notebook can regenerate it after experiments finish.
- Verification:
  - `python3` JSON load of `Project #3.ipynb`: passed.
  - Extracted code-cell syntax compile with `compile(...)`: passed.
  - `rg` confirmed required Step 5 reporting/figure/report-note functions are present.
  - Confirmed `RUN_FINAL_TEST = False` remains the default and the final test function still asserts the guard.
  - Confirmed only the Step 5 reporting scaffold remains as ordinary markdown; no unfinished `NotImplementedError` placeholders remain for Steps 1-5.
  - `nbformat` validation could not run because `nbformat` is not installed in the local shell environment.
  - Full extracted-code execution, random-tensor checks, and smoke-mode training/search could not run because local `python3` is missing `numpy`.
- Issues/blockers:
  - Local shell environment still lacks runtime dependencies (`numpy`; `nbformat` also unavailable), so no executable smoke test or CIFAR run was performed locally.
  - Before optional boost work, run the notebook in the target environment with dependencies installed and verify at least `SMOKE_TEST=True`.
- Final handoff:
  - Core staged rewrite is complete through reporting.
  - Recommended next runtime actions in a proper notebook environment:
    1. Set `SMOKE_TEST = True` and run top-to-bottom.
    2. Run `run_static_smoke_verification()`.
    3. Run `run_step2_sanity_checks()` if CUDA/CPU runtime is ready.
    4. Run `run_step3_smoke_training(download=True)` or `run_scenario_model_search("exp", 0.1)` in smoke mode.
    5. After real experiments finish, run `prepare_report_artifacts()`.
  - Use `talcplusplus_step_prompts/06_optional_boost_prompt.md` only after the core smoke path is stable.

## Optional Step 6 Notes: Score Boosts

Append notes here after running `06_optional_boost_prompt.md`.

Template:

```markdown
### YYYY-MM-DD HH:MM - Step 6
- Changed files:
- Optional features added:
- Verification:
- Issues/blockers:
- Final handoff:
```
