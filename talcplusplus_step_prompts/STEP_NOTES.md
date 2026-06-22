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

### 2026-06-21 00:10 KST - Server Smoke Validation After Steps 1-5
- Changed files:
  - `Project #3.ipynb`
  - `.codex_project3_smoke_runner.py`
  - `report_notes.md`
  - generated `data/`, `logs/`, and `results/` artifacts
- What was verified:
  - Server environment `cv-proj` has Python 3.11, NumPy, PyTorch, Torchvision, Matplotlib, nbformat, notebook, and Jupyter.
  - Unsandboxed GPU runtime sees `NVIDIA TITAN RTX`; notebook smoke runner used `DEVICE=cuda`.
  - `Project #3.ipynb` JSON load passed.
  - Extracted code-cell syntax compile passed.
  - nbformat validation passed with only `MissingIDFieldWarning`.
  - CIFAR-100 tarball was downloaded from a Hugging Face mirror after the official Toronto URL was unusably slow; md5 matched the official value `eb9058c3a382ffc7106e4002c42a8d85`.
- Runtime fixes made:
  - Added `include_test=False` default to `build_datasets_for_scenario(...)` so ordinary smoke/search paths do not instantiate the CIFAR-100 test split. The guarded final evaluation path passes `include_test=True`.
  - Set `NUM_WORKERS = 0` in `SMOKE_TEST` mode to avoid sandbox multiprocessing/socket failures; full runs still use 4 workers.
  - Fixed `LDAMLoss` CUDA autocast dtype handling by computing LDAM logits/margins/weights in float32.
  - Lowered smoke-only LDAM candidate LR to `0.01` to avoid NaN loss under the very short no-warmup smoke schedule. Full non-smoke grid is unchanged.
- Smoke command:
  - `conda run --no-capture-output -n cv-proj python -u .codex_project3_smoke_runner.py`
- Smoke result:
  - Completed successfully on GPU in 131.11 seconds.
  - `run_static_smoke_verification()` passed.
  - `run_step2_sanity_checks()` passed.
  - `run_step3_smoke_training(download=True)` passed and wrote `logs/scenario_exp_0p1/candidates/step3_smoke/best_stage1.pth` and `best_stage2.pth`.
  - `run_scenario_model_search("exp", 0.1)` passed.
  - `prepare_report_artifacts()` passed.
- Generated artifacts:
  - `results/static_smoke_verification.json`
  - `results/validation_candidate_summary.csv` with 4 smoke candidate rows
  - `results/validation_pareto_summary.csv`
  - `results/calibration_summary.csv`
  - `results/selected_configs.json`
  - `results/report_artifact_summary.json`
  - `results/final_weight_artifacts.csv`
  - `logs/scenario_exp_0p1/selected/final_model.pth`
  - `logs/scenario_exp_0p1/selected/selected_config.json`
  - selected diagnostic figures/placeholders under `logs/scenario_exp_0p1/selected/figures/`
- Test-split guard:
  - `RUN_FINAL_TEST` remained `False`.
  - No `results/*test*` files were generated.
  - `results/report_artifact_summary.json` reports `final_test_rows: 0`.
- Remaining risks:
  - This was smoke-only validation for `scenario_exp_0p1`; no full four-scenario training has been run yet.
  - Some report figures are placeholders because final test/per-class selected inputs are not available in smoke mode.
  - Extra diagnostic LDAM check directories from debugging remain under `logs/scenario_exp_0p1/candidates/`; they are not selected configs.
- Final handoff:
  - Core Steps 1-5 smoke path is stable on the server.
  - Step 6 optional boost was not run.
  - Full run should wait for explicit approval.

Template:

```markdown
### YYYY-MM-DD HH:MM - Step 6
- Changed files:
- Optional features added:
- Verification:
- Issues/blockers:
- Final handoff:
```

### 2026-06-21 00:18 KST - Step 6
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
  - `report_notes.md` was regenerated as a side effect of loading notebook definitions for static verification.
- Optional features added/connected:
  - Connected existing `CifarResNet56` as a non-smoke optional model candidate only for the hardest `exp/0.01` scenario; smoke mode remains `resnet32` only.
  - Kept existing Stage 1 EMA implementation without reimplementation; candidate configs still disable EMA in smoke and enable it only outside smoke.
  - Connected existing `cRT_no_reset_MaxNorm` as a non-smoke Stage 2 validation candidate with `maxnorm_radius=2.0`; smoke Stage 2 grid remains `none` and `cRT_no_reset`.
  - Added optional BN alignment helpers:
    - default `bn_align_mode="none"`;
    - non-smoke `recompute_balanced_bn` candidate only for `cRT_no_reset`;
    - BN stats are recomputed with no gradients on the balanced training loader after ordinary cRT;
    - aligned stats are kept only if validation selection score improves, otherwise the pre-alignment model state is restored.
  - BCL, WRN, RIDE-Lite, and large Mixup/CutMix tuning were intentionally not added.
- Verification:
  - `python3` JSON load of `Project #3.ipynb`: passed.
  - Extracted code-cell syntax compile with `compile(...)`: passed.
  - `conda run --no-capture-output -n cv-proj python - <<'PY' ... nbformat.validate(...) ... PY`: passed with the existing `MissingIDFieldWarning`.
  - Static source checks confirmed `RUN_FINAL_TEST = False`, `BATCH_SIZE = 128`, hard-scenario ResNet56 routing, optional BN alignment, and MaxNorm candidate wiring.
  - Grid-only runtime check confirmed smoke candidates stay `resnet32`/2 Stage 2 configs, while non-smoke optional routing adds `resnet56` only for `exp/0.01` plus `recompute_balanced_bn` and `cRT_no_reset_MaxNorm` candidates.
  - `conda run --no-capture-output -n cv-proj python -c "import torch; ..."` showed `cuda_available False`, so the long smoke runner was not forced on CPU.
  - Loaded notebook definitions in `cv-proj` and ran `run_static_smoke_verification()` on CPU: passed, including new optional-boost checks.
  - `find results -maxdepth 1 -iname '*test*' -print`: no output.
- Issues/blockers:
  - GPU is not visible inside the current sandbox, so `.codex_project3_smoke_runner.py` was not rerun.
  - Optional Step 6 candidates have not been trained or selected in a full run.
  - Existing `results/selected_configs.json` remains smoke output and must not be treated as a final frozen config.
- Final handoff:
  - When GPU is visible, rerun smoke with `conda run --no-capture-output -n cv-proj python -u .codex_project3_smoke_runner.py`.
  - Full four-scenario training and any final frozen evaluation require explicit user approval.
  - Keep `RUN_FINAL_TEST=False` until validation-selected configs are deliberately frozen.

### 2026-06-21 01:10 KST - Step 6 Recovery After IDE Save
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
  - `report_notes.md` was regenerated as a side effect of loading notebook definitions for verification.
  - `results/static_smoke_verification.json` was refreshed by `run_static_smoke_verification()`.
- What was restored/cleaned:
  - Restored Step 6 optional wiring that had been overwritten by an older IDE notebook buffer:
    - `ENABLE_OPTIONAL_STEP6_BOOSTS`;
    - `BN_ALIGN_CANDIDATES`;
    - `model_candidates_for_scenario(...)`;
    - optional `recompute_balanced_bn` BN alignment helper and candidate wiring;
    - non-smoke `cRT_no_reset_MaxNorm` Stage 2 candidate;
    - selected-config/result fields for `bn_alignment` and `bn_align_selected_mode`.
  - Restored smoke-safe `NUM_WORKERS = 0 if SMOKE_TEST else 4`.
  - Cleared the stale first-cell `ModuleNotFoundError: No module named 'torch'` output.
  - Updated notebook kernelspec metadata to `Python (cv-proj)` / `cv-proj`.
- Verification:
  - `python3` JSON load: passed.
  - Extracted code-cell syntax compile: passed.
  - Confirmed first code cell has zero saved outputs.
  - Confirmed `RUN_FINAL_TEST = False` and no `RUN_FINAL_TEST = True` source assignment.
  - `conda run --no-capture-output -n cv-proj python - <<'PY' ... nbformat.validate(...) ... PY`: passed with the existing `MissingIDFieldWarning`.
  - Loaded notebook definitions in `cv-proj` without training and ran `run_static_smoke_verification()`: passed.
  - Grid-only runtime check confirmed non-smoke `exp/0.01` model candidates are `["resnet32", "resnet56"]`, other scenarios stay `["resnet32"]`, and Stage 2 candidates include `recompute_balanced_bn` and `cRT_no_reset_MaxNorm`.
  - `find results -maxdepth 1 -iname '*test*' -print`: no output.
- Issues/blockers:
  - No smoke or full training was rerun in this recovery step.
  - Current `results/selected_configs.json` remains smoke output and is not a final frozen config.
- Final handoff:
  - Use the `Python (cv-proj)` kernel before running notebook cells.
  - Keep `RUN_FINAL_TEST=False`.
  - For the next real experiment, run validation-only full search explicitly with `run_all_scenario_model_searches()` and then `prepare_report_artifacts()`.

### 2026-06-21 01:13 KST - Run All Entrypoint
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
- What was implemented:
  - Added `RUN_PIPELINE_ON_RUN_ALL = True` and `RUN_SEED_CONFIRMATION_ON_RUN_ALL = False` config flags.
  - Replaced the final report-scaffold-only behavior with a notebook-only Run All entrypoint:
    - asserts `RUN_FINAL_TEST is False`;
    - runs `run_static_smoke_verification()`;
    - runs `run_all_scenario_model_searches()`;
    - optionally runs seed confirmation only if explicitly enabled;
    - runs `prepare_report_artifacts()`.
  - Added `running_inside_notebook()` so extracted-code smoke runners and command-line validation do not accidentally start full training.
- Verification:
  - Extracted code-cell syntax compile: passed.
  - Source checks confirmed the Run All entrypoint and `RUN_FINAL_TEST=False` guard are present.
  - Executed extracted notebook code in `cv-proj` outside a notebook: it printed `Notebook Run All entrypoint skipped outside an interactive notebook kernel.` and did not start training.
  - `nbformat.validate(...)`: passed with the existing `MissingIDFieldWarning`.
  - `find results -maxdepth 1 -iname '*test*' -print`: no output.
- Final handoff:
  - In the IDE, select the `Python (cv-proj)` kernel.
  - Press `Run All` to start validation-only full search from the notebook button.
  - Stop immediately if the first config cell prints `Device: cpu`; full search should run on CUDA.
  - Keep `RUN_FINAL_TEST=False`; final test is still not part of Run All.

### 2026-06-21 11:54 KST - LDAM AMP Dtype Fix
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
- Issue:
  - Full Run All failed during the `LDAM_DRW` Stage 1 candidate with `RuntimeError: scatter(): Expected self.dtype to be equal to src.dtype`.
  - Root cause was CUDA autocast producing half-precision logits while LDAM margins/weights stayed float32, so `scatter_` received mixed dtypes.
- Fix:
  - Updated `LDAMLoss.forward(...)` to cast logits to float32 internally, compute margins/target logits/adjusted logits/weights in the same float32 dtype, and return float32 cross entropy.
  - Cleared saved notebook outputs so the failed traceback is not preserved in the notebook file.
- Verification:
  - `python3` JSON load and extracted code-cell syntax compile: passed.
  - Minimal LDAM smoke in `cv-proj` with half-precision logits: passed and returned a float32 loss.
  - Confirmed all saved code-cell outputs are cleared.
  - `nbformat.validate(...)`: passed with the existing `MissingIDFieldWarning`.
  - `find results -maxdepth 1 -iname '*test*' -print`: no output.
- Final handoff:
  - Restart the notebook kernel before rerunning `Run All` so the fixed `LDAMLoss` class is redefined.
  - Keep `RUN_FINAL_TEST=False`.

### 2026-06-22 KST - Frozen Final-Test Run All Prep
- Changed files:
  - `Project #3.ipynb`
  - `talcplusplus_step_prompts/STEP_NOTES.md`
  - `results/frozen_selected_configs.json`
  - `results/freeze_manifest.json`
  - `FROZEN_SELECTED_CONFIGS_DO_NOT_TUNE.txt`
- What was prepared:
  - Snapshotted the validation-selected configs from `results/selected_configs.json` into `results/frozen_selected_configs.json`.
  - Added a mode-aware notebook Run All entrypoint:
    - `RUN_FINAL_TEST = False` remains the notebook-start default.
    - `RUN_ALL_MODE = "final_test"` makes the final Run All cell execute frozen final evaluation instead of rerunning validation search.
    - Final evaluation reads `results/frozen_selected_configs.json`.
    - `ALLOW_FINAL_TEST_RERUN = False` prevents accidental repeat final-test execution if `results/final_test_summary.csv` already exists.
  - Added notebook helpers to freeze selected configs, load frozen configs, and run final test from the frozen snapshot.
- Verification:
  - `python` JSON load and extracted code-cell syntax compile: passed.
  - `nbformat.validate(...)`: passed with the existing `MissingIDFieldWarning`.
  - Confirmed `RUN_FINAL_TEST = False` is still the initial config.
  - Confirmed `RUN_ALL_MODE = "final_test"` is set for the next notebook Run All.
  - Confirmed `results/frozen_selected_configs.json` contains four selected scenario configs.
  - `find results -maxdepth 1 -iname '*test*' -print`: no output.
  - `find logs -path '*selected*' \( -iname '*test*' -o -iname 'confusion_test.npy' -o -iname 'per_class_test.csv' \) -print`: no output.
- Not run:
  - Final frozen test evaluation was not completed by Codex.
  - A direct Codex-side attempt was interrupted before writing final-test artifacts, per user request to run final test from the notebook UI.
  - No validation retraining or seed confirmation was rerun.
- Remaining risk:
  - Pressing `Run All` with `RUN_ALL_MODE = "final_test"` will use the test split. Do not change configs after that point.
  - If the first config cell prints `Device: cpu`, stop and switch to a CUDA-visible kernel before final test.
- Final handoff:
  - Use VS Code's `Run All` on `Project #3.ipynb` with the `Python (cv-proj)` kernel.
  - The run should skip validation search and execute the frozen final test from `results/frozen_selected_configs.json`.
