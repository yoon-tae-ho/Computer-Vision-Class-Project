# TALC++ Design Review and Execution Plan

Target notebook: `Project #3.ipynb`  
Original prompt reviewed: `codex_project3_talcplusplus_final_prompt.md`  
Date context: 2026-06-20, with the assignment due on 2026-06-23 at 23:00.

## Verdict

The original TALC++ design is directionally strong and fits the assignment constraints, but it is too large for a safe one-pass rewrite of the current notebook.

The current notebook is a compact CE baseline with:

- provided `IMBALANCECIFAR10` / `IMBALANCECIFAR100` dataset classes,
- one global scenario,
- simple CIFAR transforms,
- an ImageNet-style ResNet18-like model,
- a single CE training loop,
- only overall/head/tail evaluation.

The proposed design asks for a full experiment framework: multiple models, multiple losses, validation-only model selection, staged training, calibration, seed confirmation, FLOP counting, artifact logging, figures, and report notes. That is a complete rewrite. It should be split.

## Fit With The Project

The design matches the project goal well in these ways:

- It uses only the provided imbalanced CIFAR datasets.
- It explicitly covers the required four scenarios: `exp/step` x `0.1/0.01`.
- It respects batch size 128 and the 200-epoch cap per final model.
- It introduces a train-derived validation split and isolates the test split.
- TALC++ is framed as a coherent long-tail method rather than a raw hyperparameter search.
- The three-stage structure is defensible:
  - tail-aware representation learning,
  - BN-safe classifier adaptation,
  - validation-gated calibration.
- It requires params/FLOPs/MACs and qualitative outputs, which the assignment asks for.

## Main Risks In The Original Design

1. Scope explosion:
   - `3+` backbones x `6` Stage 1 variants x schedules x Stage 2 variants x BN alignment x calibration x seeds x four scenarios is not realistic in a notebook under the deadline.

2. Optional methods are too heavy for the first implementation:
   - BCL memory queues, RIDE-Lite experts, LWS, BN alignment, Mixup/CutMix, EMA, and full seed confirmation are useful, but each adds a failure surface.

3. Validation coverage can be sparse:
   - In `exp-0.01`, some tail classes may have few or zero validation samples. The design handles this by logging coverage, but model selection must not over-trust noisy tail validation.

4. Full-train retraining is methodologically delicate:
   - If calibration is tuned on a split-trained model, then retraining on full train changes logits and feature statistics. For the first complete version, use `split_calibrated` as the default final track.

5. The current baseline model is not the right main model:
   - The notebook's ResNet18-like backbone has larger stages and ImageNet-style depth assumptions. A CIFAR ResNet-32 main model is more aligned with accuracy/FLOP tradeoffs.

6. A one-pass implementation is likely to produce a notebook that does not run top-to-bottom:
   - The right priority is a runnable core pipeline first, then optional score improvements.

## Revised Core Design

Build a compact, reliable TALC++ core first.

Core models:

- Required: `CifarResNet32`
- Optional in later step: `CifarResNet56`
- Defer unless time remains: `WideResNet16_2`, `RIDE_Lite`

Core Stage 1 candidates:

- `CE_StrongAug`
- `LDAM_DRW`
- `BalancedSoftmax`
- Optional if simple: `LogitAdjustedCE`

Core Stage 2 candidates:

- `none`
- `cRT_no_reset`
- `cRT_reset`
- `cRT_no_reset_LabelAwareSmoothing`
- Optional if simple: `cRT_no_reset_MaxNorm`

Core Stage 3:

- validation-gated calibration with identity/no-calibration included,
- `tau_norm`,
- post-hoc logit adjustment,
- head/medium/tail temperature.

Default final track:

- `FINAL_SUBMISSION_TRACK = "split_calibrated"`

Reason:

- It is cleanest with respect to validation/test separation.
- It avoids choosing full-train vs split-trained after seeing test results.

## Bounded Search Policy

For `SMOKE_TEST=True`:

- scenario: `exp, 0.1`
- model: `resnet32`
- seeds: `[0]`
- Stage 1 epochs: `2`
- Stage 2 epochs: `1`
- BN alignment epochs: `0`
- tiny candidate set

For a realistic full run:

- scenarios: all four required scenarios
- model: start with `resnet32`
- Stage 1 candidates: CE, LDAM-DRW, Balanced Softmax
- choose top 2 Stage 1 checkpoints by validation score
- Stage 2 candidates only on top Stage 1 checkpoints
- calibration only on top Stage 2 checkpoint(s)
- confirm seeds only for the selected config if compute permits

For the hardest scenario `exp-0.01`:

- run extra ablations after the core pipeline works,
- compare `resnet32` vs `resnet56` only if time allows,
- add BCL only after the basic notebook passes smoke tests.

## Step Split

Use these files in order:

1. `01_foundation_prompt.md`
2. `02_models_losses_prompt.md`
3. `03_training_pipeline_prompt.md`
4. `04_orchestration_calibration_prompt.md`
5. `05_results_reporting_prompt.md`
6. `06_optional_boost_prompt.md` only if the core pipeline is already stable.

Shared handoff notes:

- `STEP_NOTES.md`

Each step prompt instructs Codex to read `STEP_NOTES.md` first and append a completion note at the end.

## Overall Acceptance Criteria

Before considering the project implementation complete:

- `Project #3.ipynb` is valid JSON and opens as a notebook.
- The notebook has `SMOKE_TEST` mode.
- Smoke mode can run top-to-bottom or, if local dependencies/data are missing, the failure is explicitly recorded.
- The code preserves the provided imbalanced CIFAR dataset classes.
- The full mode supports all four required scenarios.
- No test split is used for validation, checkpoint selection, seed selection, or calibration.
- `BATCH_SIZE == 128` is asserted.
- Final model training budget per scenario is `<= 200` epochs.
- Final artifacts include `.pth` weights, CSV/JSON summaries, and report notes.
- Params, MACs, FLOPs, overall/macro/head/medium/tail metrics are logged.
- The final test evaluation cell is clearly guarded and labeled as frozen evaluation.
