# Step 3 Prompt: Stage 1 and Stage 2 Training Pipeline

You are editing `Project #3.ipynb`.

Before editing:

1. Read `talcplusplus_step_prompts/STEP_NOTES.md`.
2. Read `talcplusplus_step_prompts/00_design_review_and_execution_plan.md`.
3. Inspect the notebook after Steps 1 and 2.

Goal for this step:

Implement the core TALC++ training pipeline: Stage 1 tail-aware representation learning and Stage 2 BN-safe decoupled classifier adaptation.

Core scope:

- Implement reliable CE, LDAM-DRW, and Balanced Softmax training.
- Implement BN-safe cRT classifier adaptation.
- Save checkpoints and logs.
- Keep optional score boosts out of this step unless they are already trivial from existing helpers.

Required Stage 1 function:

```python
def train_stage1_representation(config, train_loader, val_loader, model, cls_num_list, device):
    ...
```

Stage 1 modes:

- `CE_StrongAug`
- `LDAM_DRW`
- `BalancedSoftmax`
- optional if already easy: `LogitAdjustedCE`

Stage 1 requirements:

- Natural sampling by default.
- SGD with momentum, weight decay, Nesterov.
- Warmup + cosine schedule.
- AMP when CUDA is available.
- Validate at least every epoch in smoke mode and at a configurable interval in full mode.
- Track best checkpoint by validation `selection_score`.
- Save:
  - model state,
  - config,
  - class counts,
  - epoch,
  - validation metrics,
  - params/MACs/FLOPs.

LDAM-DRW requirements:

- Before `DRW_START_EPOCH`, use LDAM without class-balanced weights.
- At and after `DRW_START_EPOCH`, use effective-number class-balanced weights.

Balanced Softmax requirements:

- Use `BalancedSoftmaxLoss`.
- If Mixup/CutMix helpers already exist, support them only when `alpha > 0`.
- Keep no-mixup as the default.

EMA:

- Add a simple `ModelEMA` only if it can be implemented cleanly.
- If added, compare raw vs EMA by validation and record which was selected.
- If not added, record in notes that EMA is deferred.

Required Stage 2 function:

```python
def train_stage2_classifier(config, model, train_balanced_loader, val_loader, cls_num_list, device):
    ...
```

Stage 2 modes:

- `none`
- `cRT_no_reset`
- `cRT_reset`
- `cRT_no_reset_LabelAwareSmoothing`
- optional if already implemented: `cRT_no_reset_MaxNorm`

BN-safe rule:

- Freeze the backbone.
- Keep frozen BatchNorm layers in eval mode during classifier adaptation.
- Keep only the classifier trainable.
- Do not call `model.train()` in a way that reactivates all frozen BN layers.
- Use `set_backbone_eval_classifier_train(model)`.

Stage 2 losses:

- CE
- Label-aware smoothing CE
- Optional: Balanced Softmax CE
- Do not use LDAM in Stage 2 by default.

Checkpoint/log requirements:

- Save best Stage 1 checkpoint.
- Save best Stage 2 checkpoint.
- Save per-epoch CSV logs.
- Save validation metrics JSON.
- Include enough config information to reproduce the run.

Training-budget safeguards:

- Assert `stage1_epochs + stage2_epochs + bn_align_epochs <= 200`.
- In `SMOKE_TEST`, keep the run tiny.

Verification:

- Validate notebook JSON.
- Run random-tensor checks from Step 2 if practical.
- If CIFAR data is present or download is allowed in the user's environment, run `SMOKE_TEST=True` for a single tiny scenario and verify that:
  - Stage 1 completes,
  - Stage 2 completes,
  - checkpoints/logs are written.
- If data or environment prevents running smoke mode, record the exact blocker in `STEP_NOTES.md`.

Scope limits:

- Do not implement full scenario orchestration yet.
- Do not implement final frozen test evaluation yet.
- Do not implement full calibration grid yet.
- Do not add BCL or RIDE-Lite in this step.

Before finishing:

1. Append a note to `talcplusplus_step_prompts/STEP_NOTES.md` under "Step 3 Notes: Training Pipeline".
2. Include changed files, verification commands/results, blockers, and the next recommended prompt:
   `talcplusplus_step_prompts/04_orchestration_calibration_prompt.md`
