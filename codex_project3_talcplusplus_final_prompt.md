# Codex Prompt — Final Improved High-Score Design for EEE6503 Project #3

Copy everything below into Codex. The target file is the uploaded notebook `Project #3.ipynb`.

The goal is to maximize the Project #3 score, not to minimize implementation effort. However, the final result must still look like a coherent **proposed long-tail recognition method**, not merely a large hyperparameter search. The notebook should implement the method, run the required experiments, save weights/results, and generate report-ready analysis artifacts.

---

# Prompt for Codex

You are editing `Project #3.ipynb` for **EEE6503 Spring 2026 Project #3: Long-tail Visual Recognition for Image Classification**.

Convert the current baseline notebook into a high-scoring, reproducible long-tail CIFAR-100 experimental pipeline. The final proposed method should be presented as:

> **TALC++: Tail-Aware Logit-Calibrated Decoupled Learning for Long-Tail CIFAR-100**

Important framing:

- TALC++ is **not** just “try many existing methods and pick one.”
- TALC++ is a coherent **three-stage method**:
  1. **Tail-aware representation learning**
  2. **BN-safe decoupled classifier adaptation**
  3. **Validation-gated group-wise calibration**
- Candidate losses/backbones/calibrations are implementation variants and ablations of these three stages.
- In the report, describe TALC++ as a structured method. Use validation-selected variants only as a principled way to adapt the same method to the four different imbalance scenarios.

---

## 0. Non-negotiable assignment constraints

Follow these rules strictly:

1. Use **only the provided imbalanced CIFAR-100 datasets** created by the notebook's `IMBALANCECIFAR100` class.
2. Do **not** use ImageNet pretraining, external data, self-supervised external checkpoints, CLIP, foundation models, external pseudo-labels, or any non-CIFAR training images.
3. Run and report all four long-tail scenarios:
   - `IMB_TYPE='exp'`, `IMB_FACTOR=0.1`
   - `IMB_TYPE='exp'`, `IMB_FACTOR=0.01`
   - `IMB_TYPE='step'`, `IMB_FACTOR=0.1`
   - `IMB_TYPE='step'`, `IMB_FACTOR=0.01`
4. Batch size must remain fixed at **128**.
5. Total training budget per final model per scenario must be `<= 200` epochs. If using multiple stages, the sum must be `<= 200`.
6. Define validation data only from the training split. Use this validation split for checkpoint selection, hyperparameter selection, component selection, seed selection, and calibration.
7. Use the provided test split only for the final frozen evaluation. Do not tune anything after looking at test results.
8. Save `.pth` final model weights per scenario.
9. Report accuracy, number of parameters, and FLOPs/MACs.
10. Include both quantitative and qualitative results.
11. The final notebook must run top-to-bottom and must include `SMOKE_TEST` mode.

---

## 1. Current notebook context

The uploaded notebook already contains:

- Basic imports: `os`, `numpy`, `torch`, `torch.nn`, `torchvision`, `torchvision.transforms`.
- Config variables similar to:
  ```python
  DATASET = 'CIFAR100'
  IMB_TYPE = 'exp'
  IMB_FACTOR = 0.1
  BATCH_SIZE = 128
  EPOCHS = 200
  ```
- Dataset classes:
  - `IMBALANCECIFAR10`
  - `IMBALANCECIFAR100`
- Four target scenarios:
  ```python
  IMB_TYPE in ['exp', 'step']
  IMB_FACTOR in [0.1, 0.01]
  ```
- Baseline CIFAR transforms.
- A ResNet18-like baseline model.
- A baseline CE training loop.
- A simple `compute_accuracy` function.

Refactor the notebook into a reusable experiment system. Preserve the dataset classes, but replace one-off baseline logic with robust training/evaluation functions.

---

## 2. Final method identity: TALC++

Implement TALC++ as a **three-stage long-tail recognition method**.

### 2.1 Stage 1 — Tail-aware representation learning

Goal: learn discriminative features without letting frequent classes dominate.

Main Stage 1 candidates:

1. **LDAM-DRW**: class-dependent margins plus deferred re-weighting.
2. **Balanced Softmax**: distribution-aware loss correcting the mismatch between imbalanced training priors and balanced evaluation.
3. **CE_StrongAug**: strong baseline for ablation.
4. Optional enhanced variant: **Balanced Contrastive Regularization** added to LDAM-DRW or Balanced Softmax.

Do not make BCL mandatory. Treat it as `TALC++-BCL`, an enhanced representation variant selected only if validation improves.

### 2.2 Stage 2 — BN-safe decoupled classifier adaptation

Goal: reduce classifier bias after representation learning.

Freeze the backbone and adapt only the classifier under class-balanced exposure. Critical rule: frozen backbone BatchNorm layers should stay in `eval()` mode by default during classifier retraining.

Main Stage 2 candidates:

1. **cRT_no_reset**: keep Stage 1 classifier initialization and train classifier only. This should be the default candidate.
2. **cRT_reset**: reset classifier and train classifier only.
3. **cRT_no_reset + Label-Aware Smoothing**.
4. **cRT_no_reset + MaxNorm/weight balancing**.
5. Optional: **LWS** as a lightweight classifier-only adaptation.

The report should emphasize the BN-safe decoupling mechanism, not the raw number of candidates.

### 2.3 Stage 3 — Validation-gated group-wise calibration

Goal: correct residual head-class bias while avoiding over-correction.

Calibration must be selected using validation only. Include no-calibration in every grid.

Use a clean and explainable calibration family:

```python
logits_calibrated_c = logits_c / T_group(c) - tau_la * log(prior_c + eps)
```

where:

- `group(c)` is head/medium/tail based on training class frequency.
- `T_group` is a group-wise temperature for head/medium/tail classes.
- `tau_la` is post-hoc logit adjustment strength.
- `tau_la=0` and `T_head=T_medium=T_tail=1` mean no calibration.

Also implement optional classifier tau-normalization:

```python
W'_c = W_c / (||W_c|| ** tau_norm + eps)
```

Critical: if calibration does not improve validation score, select no calibration.

### 2.4 What to call the final method in tables

Use these names consistently:

- `CE`: old baseline.
- `CE_StrongAug`: stronger CE baseline.
- `LDAM_DRW`: Stage 1 only.
- `BS`: Balanced Softmax Stage 1 only.
- `TALC++_no_calib`: Stage 1 + BN-safe decoupled classifier adaptation.
- `TALC++`: full three-stage method with validation-gated calibration.
- `TALC++_BCL`: optional enhanced version if BCL improves validation.
- `RIDE_Lite_TALC++`: optional upper-bound/high-accuracy variant, not the default main method.

---

## 3. Model architectures

Implement multiple model choices, but make the main efficient method `CifarResNet32_TALC++`.

### 3.1 Required common interface

Each model must implement:

```python
forward_features(x) -> Tensor          # [B, D]
forward_classifier(features) -> logits # [B, num_classes]
forward(x, return_features=False)
reset_classifier(num_classes)
freeze_backbone(freeze_bn=True)
unfreeze_all()
get_classifier()
set_classifier(new_classifier)
```

For `forward(x, return_features=True)`, return `(logits, features)`.

### 3.2 Default efficient backbone: `CifarResNet32`

Implement CIFAR-style ResNet-32:

- Initial conv: `3x3`, 16 channels, stride 1, padding 1, no maxpool.
- Three residual stages:
  - stage 1: 16 channels, 5 blocks, stride 1
  - stage 2: 32 channels, 5 blocks, first block stride 2
  - stage 3: 64 channels, 5 blocks, first block stride 2
- Global average pooling.
- Linear classifier.

This is the default main model because the assignment evaluates accuracy, parameters, and FLOPs.

### 3.3 Stronger candidate backbone: `CifarResNet56`

Implement CIFAR-style ResNet-56:

- Same stem/stage channels as ResNet-32.
- 9 residual blocks per stage.
- Use it only if the validation improvement justifies extra FLOPs.

### 3.4 Optional high-accuracy model: `WideResNet16_2`

Implement compact WideResNet-16-2:

- Depth 16, widen factor 2.
- CIFAR stem, no ImageNet-style maxpool.
- Dropout optional, default `0.0` or `0.1`.

### 3.5 Optional upper-bound model: `RIDE_Lite`

Implement only if clean:

- Shared stem and early stages.
- 2 expert heads by default.
- Each expert has its own final stage and classifier.
- Average expert logits at inference.
- Optional diversity loss.
- Count parameters and FLOPs correctly.

Do not automatically choose RIDE-Lite. It is an upper-bound/high-accuracy variant. Select it only if validation gain is large enough to justify efficiency cost. In the report, keep the main method as ResNet32/ResNet56 TALC++ unless RIDE-Lite is clearly dominant.

---

## 4. Data pipeline

### 4.1 Transforms

Create transform builders:

```python
def build_transforms(use_randaugment=True, use_cutout=True, use_random_erasing=False):
    ...
```

Training transform:

```python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip()
# Optional if available:
# transforms.RandAugment(num_ops=2, magnitude=9)
transforms.ToTensor()
transforms.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))
# Optional Cutout or RandomErasing after ToTensor+Normalize
```

Evaluation transform:

```python
transforms.ToTensor()
transforms.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))
```

Implement custom `Cutout` if needed. If `RandAugment` or `RandomErasing` is unavailable, skip gracefully.

### 4.2 Protected stratified validation split

Create validation only from the imbalanced training split.

Implement:

```python
def make_protected_stratified_split(targets, val_ratio=0.1, seed=0,
                                    min_train_per_class=2,
                                    prefer_one_val_for_rare=True):
    """
    For each class:
      - Shuffle class indices with a seeded RNG.
      - If n <= min_train_per_class, put all samples in train and 0 in val.
      - Else allocate val_count = max(1, round(n * val_ratio)) if possible.
      - Ensure train_count >= min_train_per_class whenever possible.
    Return train_indices, val_indices.
    """
```

Implementation details:

- Instantiate the imbalanced CIFAR-100 training dataset twice with the same `rand_number`:
  - one with train transforms,
  - one with eval transforms.
- Assert their targets are identical.
- Use `Subset(train_aug_dataset, train_indices)` for training.
- Use `Subset(train_eval_dataset, val_indices)` for validation.
- Use class counts from the **training subset only** for losses, priors, samplers, and group definitions.
- Never use the CIFAR-100 test set for validation.

### 4.3 Validation coverage logging

Severe imbalance may leave some tail classes with no validation examples. Log coverage explicitly:

```python
val_class_coverage = num_classes_with_val_examples / 100
tail_val_coverage = num_tail_classes_with_val_examples / 33
medium_val_coverage = num_medium_classes_with_val_examples / 34
head_val_coverage = num_head_classes_with_val_examples / 33
```

Save these in every validation summary. If a class is absent from validation, do not include it in validation macro accuracy, but record that it is absent.

### 4.4 Optional final retraining on full training split

Implement two clearly separated final tracks:

#### Track A: split-trained calibrated model

- Train on train subset.
- Select checkpoint/config/calibration on validation subset.
- Final test evaluation uses the calibrated split-trained model.
- This is the most methodologically clean track.

#### Track B: full-train fixed-config retraining

- After all choices are frozen using validation, rebuild the training dataset using all imbalanced training images, including the previous validation images.
- Retrain the selected method with the exact frozen config and schedule.
- Use the calibration parameters selected during validation development or default no-calibration if specified.
- Do not tune anything on test.

Critical rule:

```text
Set FINAL_SUBMISSION_TRACK = 'split_calibrated' or 'full_train_fixed' before running final test.
Do not choose between Track A and Track B after seeing test results.
```

If both tracks are evaluated for analysis, label them separately and do not use test results to change the final method.

---

## 5. Metrics and model selection

Replace the old `compute_accuracy` with a complete evaluator.

```python
def evaluate(model, loader, device, num_classes, cls_num_list=None,
             calibration=None, criterion=None, return_details=False):
    """
    Return dict containing:
      overall_acc
      macro_acc
      head_acc
      medium_acc
      tail_acc
      head50_acc
      tail50_acc
      per_class_acc: np.ndarray [num_classes]
      per_class_count: np.ndarray [num_classes]
      confusion: np.ndarray [num_classes, num_classes]
      avg_loss optional
      ece optional
      val_class_coverage optional
      tail_val_coverage optional
    """
```

Definitions:

- `overall_acc`: sample-level top-1 accuracy.
- `macro_acc`: mean class-wise accuracy over classes present in the evaluated loader.
- Frequency-based groups are based on descending **training-subset** class count:
  - head: top 33 classes
  - medium: middle 34 classes
  - tail: bottom 33 classes
- Also compute `head50_acc` and `tail50_acc` for compatibility with the original notebook assumption that lower class IDs are more frequent.

### 5.1 Main validation selection score

Use validation for checkpoint selection, method selection, calibration selection, and final seed selection.

Use this default score:

```python
selection_score = 0.55 * overall_acc + 0.25 * macro_acc + 0.20 * tail_acc
```

Rationale:

- Overall accuracy is likely the most important single performance metric.
- Macro/tail accuracy protect against head-class dominance and support long-tail analysis.
- This avoids selecting a model that improves tail accuracy while sacrificing too much overall accuracy.

Tie-breakers:

1. Higher overall accuracy.
2. Higher macro accuracy.
3. Higher tail accuracy.
4. If score gap is `< 0.3` percentage points, choose smaller FLOPs.
5. If still tied, choose fewer parameters.
6. If still tied, choose weaker calibration strength.

### 5.2 Seed policy to avoid cherry-picking

Use this policy:

```python
SEARCH_SEEDS = [0]
CONFIRM_SEEDS = [0, 1, 2]
```

Procedure:

1. Run the candidate search with `seed=0`.
2. Select the method/config using validation only.
3. Re-run the selected method with `seeds=[0,1,2]`.
4. Report mean ± std over seeds in `seed_confirm_summary.csv` when feasible.
5. For the final `.pth` submission per scenario, choose the seed with the best validation `selection_score`, not the best test score.
6. Never choose a seed after looking at test accuracy.

---

## 6. Loss functions

Implement all losses below.

### 6.1 Cross entropy baseline

```python
nn.CrossEntropyLoss()
```

### 6.2 LDAM loss

Implement:

```python
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30.0, weight=None):
        ...
    def set_weight(self, weight):
        ...
    def forward(self, logits, target):
        ...
```

Formula:

```python
m_c = 1.0 / (n_c ** 0.25)
m_c = m_c * (max_m / max(m_c))
```

Subtract `m_y` only from the target-class logit, then apply `s * logits` before cross-entropy.

### 6.3 Class-balanced effective-number weights

Implement:

```python
def class_balanced_weights(cls_num_list, beta=0.9999):
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / np.sum(weights) * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32)
```

Use these weights for DRW.

### 6.4 Balanced Softmax loss

Implement:

```python
class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        self.log_counts = torch.log(torch.tensor(cls_num_list, dtype=torch.float32) + 1e-12)
    def forward(self, logits, target):
        return F.cross_entropy(logits + self.log_counts.to(logits.device), target)
```

Also implement a soft-label version for mixup/cutmix:

```python
def balanced_softmax_soft_loss(logits, soft_targets, cls_num_list):
    logits_adj = logits + log_counts
    log_prob = F.log_softmax(logits_adj, dim=1)
    return -(soft_targets * log_prob).sum(dim=1).mean()
```

### 6.5 Logit-adjusted CE as Stage 1 candidate

Implement training-time logit adjustment:

```python
log_prior = torch.log(prior + 1e-12)
loss = F.cross_entropy(logits + tau_train * log_prior, target)
```

Candidate grid:

```python
TAU_TRAIN_GRID = [0.0, 0.5, 1.0, 1.5]
```

This is a Stage 1 candidate, not the same as Stage 3 post-hoc calibration.

### 6.6 Label-aware smoothing loss

Implement class-dependent label smoothing for Stage 2:

```python
epsilon_c = eps_min + (eps_max - eps_min) * normalized_frequency_c
normalized_frequency_c = (n_c - n_min) / (n_max - n_min + 1e-12)
```

Thus head classes receive more smoothing and tail classes receive less smoothing.

Defaults:

```python
eps_min = 0.0
eps_max = 0.2
```

Implement:

```python
class LabelAwareSmoothingCE(nn.Module):
    ...
```

Include `eps_max=0.0` as no-smoothing option.

### 6.7 Optional balanced contrastive regularization

Implement optional BCL for Stage 1 only. It should never be required for TALC++ to run.

Requirements:

1. Use normalized features from `forward_features`.
2. Maintain optional per-class memory queue, e.g. `K=32` features per class.
3. Positives are same-class features from current batch and memory.
4. Denominator should be class-balanced: average negative contributions per class instead of letting head classes dominate.
5. Skip anchors with no positives; return zero if no valid anchors.
6. Combine with classification loss:
   ```python
   total_loss = cls_loss + lambda_bcl * bcl_loss
   ```

Default grid:

```python
LAMBDA_BCL_GRID = [0.0, 0.05, 0.1]
BCL_TEMPERATURE = 0.1
BCL_QUEUE_SIZE = 32
```

In the report, include BCL only if it consistently improves validation or the hardest scenario.

---

## 7. Augmentation, regularization, and optimizer

### 7.1 Mixup/CutMix

Implement optional helpers:

```python
def mixup_data(x, y, alpha):
    ...

def cutmix_data(x, y, alpha):
    ...
```

Rules:

- For LDAM candidates, default to no mixup/cutmix unless a careful soft-label LDAM implementation is added.
- For Balanced Softmax candidates, allow mixup/cutmix using soft-label Balanced Softmax.
- Stage 2 classifier adaptation may use mixup with Label-Aware Smoothing, but include no-mixup as default.

Candidate grid:

```python
MIXUP_ALPHA_GRID = [0.0, 0.2, 0.4]
CUTMIX_ALPHA_GRID = [0.0, 1.0]
```

### 7.2 EMA

Implement Exponential Moving Average:

```python
class ModelEMA:
    def __init__(self, model, decay=0.999):
        ...
    def update(self, model):
        ...
    def copy_to(self, model):
        ...
```

EMA rule:

1. During Stage 1, maintain EMA.
2. At the end of Stage 1, compare raw and EMA checkpoints by validation `selection_score`.
3. Use the better checkpoint as the starting point for Stage 2.
4. Do not maintain EMA during Stage 2 classifier-only adaptation unless explicitly implemented and validated.
5. Save whether raw or EMA was selected.

### 7.3 Optimizer defaults

Use SGD with momentum:

```python
optimizer = torch.optim.SGD(
    params,
    lr=0.1,
    momentum=0.9,
    weight_decay=2e-4,
    nesterov=True,
)
```

Use cosine learning rate with warmup:

```python
warmup_epochs = 5
scheduler = warmup + CosineAnnealingLR
```

Use AMP when CUDA is available:

```python
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
```

---

## 8. Samplers and loaders

Implement:

```python
def make_natural_loader(...):
    # shuffle=True
    ...

def make_balanced_sampler(targets, num_classes):
    class_count = np.bincount(targets, minlength=num_classes)
    sample_weights = np.array([1.0 / max(class_count[y], 1) for y in targets])
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights),
                                 num_samples=len(sample_weights),
                                 replacement=True)

def make_balanced_loader(...):
    # sampler=make_balanced_sampler(...), shuffle=False
    ...
```

Usage:

- Stage 1 representation learning: natural loader by default.
- Stage 1 BCL optional: natural loader plus optional memory queue.
- Stage 2 classifier adaptation: balanced loader.
- BN alignment candidate: balanced loader or natural loader depending on mode.

---

## 9. Training schedule

Every final training run must satisfy:

```python
stage1_epochs + stage2_epochs + bn_align_epochs <= 200
```

Recommended schedule:

```python
STAGE1_EPOCHS = 160
STAGE2_EPOCHS = 35
BN_ALIGN_EPOCHS = 5
TOTAL = 200
DRW_START_EPOCH = 120
WARMUP_EPOCHS = 5
```

Candidate schedules:

```python
SCHEDULE_GRID = [
    dict(stage1_epochs=160, stage2_epochs=35, bn_align_epochs=5, drw_start_epoch=120),
    dict(stage1_epochs=170, stage2_epochs=25, bn_align_epochs=5, drw_start_epoch=130),
    dict(stage1_epochs=150, stage2_epochs=45, bn_align_epochs=5, drw_start_epoch=110),
]
```

Never exceed 200 epochs.

---

## 10. Stage 1 implementation

Implement:

```python
def train_stage1_representation(config, train_loader, val_loader, model, cls_num_list):
    ...
```

Stage 1 candidate modes:

```python
STAGE1_MODES = [
    'CE_StrongAug',
    'LDAM_DRW',
    'BalancedSoftmax',
    'LogitAdjustedCE',
    'LDAM_DRW_BCL',
    'BalancedSoftmax_BCL',
]
```

Behavior:

- Natural sampling by default.
- For `LDAM_DRW`:
  - Epochs `< DRW_START_EPOCH`: LDAM without class-balanced weights.
  - Epochs `>= DRW_START_EPOCH`: LDAM with effective-number class-balanced weights.
- For `BalancedSoftmax`: use Balanced Softmax loss, optionally with mixup/cutmix.
- For `LogitAdjustedCE`: tune `tau_train` by validation.
- For `*_BCL`: add balanced contrastive regularization after warmup. Include `lambda_bcl=0.0` as fallback.
- Evaluate validation every epoch or every fixed interval.
- Track best raw and EMA checkpoints by `selection_score`.
- Save the best Stage 1 checkpoint.

Checkpoint fields:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'ema_state_dict': ema_state_dict_or_none,
    'model_name': model_name,
    'stage1_mode': stage1_mode,
    'cls_num_list': cls_num_list,
    'epoch': epoch,
    'val_metrics': metrics,
    'val_class_coverage': val_class_coverage,
    'tail_val_coverage': tail_val_coverage,
    'config': config,
}, path)
```

---

## 11. Stage 2 implementation: BN-safe decoupled classifier adaptation

Implement:

```python
def train_stage2_classifier(config, model, train_balanced_loader, val_loader, cls_num_list):
    ...
```

### 11.1 Critical BN rule

During classifier adaptation:

- Freeze backbone parameters.
- Put backbone in `eval()` mode by default.
- Keep classifier in `train()` mode.
- Do not update backbone BatchNorm running statistics unless running explicit BN alignment.

Helper:

```python
def set_backbone_eval_classifier_train(model):
    model.eval()
    model.get_classifier().train()
```

Make sure `model.train()` is not accidentally called on the entire model during Stage 2 after freezing BN.

### 11.2 cRT variants

Implement:

1. `cRT_no_reset`: keep Stage 1 classifier initialization, train classifier only.
2. `cRT_reset`: reinitialize classifier, train classifier only.

Defaults:

```python
CRT_LR_GRID = [0.01, 0.05, 0.1]
CRT_WEIGHT_DECAY_GRID = [0.0, 1e-4, 5e-4]
```

### 11.3 LWS candidate

Implement Learnable Weight Scaling:

- Freeze backbone.
- Freeze classifier weight directions.
- Learn positive per-class scale parameters `s_c`.
- Train only scales and optionally bias.

### 11.4 MaxNorm / weight balancing candidate

Implement:

```python
def apply_maxnorm_to_classifier(classifier, radius):
    # If ||w_c|| > radius, project w_c to radius.
```

Candidate radii:

```python
MAXNORM_RADIUS_GRID = [None, 1.0, 2.0, 5.0]
```

Apply after optimizer steps for candidates that enable it.

### 11.5 Stage 2 losses

Candidate losses:

- CE.
- Label-Aware Smoothing CE.
- Balanced Softmax CE.

Do not use LDAM in Stage 2 by default. Stage 2 is classifier boundary correction, not representation learning.

### 11.6 Stage 2 selection

Select best Stage 2 checkpoint by validation `selection_score`. Save:

- Stage 1 source checkpoint.
- Stage 2 mode.
- Reset/no-reset choice.
- LR/weight decay.
- Label smoothing values.
- MaxNorm radius.
- Validation metrics and coverage.

---

## 12. Optional BN alignment stage

Implement BN alignment as a validation-selected candidate, not mandatory.

Candidate modes:

```python
BN_ALIGN_MODES = ['none', 'recompute_balanced_bn', 'train_bn_affine_only']
```

### 12.1 `none`

Default. Keep backbone BN frozen/eval.

### 12.2 `recompute_balanced_bn`

- Freeze all learnable parameters.
- Put only BN layers in train mode to recompute running mean/variance.
- Run balanced training loader for `BN_ALIGN_EPOCHS` without gradients.
- Evaluate on validation.

### 12.3 `train_bn_affine_only`

- Freeze convolution and classifier weights.
- Train only BN affine parameters for a few epochs using small LR.
- Use balanced loader.
- Select only if validation improves.

Save selected BN alignment mode.

---

## 13. Stage 3 implementation: validation-gated calibration

Implement:

```python
def tune_calibration(model, val_loader, cls_num_list, device, groups):
    ...
```

Represent calibration as a dictionary:

```python
calibration = {
    'tau_norm': 0.0,
    'logit_adjust_tau': 0.0,
    'T_head': 1.0,
    'T_medium': 1.0,
    'T_tail': 1.0,
}
```

### 13.1 Tau-normalization

For classifier weights:

```python
W'_c = W_c / (||W_c|| ** tau_norm + eps)
```

Grid:

```python
TAU_NORM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
```

Do not permanently corrupt weights during search. Save/restore classifier state for each candidate.

### 13.2 Group-wise temperature + logit adjustment

For inference-time logits:

```python
logits_calibrated_c = logits_c / T_group(c) - tau_la * log(prior_c + 1e-12)
```

Group temperature grid:

```python
T_HEAD_GRID = [1.0, 1.25, 1.5, 2.0]
T_MEDIUM_GRID = [0.9, 1.0, 1.1, 1.25]
T_TAIL_GRID = [0.75, 1.0, 1.25]
LOGIT_ADJUST_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
```

The no-calibration setting must be present:

```python
T_head = T_medium = T_tail = 1.0
tau_la = 0.0
tau_norm = 0.0
```

### 13.3 Calibration selection rule

Choose calibration by validation `selection_score`.

Tie-breakers:

1. Higher overall accuracy.
2. Higher macro accuracy.
3. Higher tail accuracy.
4. Smaller total calibration strength.

Critical requirement:

```text
If calibration does not improve validation score over the uncalibrated model, select no calibration.
```

Save the full calibration grid table.

---

## 14. Experiment orchestration

Implement clean experiment drivers.

```python
SCENARIOS = [
    ('exp', 0.1),
    ('exp', 0.01),
    ('step', 0.1),
    ('step', 0.01),
]

SEARCH_SEEDS = [0]
CONFIRM_SEEDS = [0, 1, 2]
```

Functions:

```python
def run_single_candidate(config, final_test=False):
    # Build data splits.
    # Build model.
    # Train Stage 1.
    # Train Stage 2 variants.
    # Try BN alignment variants.
    # Tune calibration on validation.
    # Return validation metrics and artifact paths.
```

```python
def run_scenario_model_search(imb_type, imb_factor):
    # Run candidates for this scenario using validation only.
    # Compare by selection_score.
    # Save candidate_pareto.csv.
    # Return selected frozen config.
```

```python
def run_seed_confirmation(selected_config):
    # Re-run selected method with CONFIRM_SEEDS.
    # Report mean/std.
    # Pick final seed by validation score only.
```

```python
def run_final_frozen_evaluation(selected_configs):
    # No tuning here.
    # Evaluate final selected models on test exactly after configs are frozen.
    # Save final_test_summary.csv.
```

### 14.1 Candidate grid with coherent method hierarchy

Do not perform an uncontrolled combinatorial explosion. Use a structured hierarchy:

```python
MODEL_CANDIDATES = ['resnet32', 'resnet56', 'wrn16_2']
OPTIONAL_MODEL_CANDIDATES = ['ride_lite_2expert']

STAGE1_CANDIDATES = [
    dict(stage1_mode='CE_StrongAug'),
    dict(stage1_mode='LDAM_DRW', max_m=0.5, s=30.0, beta=0.9999),
    dict(stage1_mode='BalancedSoftmax', mixup_alpha=0.2, cutmix_alpha=0.0),
    dict(stage1_mode='LogitAdjustedCE', tau_train=1.0),
    dict(stage1_mode='LDAM_DRW_BCL', lambda_bcl=0.1),
    dict(stage1_mode='BalancedSoftmax_BCL', lambda_bcl=0.1, mixup_alpha=0.2),
]

STAGE2_CANDIDATES = [
    dict(stage2_mode='none'),
    dict(stage2_mode='cRT_no_reset', loss='CE'),
    dict(stage2_mode='cRT_reset', loss='CE'),
    dict(stage2_mode='cRT_no_reset', loss='LabelAwareSmoothingCE', eps_max=0.2),
    dict(stage2_mode='cRT_no_reset_MaxNorm', maxnorm_radius=2.0),
    dict(stage2_mode='LWS'),
]
```

Search procedure:

1. For each scenario and model, run Stage 1 candidates with `seed=0`.
2. Select top 2 Stage 1 checkpoints by validation score per model.
3. For each selected Stage 1 checkpoint, run Stage 2 candidates.
4. For top Stage 2 candidates, run BN alignment candidates.
5. For top BN-aligned candidates, run calibration grid.
6. Select final method/config by validation only.
7. Re-run selected method with seeds `[0,1,2]` for confirmation.
8. Choose final seed by validation only.
9. Run final frozen test evaluation.

---

## 15. Baselines and ablations

Implement and save these methods:

1. `CE`: simple CE baseline, natural sampling.
2. `CE_StrongAug`: CE + strong augmentation + cosine LR + EMA.
3. `LDAM_DRW`: Stage 1 LDAM-DRW only.
4. `BalancedSoftmax`: Stage 1 Balanced Softmax only.
5. `TALC++_no_calib`: Stage 1 + BN-safe decoupled classifier adaptation, no calibration.
6. `TALC++`: full method.
7. `TALC++_BCL`: optional if BCL improves validation.
8. `RIDE_Lite_TALC++`: optional upper-bound if implemented.

Minimum across all four scenarios:

- `CE`
- `LDAM_DRW`
- `BalancedSoftmax`
- `TALC++_no_calib`
- `TALC++`

For the hardest scenario `exp-0.01`, run the full ablation:

- backbone comparison,
- Stage 1 loss comparison,
- cRT reset vs no-reset,
- BN-safe vs BN-updating classifier retraining,
- no calibration vs tau-norm vs group-wise calibration,
- BCL optional variant.

All ablation configs must be frozen before final test evaluation. Do not modify configs after seeing test results.

---

## 16. Logging and artifacts

Create this structure:

```text
logs/
  scenario_exp_0.1/
    candidates/
      <candidate_id>/
        config.json
        train_log.csv
        best_stage1.pth
        best_stage2.pth
        best_calibrated.pth
        val_metrics.json
        per_class_val.csv
        confusion_val.npy
    selected/
      selected_config.json
      seed_confirm_summary.csv
      final_model.pth
      test_metrics.json
      per_class_test.csv
      confusion_test.npy
      figures/
        class_distribution.png
        per_class_acc_vs_freq.png
        confusion_tail_to_head.png
        qualitative_tail_examples.png
        feature_pca.png
        calibration_grid_heatmap.png
results/
  validation_candidate_summary.csv
  validation_pareto_summary.csv
  selected_configs.json
  seed_confirm_summary.csv
  final_test_summary.csv
  ablation_summary.csv
  per_class_all_scenarios.csv
  calibration_summary.csv
report_notes.md
FROZEN_RESULTS_DO_NOT_TUNE.txt
```

Checkpoint format:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'model_name': model_name,
    'num_classes': num_classes,
    'imb_type': imb_type,
    'imb_factor': imb_factor,
    'seed': seed,
    'train_cls_num_list': train_cls_num_list,
    'val_cls_num_list': val_cls_num_list,
    'val_class_coverage': val_class_coverage,
    'tail_val_coverage': tail_val_coverage,
    'stage1_mode': stage1_mode,
    'stage2_mode': stage2_mode,
    'bn_align_mode': bn_align_mode,
    'calibration': calibration,
    'use_ema_stage1': use_ema_stage1,
    'stage1_epochs': stage1_epochs,
    'stage2_epochs': stage2_epochs,
    'bn_align_epochs': bn_align_epochs,
    'val_metrics': best_val_metrics,
    'selection_score': selection_score,
    'params_total': params_total,
    'macs': macs,
    'flops': flops,
    'config': config_dict,
}, checkpoint_path)
```

---

## 17. Parameters, MACs, and FLOPs

Implement parameter counts:

```python
def count_parameters_total(model):
    return sum(p.numel() for p in model.parameters())

def count_parameters_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Implement dependency-free MAC/FLOP counter using forward hooks for:

- `nn.Conv2d`
- `nn.Linear`

Formula:

```python
conv_macs = batch * out_channels * out_h * out_w * (in_channels / groups * k_h * k_w)
linear_macs = batch * in_features * out_features
flops = 2 * macs
```

Use input size `(1, 3, 32, 32)`. Save params/MACs/FLOPs in every summary CSV.

If a dependency like `thop` is available, you may cross-check with it, but the notebook must not require it.

---

## 18. Quantitative report outputs

Generate and save these tables.

### 18.1 Final four-scenario table

Columns:

```text
method, model, imb_type, imb_factor, seed,
overall_acc, macro_acc, head_acc, medium_acc, tail_acc, head50_acc, tail50_acc,
params_M, trainable_params_M, MACs_M, FLOPs_M,
stage1_mode, stage2_mode, bn_align_mode,
tau_norm, logit_adjust_tau, T_head, T_medium, T_tail,
use_ema_stage1, final_track
```

### 18.2 Candidate/Pareto table

Columns:

```text
scenario, candidate_id, model, stage1_mode, stage2_mode, bn_align_mode, calibration,
val_overall_acc, val_macro_acc, val_tail_acc, selection_score,
val_class_coverage, tail_val_coverage,
params_M, FLOPs_M, selected
```

### 18.3 Seed confirmation table

Columns:

```text
scenario, method, seed,
val_overall_acc, val_macro_acc, val_tail_acc, selection_score,
test_overall_acc, test_macro_acc, test_tail_acc,
selected_final_seed
```

If test is not evaluated for all seeds, include validation-only seed confirmation and test only the selected final seed.

### 18.4 Ablation table

Columns:

```text
scenario, method, overall_acc, macro_acc, head_acc, medium_acc, tail_acc,
params_M, FLOPs_M, delta_over_CE, delta_tail_over_CE
```

### 18.5 Per-class table

Columns:

```text
scenario, class_id, class_name, train_count, val_count, test_count,
per_class_acc, group, most_confused_as, most_confused_as_group
```

### 18.6 Calibration table

Columns:

```text
scenario, tau_norm, logit_adjust_tau, T_head, T_medium, T_tail,
val_overall_acc, val_macro_acc, val_tail_acc, selection_score,
selected
```

---

## 19. Qualitative and diagnostic outputs

Generate report-ready figures.

### 19.1 Class distribution

Bar plot of training class counts sorted by frequency. Annotate head/medium/tail groups.

### 19.2 Per-class accuracy vs. frequency

Scatter plot:

- x-axis: training class count, log scale.
- y-axis: per-class test accuracy.
- marker or legend: head/medium/tail.

### 19.3 Tail-to-head confusion analysis

From final confusion matrix:

- Identify tail classes most frequently misclassified as head classes.
- Save top-k confusion table.
- Save heatmap of top tail-to-head confusions.

### 19.4 Qualitative tail examples

For several tail classes:

- Show correct examples.
- Show incorrect examples.
- Include true class, predicted class, and confidence.
- Use inverse normalization for visualization.

### 19.5 Feature PCA visualization

Extract features from validation or test after final model selection.

- Use at most 2,000 images.
- Use NumPy SVD PCA fallback; no hard dependency on scikit-learn.
- Color or annotate by head/medium/tail.
- Save as `feature_pca.png`.

### 19.6 Calibration grid heatmap

For each scenario, plot validation score over `tau_norm` and `logit_adjust_tau` or group temperatures for the selected model. This supports the analysis section.

---

## 20. Reproducibility and safeguards

Implement:

```python
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
```

Safeguards:

1. Assert `BATCH_SIZE == 128`.
2. Assert `DATASET == 'CIFAR100'` for final runs.
3. Assert `stage1_epochs + stage2_epochs + bn_align_epochs <= 200`.
4. Assert no pretrained weights are loaded.
5. Assert train/validation indices are disjoint.
6. Assert test loader is not passed into training, checkpoint selection, candidate selection, seed selection, or calibration functions.
7. Include `final_test=False` default in all development functions.
8. Add an explicit final cell named: `FINAL FROZEN TEST EVALUATION — DO NOT RUN UNTIL CONFIGS ARE FIXED`.
9. Before final test, write selected configs to `results/selected_configs.json`.
10. After final test evaluation, write `FROZEN_RESULTS_DO_NOT_TUNE.txt`.
11. Print device, GPU name, PyTorch version, CUDA version, and random seed.

`SMOKE_TEST` mode:

```python
SMOKE_TEST = False
if SMOKE_TEST:
    SCENARIOS = [('exp', 0.1)]
    SEARCH_SEEDS = [0]
    CONFIRM_SEEDS = [0]
    MODEL_CANDIDATES = ['resnet32']
    OPTIONAL_MODEL_CANDIDATES = []
    STAGE1_EPOCHS = 2
    STAGE2_EPOCHS = 1
    BN_ALIGN_EPOCHS = 0
    RUN_FULL_GRID = False
```

---

## 21. Notebook organization

Refactor the notebook into clearly labeled cells:

1. Title, assignment constraints, and TALC++ method overview.
2. Imports and global config.
3. Reproducibility helpers.
4. Dataset classes from the provided notebook.
5. Transforms and protected train/validation split.
6. Model definitions.
7. Loss functions.
8. Mixup/CutMix/Cutout helpers.
9. Samplers and loaders.
10. Metrics and evaluation.
11. Params/FLOPs utilities.
12. Stage 1 training functions.
13. Stage 2 BN-safe classifier adaptation functions.
14. BN alignment functions.
15. Stage 3 calibration functions.
16. Experiment orchestration and candidate search.
17. Seed confirmation runner.
18. Ablation runner.
19. Final frozen test evaluation.
20. Visualization functions.
21. Report summary and `report_notes.md` generation.

---

## 22. Notebook markdown: proposed method description

Add this markdown cell to the notebook and adapt after results are available:

```markdown
## Proposed Method: TALC++

We propose TALC++, a three-stage long-tail recognition method for imbalanced CIFAR-100. TALC++ is designed to reduce head-class bias while preserving overall accuracy and computational efficiency.

First, TALC++ performs tail-aware representation learning. We train the backbone with long-tail-aware objectives such as LDAM-DRW or Balanced Softmax. LDAM imposes larger margins on rare classes, while deferred re-weighting avoids unstable early optimization. Balanced Softmax provides a distribution-aware alternative that compensates for the mismatch between imbalanced training frequencies and balanced evaluation. We also evaluate a balanced contrastive variant as an optional enhancement to improve feature geometry.

Second, TALC++ decouples representation and classifier learning. After the representation stage, the backbone is frozen and the classifier is adapted using class-balanced sampling. Unlike naive classifier retraining, TALC++ keeps frozen BatchNorm layers in evaluation mode during classifier adaptation, preventing the balanced sampler from corrupting learned feature statistics. We compare cRT with and without classifier reinitialization, label-aware smoothing, and weight balancing.

Third, TALC++ uses validation-gated group-wise calibration. For each class, logits are calibrated by a head/medium/tail group-wise temperature and a class-prior logit adjustment. The calibration grid always includes the identity setting, so calibration is applied only when it improves validation performance. This prevents over-correction after long-tail training and classifier adaptation.

For each long-tail scenario, all method choices are selected using a validation split derived from the training split. The test split is used only for final frozen evaluation.
```

---

## 23. Report notes generation

Create `report_notes.md` automatically. Include the following structure:

```markdown
# Report Notes — EEE6503 Project #3

## Introduction
- Modern visual datasets often have long-tailed class distributions.
- Standard cross-entropy training is biased toward frequent classes.
- This project studies imbalanced CIFAR-100 under four scenarios: exp/step × 0.1/0.01.
- Evaluation considers accuracy, parameters, and FLOPs.

## Proposed Method
- TALC++ as a three-stage method:
  1. Tail-aware representation learning.
  2. BN-safe decoupled classifier adaptation.
  3. Validation-gated group-wise calibration.
- Explain why the method balances overall accuracy and tail robustness.
- Mention that candidate variants are selected using validation only.

## Experimental Setup
- Dataset and four imbalance scenarios.
- Protected stratified validation split and validation coverage.
- Backbone candidates and final selected model.
- Training budget: max 200 epochs, batch size 128.
- Test split used only for final frozen evaluation.

## Quantitative Results
- Four-scenario final table.
- Baseline vs TALC++ comparison.
- Head/medium/tail accuracy.
- Macro accuracy.
- Params and FLOPs.
- Ablation table.
- Seed mean ± std when available.

## Qualitative Results
- Class distribution.
- Per-class accuracy vs frequency.
- Tail-to-head confusion analysis.
- Tail class correct/incorrect examples.
- Feature PCA visualization.

## Analysis
- Which component improved overall/macro/tail accuracy most.
- Whether calibration helped or hurt by scenario.
- Why BN-safe classifier retraining matters.
- Tradeoff between accuracy and efficiency.
- Failure cases: tail classes still confused with semantically similar head classes.
- Note that severe imbalance makes tail validation estimates noisy, and report validation coverage.

## Concluding Remarks
- TALC++ improves long-tail robustness while keeping computation controlled.
- Decoupled classifier adaptation and validation-gated calibration reduce head-class bias.
- Limitations: severe imbalance and small tail validation sets.

## References
- Cao et al., Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss, NeurIPS 2019.
- Kang et al., Decoupling Representation and Classifier for Long-Tailed Recognition, ICLR 2020.
- Menon et al., Long-tail Learning via Logit Adjustment, ICLR 2021.
- Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.
- Zhong et al., Improving Calibration for Long-Tailed Recognition, CVPR 2021.
- Alshammari et al., Long-Tailed Recognition via Weight Balancing, CVPR 2022.
- Zhu et al., Balanced Contrastive Learning for Long-Tailed Visual Recognition, CVPR 2022.
- Wang et al., Long-tailed Recognition by Routing Diverse Distribution-Aware Experts, ICLR 2021.
```

---

## 24. Common bugs to avoid

Avoid these mistakes:

- Do not tune anything using test accuracy.
- Do not run calibration on the test set.
- Do not compute priors from test labels.
- Do not compute class counts from train+val unless in explicit full-train fixed-config retraining after all validation choices are frozen.
- Do not exceed 200 epochs per final model.
- Do not change batch size from 128.
- Do not accidentally use CIFAR-10 for final results.
- Do not load pretrained weights.
- Do not let validation dataset use random train augmentations.
- Do not let cRT update frozen backbone BatchNorm statistics unless running explicit BN alignment.
- Do not force classifier reinitialization; evaluate reset and no-reset variants by validation.
- Do not force calibration; select no calibration if it wins.
- Do not report only overall accuracy; include macro and head/medium/tail.
- Do not rely on `thop`, `fvcore`, or `sklearn` without fallback code.
- Do not overwrite best checkpoints without saving config and metrics.
- Do not choose a higher-FLOP model if validation score is statistically tied with a smaller model.
- Do not choose Track A vs Track B after seeing test results.
- Do not choose the final seed by test accuracy.

---

## 25. Acceptance checklist

Before finishing, verify:

- [ ] Notebook runs in `SMOKE_TEST=True` mode.
- [ ] Full mode supports all four scenarios.
- [ ] `BATCH_SIZE == 128` is asserted.
- [ ] All final runs satisfy total epochs `<= 200`.
- [ ] Validation split is from training split only.
- [ ] Validation coverage is logged.
- [ ] Test evaluation is isolated in the final frozen evaluation cell.
- [ ] Stage 1 supports CE_StrongAug, LDAM-DRW, Balanced Softmax, Logit-Adjusted CE, and optional BCL variants.
- [ ] Stage 2 supports cRT no-reset, cRT reset, Label-Aware Smoothing, MaxNorm, and optional LWS.
- [ ] cRT freezes backbone and keeps frozen BN layers in eval mode by default.
- [ ] Calibration grid includes identity/no-calibration options.
- [ ] Model selection uses validation only.
- [ ] Seed selection uses validation only.
- [ ] Selected configs are saved before final test.
- [ ] Final test summary includes all four scenarios.
- [ ] Summary CSV includes overall, macro, head, medium, tail, params, MACs, FLOPs.
- [ ] Ablation table is saved.
- [ ] Per-class table is saved.
- [ ] Qualitative figures are saved.
- [ ] `report_notes.md` is generated.
- [ ] `.pth` final weights are saved per scenario.
- [ ] `FROZEN_RESULTS_DO_NOT_TUNE.txt` is created after final evaluation.

Implement the notebook now.
