# Step 2 Prompt: Models and Losses

You are editing `Project #3.ipynb`.

Before editing:

1. Read `talcplusplus_step_prompts/STEP_NOTES.md`.
2. Read `talcplusplus_step_prompts/00_design_review_and_execution_plan.md`.
3. Inspect the notebook after Step 1.

Goal for this step:

Add the model architectures, common model interface, losses, and small random-tensor checks needed by TALC++.

Required model interface:

Every model used by the experiment system must implement:

```python
forward_features(x)
forward_classifier(features)
forward(x, return_features=False)
reset_classifier(num_classes)
freeze_backbone(freeze_bn=True)
unfreeze_all()
get_classifier()
set_classifier(new_classifier)
```

For `forward(x, return_features=True)`, return `(logits, features)`.

Required models:

1. `CifarResNet32`
   - CIFAR stem: 3x3 conv, 16 channels, stride 1, no maxpool.
   - Stages: 16/32/64 channels.
   - 5 residual blocks per stage.
   - Downsample at the first block of stages 2 and 3.
   - Global average pooling.
   - Linear classifier.
   - This is the main TALC++ model.

2. `CifarResNet56`
   - Same design as ResNet32 but 9 blocks per stage.
   - Keep it implemented but do not make it the default full-grid model.

3. Optional only if clean and quick:
   - `WideResNet16_2`
   - Do not add `RIDE_Lite` in this step.

4. Add a model registry:
   - `build_model(model_name, num_classes)`
   - supported names at minimum: `"resnet32"`, `"resnet56"`.

Required losses and helpers:

1. Cross entropy baseline.

2. `LDAMLoss`
   - class-dependent margins,
   - `set_weight(...)`,
   - scale logits with `s`.

3. `class_balanced_weights(cls_num_list, beta=0.9999)`.

4. `BalancedSoftmaxLoss`.

5. `balanced_softmax_soft_loss(...)` for later Mixup/CutMix support.

6. Training-time logit-adjusted CE helper:
   - `logits + tau_train * log_prior`.

7. `LabelAwareSmoothingCE`
   - head classes receive more smoothing,
   - tail classes receive less smoothing,
   - support `eps_max=0.0` as no smoothing.

8. Optional, only if simple:
   - `mixup_data(...)`,
   - `cutmix_data(...)`.

9. Classifier adaptation helpers:
   - `clone_classifier_state(...)`,
   - `apply_maxnorm_to_classifier(...)`,
   - `set_backbone_eval_classifier_train(model)`.

10. Calibration helper foundations:
    - group index tensors,
    - classifier tau-normalization utility that can apply and restore state safely,
    - logit calibration function:

```python
logits_calibrated_c = logits_c / T_group(c) - tau_la * log(prior_c + eps)
```

Scope limits:

- Do not implement the full calibration search yet.
- Do not implement Stage 1 or Stage 2 training loops yet.
- Do not implement BCL memory queues yet.
- Do not implement RIDE-Lite yet.

Verification:

- Validate notebook JSON.
- Run a random-tensor smoke check if local imports work:
  - build `resnet32`,
  - pass random `(2, 3, 32, 32)` input,
  - verify logits shape `(2, 100)`,
  - verify `return_features=True`,
  - verify params/MACs/FLOPs helper works on the model.
- Instantiate each required loss with a small fake `cls_num_list` and fake logits/targets.

Before finishing:

1. Append a note to `talcplusplus_step_prompts/STEP_NOTES.md` under "Step 2 Notes: Models and Losses".
2. Include changed files, verification commands/results, blockers, and the next recommended prompt:
   `talcplusplus_step_prompts/03_training_pipeline_prompt.md`
