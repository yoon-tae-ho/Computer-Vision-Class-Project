# Report Notes - EEE6503 Project #3

## Introduction

- Modern visual datasets often have long-tailed class distributions.
- Standard cross-entropy training is biased toward frequent classes.
- This project studies imbalanced CIFAR-100 under four scenarios: `exp/0.1`, `exp/0.01`, `step/0.1`, and `step/0.01`.
- Evaluation considers accuracy, macro/group accuracy, parameters, MACs, and FLOPs.

## Proposed Method

TALC++ is a three-stage long-tail recognition method.

1. Tail-aware representation learning: train a CIFAR backbone with CE strong augmentation, LDAM-DRW, Balanced Softmax, or logit-adjusted CE.
2. BN-safe decoupled classifier adaptation: freeze the learned backbone, keep frozen BatchNorm layers in eval mode, and retrain only the classifier with class-balanced exposure.
3. Validation-gated group-wise calibration: tune group temperatures, post-hoc logit adjustment, and tau-normalization on validation only. The identity setting is always included, so calibration is used only when it improves validation selection score.

## Experimental Setup

- Dataset: CIFAR-100 with the provided `IMBALANCECIFAR100` class.
- Validation split: protected stratified split from the imbalanced training split only.
- Default model: `CifarResNet32`.
- Training budget: `stage1 + stage2 + bn_align <= 200` epochs; batch size fixed at 128.
- Test split: used only by the guarded final frozen evaluation cell after selected configs are fixed.
- Final track: `split_calibrated`.

## Quantitative Results

Final test summary file: `results/final_test_summary.csv`.

| method | model | imb_type | imb_factor | overall_acc | macro_acc | tail_acc | FLOPs_M |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TALC++ | resnet32 | exp | 0.01 | 44.4 | 44.4 | 31.515151515151516 | 138.260992 |
| TALC++ | resnet32 | exp | 0.1 | 60.37 | 60.37 | 63.45454545454545 | 138.260992 |
| TALC++ | resnet32 | step | 0.01 | 46.05 | 46.05 | 23.12121212121212 | 138.260992 |
| TALC++ | resnet32 | step | 0.1 | 61.77 | 61.77 | 52.03030303030303 | 138.260992 |

Ablation summary file: `results/ablation_summary.csv`.

| scenario | method | overall_acc | macro_acc | tail_acc | delta_over_CE | delta_tail_over_CE |
| --- | --- | --- | --- | --- | --- | --- |
| scenario_exp_0p01 | TALC++ | 44.4 | 44.4 | 31.515151515151516 |  |  |
| scenario_exp_0p1 | TALC++ | 60.37 | 60.37 | 63.45454545454545 |  |  |
| scenario_step_0p01 | TALC++ | 46.05 | 46.05 | 23.12121212121212 |  |  |
| scenario_step_0p1 | TALC++ | 61.77 | 61.77 | 52.03030303030303 | 0.0 | 0.0 |

Seed confirmation file: `results/seed_confirm_summary.csv`.

- Rows available: 0
- Final seed should be selected by validation score only, never test score.

Calibration summary file: `results/calibration_summary.csv`.

- Rows available: 69120
- Identity/no-calibration is included in every calibration grid.

## Qualitative Results

Expected figure files under each `logs/scenario_<type>_<factor>/selected/figures/` directory:

- `class_distribution.png`
- `per_class_acc_vs_freq.png`
- `confusion_tail_to_head.png`
- `qualitative_tail_examples.png`
- `feature_pca.png`
- `calibration_grid_heatmap.png`

If a figure cannot be generated in the current environment, the notebook writes a `.txt` placeholder next to the target image path.

## Analysis

- Compare CE/CE strong augmentation against LDAM-DRW and Balanced Softmax to identify the strongest representation learner.
- Compare Stage 1 only against `TALC++_no_calib` to isolate BN-safe classifier adaptation.
- Compare `TALC++_no_calib` against `TALC++` to isolate validation-gated calibration.
- Inspect macro and tail accuracy together with overall accuracy to avoid over-correcting toward tail classes at a large overall cost.
- Use validation coverage when interpreting the hardest `exp/0.01` scenario because tail validation estimates can be noisy.

## Concluding Remarks

TALC++ is designed to improve long-tail robustness while keeping computation controlled. Its main safety mechanisms are validation-only model selection, BN-safe classifier adaptation, and calibration that is applied only when validation improves.

## Artifact Checklist

{
  "final_test_summary": true,
  "validation_candidate_summary": true,
  "validation_pareto_summary": true,
  "seed_confirm_summary": true,
  "ablation_summary": true,
  "per_class_all_scenarios": true,
  "calibration_summary": true,
  "selected_configs": true
}

## References

- Cao et al., Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss, NeurIPS 2019.
- Kang et al., Decoupling Representation and Classifier for Long-Tailed Recognition, ICLR 2020.
- Menon et al., Long-tail Learning via Logit Adjustment, ICLR 2021.
- Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.
- Zhong et al., Improving Calibration for Long-Tailed Recognition, CVPR 2021.
- Alshammari et al., Long-Tailed Recognition via Weight Balancing, CVPR 2022.
