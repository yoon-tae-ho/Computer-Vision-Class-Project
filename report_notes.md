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

_No rows available yet._

Ablation summary file: `results/ablation_summary.csv`.

| scenario | method | overall_acc | macro_acc | tail_acc | delta_over_CE | delta_tail_over_CE |
| --- | --- | --- | --- | --- | --- | --- |
| scenario_exp_0p1 | CE_StrongAug | 65.71428571428571 | 58.83373673859627 | 45.46536796536797 | 0.0 | 0.0 |
| scenario_exp_0p1 | LDAM_DRW | 2.5510204081632653 | 1.0 | 0.0 | -63.16326530612244 | -45.46536796536797 |
| scenario_exp_0p1 | BalancedSoftmax | 63.52040816326531 | 59.66933190165746 | 52.142857142857146 | -2.1938775510203996 | 6.6774891774891785 |
| scenario_exp_0p1 | BalancedSoftmax | 60.96938775510204 | 59.65168608831531 | 63.60149110149111 | -4.7448979591836675 | 18.136123136123146 |
| scenario_exp_0p1 | TALC++ | 62.80612244897959 | 58.678756144795834 | 56.5981240981241 | -2.908163265306115 | 11.132756132756136 |
| scenario_exp_0p1 | TALC++ | 62.80612244897959 | 58.678756144795834 | 56.5981240981241 | -2.908163265306115 | 11.132756132756136 |
| scenario_exp_0p1 | TALC++ | 62.704081632653065 | 58.57228722654015 | 55.588023088023085 | -3.0102040816326436 | 10.122655122655118 |
| scenario_exp_0p1 | TALC++ | 62.80612244897959 | 59.10531143107449 | 56.15680615680616 | -2.908163265306115 | 10.691438191438195 |

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
  "final_test_summary": false,
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
