# Report Notes - EEE6503 Project #3

## Introduction

- Modern visual datasets often have long-tailed class distributions.
- Standard cross-entropy training is biased toward frequent classes.
- This project studies imbalanced CIFAR-100 under four scenarios: `exp/0.1`, `exp/0.01`, `step/0.1`, and `step/0.01`.
- Evaluation considers accuracy, macro/group accuracy, parameters, MACs, and FLOPs.

## Proposed Method

TALC++ is a three-stage long-tail recognition method.

1. Tail-aware representation learning with CE strong augmentation, LDAM-DRW, Balanced Softmax, or logit-adjusted CE.
2. BN-safe decoupled classifier adaptation by freezing the backbone, keeping frozen BatchNorm layers in eval mode, and retraining only the classifier with class-balanced exposure.
3. Validation-gated group-wise calibration with identity/no-calibration included in every grid.

## Experimental Setup

- Dataset: CIFAR-100 with the provided `IMBALANCECIFAR100` class.
- Validation split: protected stratified split from the imbalanced training split only.
- Default model: `CifarResNet32`.
- Training budget: `stage1 + stage2 + bn_align <= 200` epochs; batch size fixed at 128.
- Test split: used only by the guarded final frozen evaluation cell after selected configs are fixed.
- Final track: `split_calibrated`.

## Quantitative Results

Fill this section after running the experiment pipeline and `prepare_report_artifacts()`.

Expected files:

- `results/final_test_summary.csv`
- `results/validation_candidate_summary.csv`
- `results/validation_pareto_summary.csv`
- `results/seed_confirm_summary.csv`
- `results/ablation_summary.csv`
- `results/per_class_all_scenarios.csv`
- `results/calibration_summary.csv`

## Qualitative Results

Expected figure files under each `logs/scenario_<type>_<factor>/selected/figures/` directory:

- `class_distribution.png`
- `per_class_acc_vs_freq.png`
- `confusion_tail_to_head.png`
- `qualitative_tail_examples.png`
- `feature_pca.png`
- `calibration_grid_heatmap.png`

## Analysis

- Compare CE/CE strong augmentation against LDAM-DRW and Balanced Softmax.
- Compare Stage 1 only against `TALC++_no_calib`.
- Compare `TALC++_no_calib` against `TALC++`.
- Inspect macro and tail accuracy together with overall accuracy.
- Report validation coverage, especially for `exp/0.01`.

## Concluding Remarks

TALC++ is designed to improve long-tail robustness while keeping computation controlled. Its main safety mechanisms are validation-only model selection, BN-safe classifier adaptation, and validation-gated calibration.

## References

- Cao et al., Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss, NeurIPS 2019.
- Kang et al., Decoupling Representation and Classifier for Long-Tailed Recognition, ICLR 2020.
- Menon et al., Long-tail Learning via Logit Adjustment, ICLR 2021.
- Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.
- Zhong et al., Improving Calibration for Long-Tailed Recognition, CVPR 2021.
- Alshammari et al., Long-Tailed Recognition via Weight Balancing, CVPR 2022.
