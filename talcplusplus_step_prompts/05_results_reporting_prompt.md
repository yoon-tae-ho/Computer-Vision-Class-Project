# Step 5 Prompt: Results, Figures, Report Notes, and Final Smoke Verification

You are editing `Project #3.ipynb`.

Before editing:

1. Read `talcplusplus_step_prompts/STEP_NOTES.md`.
2. Read `talcplusplus_step_prompts/00_design_review_and_execution_plan.md`.
3. Inspect the notebook after Steps 1-4.

Goal for this step:

Finish the core TALC++ notebook by adding report-ready summaries, figures, final artifact generation, and a final smoke verification path.

Required quantitative outputs:

Generate and save these tables when the relevant experiment data exists:

- `results/final_test_summary.csv`
- `results/ablation_summary.csv`
- `results/per_class_all_scenarios.csv`
- `results/seed_confirm_summary.csv`
- `results/calibration_summary.csv`
- scenario candidate summaries under `logs/`

Final table columns should include at least:

```text
method, model, imb_type, imb_factor, seed,
overall_acc, macro_acc, head_acc, medium_acc, tail_acc, head50_acc, tail50_acc,
params_M, trainable_params_M, MACs_M, FLOPs_M,
stage1_mode, stage2_mode, bn_align_mode,
tau_norm, logit_adjust_tau, T_head, T_medium, T_tail,
use_ema_stage1, final_track
```

Required qualitative/diagnostic figures:

Implement figure functions with graceful fallback if matplotlib is unavailable:

1. `class_distribution.png`
2. `per_class_acc_vs_freq.png`
3. `confusion_tail_to_head.png`
4. `qualitative_tail_examples.png`
5. `feature_pca.png`
6. `calibration_grid_heatmap.png`

Figure requirements:

- Use inverse normalization for qualitative images.
- Feature PCA must have a NumPy SVD fallback and no hard dependency on scikit-learn.
- Tail-to-head confusion should also save a small CSV table.

Required report notes:

Automatically create `report_notes.md` with these sections:

- Introduction
- Proposed Method
- Experimental Setup
- Quantitative Results
- Qualitative Results
- Analysis
- Concluding Remarks
- References

The notes should describe TALC++ as:

1. tail-aware representation learning,
2. BN-safe decoupled classifier adaptation,
3. validation-gated group-wise calibration.

Final weights:

- Ensure final selected `.pth` weights are saved per scenario under `logs/scenario_<type>_<factor>/selected/final_model.pth`.
- Checkpoint metadata should include config, validation metrics, calibration, params/MACs/FLOPs, class counts, and selected seed.

Notebook polish:

- Add clear markdown explanations for TALC++ and the assignment safeguards.
- Ensure the notebook reads as a coherent proposed method, not just a search script.
- Keep the final frozen test cell guarded.
- Make `SMOKE_TEST` mode obvious and easy to switch.

Verification:

- Validate notebook JSON.
- Run random-tensor model/loss/FLOP checks if available.
- Run `SMOKE_TEST=True` top-to-bottom if local data/environment permits.
- If full notebook execution is not possible locally, record the exact missing dependency, data, or runtime blocker.
- Confirm that no code path uses test data for tuning.

Before finishing:

1. Append a note to `talcplusplus_step_prompts/STEP_NOTES.md` under "Step 5 Notes: Results and Reporting".
2. Include changed files, verification commands/results, blockers, and final handoff.
3. If the core pipeline is stable and time remains, recommend:
   `talcplusplus_step_prompts/06_optional_boost_prompt.md`
