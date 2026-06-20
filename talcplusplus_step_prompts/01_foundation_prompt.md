# Step 1 Prompt: Notebook Foundation

You are editing `Project #3.ipynb`.

Before editing:

1. Read `talcplusplus_step_prompts/STEP_NOTES.md`.
2. Read `talcplusplus_step_prompts/00_design_review_and_execution_plan.md`.
3. Inspect the current notebook structure.

Goal for this step:

Build the reliable foundation for the TALC++ notebook rewrite. Do not implement the full training system yet. This step should make the notebook organized, valid, and ready for models/losses/training to be added in later steps.

Required edits:

1. Reorganize the notebook into clearly labeled cells:
   - title and assignment constraints,
   - imports and global config,
   - reproducibility helpers,
   - dataset classes,
   - transforms,
   - protected train/validation split,
   - data loaders and samplers,
   - metrics/evaluation helpers,
   - params/MACs/FLOPs helpers,
   - artifact/logging helpers,
   - placeholder cells for models, losses, training, orchestration, final frozen evaluation, and reporting.

2. Preserve the provided dataset classes:
   - `IMBALANCECIFAR10`
   - `IMBALANCECIFAR100`

3. Add global config with at least:
   - `DATASET = "CIFAR100"`
   - `BATCH_SIZE = 128`
   - `SMOKE_TEST`
   - `SCENARIOS`
   - `SEARCH_SEEDS`
   - `CONFIRM_SEEDS`
   - `FINAL_SUBMISSION_TRACK = "split_calibrated"`
   - `RUN_FINAL_TEST = False`
   - `DEVICE`
   - epoch settings for full mode and smoke mode.

4. Add safeguards:
   - assert `BATCH_SIZE == 128`,
   - assert total final epochs `<= 200`,
   - assert final mode uses CIFAR-100,
   - keep final test evaluation guarded behind `RUN_FINAL_TEST`.

5. Implement reproducibility:
   - `set_seed(seed)`,
   - deterministic random split behavior,
   - print device, PyTorch version, CUDA version, and seed.

6. Implement transform builders:
   - `build_transforms(use_randaugment=True, use_cutout=True, use_random_erasing=False)`,
   - custom `Cutout`,
   - graceful fallback if `RandAugment` or `RandomErasing` is unavailable.

7. Implement protected validation split:
   - `make_protected_stratified_split(...)`,
   - derive validation only from the imbalanced training split,
   - keep train and validation indices disjoint,
   - ensure validation uses eval transforms, not random train transforms,
   - compute class counts from the training subset only.

8. Implement loader helpers:
   - `make_natural_loader(...)`,
   - `make_balanced_sampler(...)`,
   - `make_balanced_loader(...)`.

9. Implement evaluation foundations:
   - class grouping by training-subset frequency: head 33, medium 34, tail 33,
   - `selection_score = 0.55 * overall_acc + 0.25 * macro_acc + 0.20 * tail_acc`,
   - `evaluate(...)` returning overall, macro, head, medium, tail, head50, tail50, per-class accuracy/counts, confusion, optional loss, and validation coverage.

10. Implement params/MACs/FLOPs helpers:
    - dependency-free forward hooks for `nn.Conv2d` and `nn.Linear`,
    - input size `(1, 3, 32, 32)`,
    - report MACs and FLOPs.

11. Implement artifact helpers:
    - safe JSON/CSV writing,
    - `logs/scenario_<type>_<factor>/...` path builders,
    - `results/` directory setup.

Important scope limits:

- Do not implement Stage 1/Stage 2 training loops yet.
- Do not implement BCL, RIDE-Lite, or full candidate search in this step.
- Keep placeholder cells explicit so the next step has obvious insertion points.

Verification:

- Validate that `Project #3.ipynb` is valid JSON.
- If `nbformat` is available, run notebook validation.
- Run lightweight syntax checks by extracting code cells to a temporary Python file if practical.
- Do not require full CIFAR download for this step if data is not already present.

Before finishing:

1. Append a note to `talcplusplus_step_prompts/STEP_NOTES.md` under "Step 1 Notes: Foundation".
2. Include changed files, verification commands/results, blockers, and the next recommended prompt:
   `talcplusplus_step_prompts/02_models_losses_prompt.md`
