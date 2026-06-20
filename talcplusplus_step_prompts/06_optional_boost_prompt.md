# Optional Step 6 Prompt: Score Boosts After Core Stability

Use this prompt only after Steps 1-5 are complete and `SMOKE_TEST=True` works.

You are editing `Project #3.ipynb`.

Before editing:

1. Read `talcplusplus_step_prompts/STEP_NOTES.md`.
2. Read `talcplusplus_step_prompts/00_design_review_and_execution_plan.md`.
3. Confirm that the core TALC++ pipeline is already stable.

Goal for this optional step:

Add carefully bounded score improvements without destabilizing the notebook.

Allowed optional additions, in priority order:

1. `CifarResNet56` ablation for the hardest scenario `exp-0.01`.
2. EMA during Stage 1 if not already implemented.
3. MaxNorm classifier projection in Stage 2 if not already implemented.
4. BN alignment candidate:
   - `none`
   - `recompute_balanced_bn`
   - optional `train_bn_affine_only`
5. `TALC++_BCL` as an optional Stage 1 variant.
6. `WideResNet16_2` only if clean.

Avoid unless there is substantial time:

- `RIDE_Lite`
- large combinatorial grids,
- extensive Mixup/CutMix tuning,
- adding new dependencies.

BCL constraints if implemented:

- It must be optional.
- It must not be required for TALC++ to run.
- It must use normalized features.
- It must handle anchors with no positives.
- It must return zero if no valid anchors exist.
- It must be included only if validation improves.

BN alignment constraints if implemented:

- Treat it as validation-selected.
- Keep `none` as the default candidate.
- Never update BN statistics during ordinary cRT unless this explicit BN alignment step is selected.

Verification:

- Validate notebook JSON.
- Re-run `SMOKE_TEST=True`.
- Verify that optional features can be disabled.
- Verify that the core TALC++ path still works unchanged.

Before finishing:

1. Append a note to `talcplusplus_step_prompts/STEP_NOTES.md` under "Optional Step 6 Notes: Score Boosts".
2. Include changed files, verification commands/results, blockers, and final handoff.
