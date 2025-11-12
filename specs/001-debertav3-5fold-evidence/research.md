# Research Notes — 5-Fold DeBERTaV3 Evidence Binding

## Topics & Decisions

- CV splitting with group stratification:
  - Prefer scikit-learn's StratifiedGroupKFold if available (version-dependent).
  - Otherwise: use a third-party iterative stratification implementation or
    approximate with GroupKFold plus post-hoc balancing checks.
  - Groups: `post_id`; target: binary label.
- Optimizer:
  - Use HF Trainer with `optim=adamw_torch_fused`; fallback to `adamw_torch` if
    fused unsupported on platform/PyTorch build.
- Fine-tuning:
  - RESOLVED — Full fine‑tune of all Transformer layers (no backbone freezing
    by default). Optionally expose config to freeze for low‑VRAM scenarios.
- Loss & Imbalance:
  - Default weighted cross-entropy (inverse-frequency weights per fold).
  - Optional Focal Loss (γ=2.0; α derived from class frequencies) via Hydra flag.
- Metrics:
  - Accuracy, macro-F1, positive-class F1, ROC-AUC, PR-AUC.
  - Log per-fold; aggregate mean/std in parent MLflow run.
  - Selection: Macro-F1 is the primary model selection metric (resolved).
- Precision:
  - RESOLVED — Prefer BF16; fall back to FP16; else FP32. Use HF Trainer
    `bf16`/`fp16` flags and log selected mode.
 - Scheduler:
   - RESOLVED — Linear LR schedule with warmup_ratio=0.06 via TrainingArguments.
 - HPO:
   - RESOLVED — Not in scope for this feature (fixed hyperparameters). If added
     later, use Optuna with `sqlite:///optuna.db` and MLflow logging.
- NSP-style inputs:
  - Use HF tokenizers with sentence-pair encoding: `encode_plus(text=criterion,
    text_pair=sentence, truncation=True, padding)`. Preserve max_length and
    special tokens per model.
- Reproducibility:
  - Log Hydra config, seeds, `pip freeze`, git SHA, data manifest (filenames +
    checksums). Set deterministic flags where feasible.
 - Negative sampling:
   - RESOLVED — Stratified random negatives to achieve a 1:3 pos:neg ratio at
     dataset construction; maintain grouping by `post_id` for CV.

## Open Items

- Library choice for group-stratified CV if scikit-learn version lacks it.
- Tokenization decision: RESOLVED — max_length=512; truncation=longest_first.
