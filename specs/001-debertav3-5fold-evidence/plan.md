# Implementation Plan: 5-Fold DeBERTaV3-base Evidence Binding (NSP-Style)

**Branch**: `001-debertav3-5fold-evidence` | **Date**: 2025-11-12 | **Spec**: spec.md
**Input**: Feature specification from `/specs/001-debertav3-5fold-evidence/spec.md`

## Summary

Train and evaluate a binary classifier using `microsoft/deberta-v3-base` with a
Hugging Face sequence classification head on NSP-style criterion–sentence pairs
(`[CLS] <criterion> [SEP] <sentence> [SEP]`). Use 5-fold CV with
GroupStratifiedKFold by `post_id` (fallback GroupKFold) to avoid leakage.
Parameters are managed via Hydra; experiments and artifacts tracked in MLflow
(`sqlite:///mlflow.db`, `./mlruns`). HPO: not in scope for this feature.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: PyTorch, Transformers, Datasets/Pandas, scikit-learn,
iterative stratification (or equivalent for group‑stratified CV), Hydra,
MLflow, Optuna  
**Storage**: Local files for data; MLflow SQLite DB `mlflow.db`, artifacts under `mlruns/`; optional
Optuna SQLite `optuna.db`  
**Testing**: pytest (unit tests for data parsing, metric computation, splitting)  
**Target Platform**: Linux/macOS dev; GPU optional for speed  
**Project Type**: single (library + CLI)  
**Performance Goals**: Reasonable epoch time on DeBERTaV3‑base; CV completes without OOM  
**Constraints**: Enforce reproducibility (seed, deterministic where possible); honor NSP input format  
**Scale/Scope**: Dataset size TBD; CV=5; batch size tuned to memory

## Constitution Check

Gates (all PASS):
- P1 BERT-based classifier (HF): using DeBERTaV3‑base ✓
- P2 NSP input format: criterion–sentence pairs ✓
- P3 Hydra config: configs/ with overrides ✓
- P4 MLflow local: sqlite:///mlflow.db + ./mlruns ✓
- P5 Optuna if HPO: N/A (not used in this feature) ✓
- P6 Reproducibility: seeds, env snapshot, Hydra config logging ✓

## Project Structure

### Documentation (this feature)

```text
specs/001-debertav3-5fold-evidence/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
```

### Source Code (repository root)

```text
src/Project/SubProject/
├── models/              # HF model wrappers (extend if needed)
├── utils/               # logging, seeding, mlflow helpers
└── engine/              # NEW: training/inference entrypoints (to add)

scripts/
└── train_cv.py          # NEW: CLI for 5-fold training (to add)

tests/
├── unit/
└── integration/
```

**Structure Decision**: Single-project Python package aligned to existing
`src/Project/SubProject/` layout; add `engine/` and `scripts/train_cv.py` for
Trainer-based CV loop and Hydra integration. Tests mirror under `tests/`.

## Phase 0: Research (resolve unknowns)

Topics and decisions captured in research.md:
- GroupStratifiedKFold implementation: prefer scikit‑learn's
  StratifiedGroupKFold if available; otherwise use a third‑party iterative
  stratification library or approximate via GroupKFold + per‑fold class
  balancing checks.
- Fused AdamW availability: use HF Trainer with `optim=adamw_torch_fused` and
  auto‑fallback to `adamw_torch` if unsupported.
- Loss implementation in Trainer: override `compute_loss` to support weighted CE
  and optional Focal Loss (γ=2.0, α from frequencies) with Hydra switch.
- Metrics: compute accuracy, macro‑F1, positive‑class F1, ROC‑AUC, PR‑AUC via
  scikit‑learn; log per‑fold and aggregate.

## Phase 1: Design & Contracts

Artifacts to generate:
- data-model.md: Sample, FoldSplit, ModelArtifact entities and fields
- quickstart.md: end‑to‑end commands (MLflow UI + training CLI + inference)
- contracts/: N/A (no external API; CLI contract described in quickstart)

Agent context updated via `.specify/scripts/bash/update-agent-context.sh codex`.

## Phase 2: Implementation Strategy (high level)

1) Data loader: build criterion–sentence pairs; construct stratified random
   negatives to target 1:3 pos:neg ratio (`data.neg_ratio=3`), then persist
   split manifests per fold (grouped by post_id)
2) Trainer (HF) setup: DeBERTaV3‑base, tokenizer (max_length=512; truncation=longest_first),
   full fine‑tune (unfreeze all layers), compute_metrics
3) Optimizer & schedule: `adamw_torch_fused` or fallback to `adamw_torch`;
   LR scheduler `linear` with `warmup_ratio=0.06`
4) Precision: prefer BF16; else FP16; else FP32 (log chosen mode)
5) Loss: weighted CE by default; focal optional
6) Metrics & selection: compute accuracy, macro/pos F1, ROC/PR AUC. Set
   `metric_for_best_model=f1_macro` and `greater_is_better=true` in Trainer.
7) CV orchestration: parent MLflow run with 5 child runs
8) Aggregation: compute mean/std metrics; save summary JSON + plots
9) Inference: simple function/CLI for single pair
10) Reproducibility: seed, env snapshot (`pip freeze`), config logging

## Complexity Tracking

N/A — No constitution violations.
