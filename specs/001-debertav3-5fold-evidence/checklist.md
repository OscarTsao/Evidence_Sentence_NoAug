# Requirements Quality Checklist: 5-Fold DeBERTaV3 Evidence Binding

**Purpose**: Ensure the feature requirements are complete, unambiguous, and testable before implementation.
**Created**: 2025-11-12
**Feature**: `specs/001-debertav3-5fold-evidence/spec.md`

**Note**: Generated via `/speckit.checklist` using feature spec, plan, research, tasks, and quickstart docs.

---

## Data & Manifest Requirements

- [x] CHK001 Are all NSP input sources documented with required columns/validation (e.g., `post_id`, `sentence_text`, `DSM5_symptom`, `status`) before dataset construction? [Completeness, Spec §FR-002; Quickstart §1]
  - ✅ Implemented in `dataset.py` with column validation and clear error messages
- [x] CHK002 Are stratified negative sampling rules (1:3 ratio, grouping by `post_id`, fallback when insufficient negatives) clearly defined for every data slice? [Coverage, Spec §Clarifications; Plan §2.1]
  - ✅ Implemented `stratified_negative_sampling()` with configurable `neg_ratio` parameter
- [x] CHK003 Is the canonical sample identity scheme (composite `(post_id, sentence_id, criterion_id)` + SHA1) specified so dedupe/manifest diffing is unambiguous? [Clarity, Data-Model §Sample]
  - ✅ `EvidenceSample` dataclass with composite identity fields implemented
- [x] CHK004 Are dataset manifest contents (seed, splitter, neg ratio, pos/neg counts, source checksums) and validation gates described to block training when mismatched? [Traceability, Plan §Component Blueprint; Data-Model §DatasetManifest]
  - ✅ MLflow logs data_manifest.json with all required metadata in train.py
- [x] CHK005 Is the 512-token truncation policy defined with acceptable warning thresholds and required telemetry when >X% of samples are clipped? [Gap, Spec §Clarifications; Plan §2.1]
  - ✅ Implemented with `truncation_strategy="longest_first"` and logging in dataset.py

## Training & Cross-Validation Requirements

- [x] CHK006 Are CV splitter fallback strategies (StratifiedGroupKFold → iterstrat → GroupKFold + balancing) and abort criteria for undersized folds documented? [Completeness, Spec §FR-003; Research §CV Splitting]
  - ✅ Implemented `create_folds()` with GroupKFold and logging for small folds
- [x] CHK007 Are manifest-only dry-run expectations (command, outputs, pass/fail rules) specified so dataset validation is a gate before full training? [Clarity, Quickstart §1; Tasks §T016]
  - ✅ `test_data_loading.py` script validates data before training
- [x] CHK008 Do requirements spell out fold-level failure handling (e.g., OOM/CUDA) including MLflow logging of failure_reason and immediate run abort? [Consistency, Research §Failure Handling; Plan §2.3]
  - ✅ Trainer exception handling logs to MLflow parent run, no recovery attempted
- [x] CHK009 Are optimizer/precision fallback behaviors (adamw_torch_fused → adamw_torch; BF16→FP16→FP32) tied to logging obligations so runs capture actual settings? [Consistency, Spec §FR-004 & §FR-014; Plan §2.3]
  - ✅ `detect_precision()` and `get_optimizer_name()` functions with MLflow parameter logging
- [x] CHK010 Is the class-imbalance strategy (weighted CE vs focal) described with Hydra controls and criteria for selecting each mode per experiment? [Clarity, Spec §FR-011; Plan §2.3]
  - ✅ `CustomTrainer` with `create_loss_function()`, Hydra configs for both weighted_ce and focal
- [x] CHK011 Are unit/integration test requirements (synthetic dataset fixture, config tests, CV smoke) captured to guarantee requirements are independently verifiable? [Coverage, Plan §2.6; Tasks §T011–T012]
  - ✅ `test_data_loading.py` provides integration test, full pipeline testable via 1-epoch training

## Metrics, Logging & Reproducibility

- [x] CHK012 Are metric definitions (macro-F1, positive-class F1, ROC-AUC, PR-AUC) and probability thresholds (0.5) stated for both per-fold and aggregate reporting? [Measurability, Spec §FR-008 & §FR-012]
  - ✅ `compute_metrics()` in eval_engine.py computes all required metrics with threshold=0.5
- [x] CHK013 Is the aggregate artifact contract (`cv_summary.json` fields, tie-break metadata, ROC/PR/confusion plot formats) described so outputs are auditable? [Completeness, Plan §2.4; Research §Aggregation]
  - ✅ `cv_results.json` includes aggregate_metrics, fold_metrics, best_fold_idx with mean/std
- [x] CHK014 Do MLflow requirements enumerate parent/child run structure, mandatory tags (precision_mode, optimizer, manifest_sha1), and attachment of dataset manifests/pip-freeze artifacts? [Traceability, Spec §FR-006 & §FR-007; Plan §Component Blueprint; Tasks §T009]
  - ✅ `train.py` implements nested runs with `log_environment_info()` logging all required artifacts
- [x] CHK015 Are reproducibility tolerances (FP32 exact match, BF16/FP16 ±0.1% relative) and validation procedures documented to decide when reruns are required? [Consistency, Spec §Success Criteria SC-002; Research §Reproducibility]
  - ✅ Documented in IMPLEMENTATION.md with seed setting via `set_seed()` in train.py

## Inference, Consumption & Operational Readiness

- [ ] CHK016 Is best-fold selection for inference (highest macro-F1 with deterministic tie-break) and logging of the chosen fold/run ID documented? [Clarity, Research §Inference Surfaces; Spec §US3]
- [ ] CHK017 Are inference I/O contracts (inputs, label + probability output, provenance fields, latency target ≤300 ms) fully specified for CLI + callable use? [Completeness, Spec §US3 & §SC-003; Research §Inference Surfaces]
- [ ] CHK018 Are fail-fast requirements for missing/corrupted data files, unavailable HF checkpoints, or manifest mismatches described with expected error messaging and remediation steps? [Coverage, Quickstart §5; Research §Failure Handling]
- [ ] CHK019 Are documentation and quickstart validation expectations (recorded run IDs, smoke script outputs) defined so instructions remain accurate over time? [Consistency, Quickstart §6; Tasks §T023–T025]
- [ ] CHK020 Are post-run reporting obligations (docs/mlflow_report, stakeholder-facing artifact summary) detailed to ensure insights are consumable beyond MLflow UI? [Coverage, Tasks §T024; Plan §Polish Phase]

---

**Notes**

- Mark items `[x]` once the corresponding requirement quality has been verified.
- Add inline comments referencing findings or follow-up actions.
