# Data Model — Evidence Binding (NSP-Style)

## Entities

- Sample
  - Fields: `sample_id` (optional), `post_id`, `criterion_id`, `criterion_text`,
    `sentence_id` (optional), `sentence_text`, `label` ∈ {0,1}
  - Notes: NSP-style input pairs are derived from criterion + sentence.

- FoldSplit
  - Fields: `sample_id`, `fold_index` ∈ {0..4}
  - Metadata: `seed`, `grouping` = `post_id`, `strategy` ∈ {StratifiedGroupKFold, GroupKFold}

- ModelArtifact
  - Fields: `run_id`, `fold_index` (optional), `model_uri`, `tokenizer_files`,
    `config.json`, `metrics.json`
  - Notes: Stored via MLflow (artifacts under `mlruns/`).

## Relationships

- One `post_id` maps to many `Sample` rows.
- One `Sample` maps to exactly one `FoldSplit` assignment.
- One parent run aggregates many child fold runs and model artifacts.

