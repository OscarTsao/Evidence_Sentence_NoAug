# Hyperparameter Optimization (HPO) Guide

This guide explains how to use Optuna for hyperparameter optimization in the DeBERTa-v3 Evidence Binding project.

## Overview

The HPO system uses Optuna to automatically search for the best hyperparameters for the model. It integrates seamlessly with the existing Hydra configuration system and MLflow tracking.

## Quick Start

### Basic Usage

Run HPO with default configuration (50 trials):

```bash
python scripts/hpo_search.py
```

Run HPO with quick configuration (10 trials for testing):

```bash
python scripts/hpo_search.py hpo=quick
```

### Using Best Parameters

After HPO completes, the best parameters are automatically saved to `configs/trainer/hpo_best.yaml`. Train with these parameters:

```bash
python scripts/train.py trainer=hpo_best
```

## Configuration

### HPO Configuration Files

- `configs/hpo/default.yaml` - Full HPO configuration (50 trials)
- `configs/hpo/quick.yaml` - Quick HPO for testing (10 trials)

### Key Configuration Options

#### Study Configuration

```yaml
hpo:
  study:
    study_name: "deberta-v3-hpo"
    storage: "sqlite:///optuna.db"  # Persistent study storage
    direction: "maximize"  # Maximize macro F1
    load_if_exists: true  # Resume from existing study
```

#### Trial Configuration

```yaml
hpo:
  trials:
    n_trials: 50  # Number of trials to run
    timeout: null  # Optional timeout in seconds
    n_jobs: 1  # Number of parallel jobs
```

#### Search Space

The following hyperparameters are optimized:

| Hyperparameter | Type | Range/Choices | Description |
|----------------|------|---------------|-------------|
| `learning_rate` | log-uniform | [1e-6, 1e-4] | Learning rate for AdamW optimizer |
| `per_device_train_batch_size` | categorical | [8, 16, 32] | Batch size per device |
| `num_train_epochs` | integer | [2, 5] | Number of training epochs |
| `warmup_ratio` | uniform | [0.0, 0.1] | Warmup ratio for learning rate scheduler |
| `weight_decay` | uniform | [0.0, 0.05] | Weight decay for regularization |
| `loss_function` | categorical | [weighted_ce, focal] | Loss function to use |
| `focal_gamma` | uniform | [1.0, 3.0] | Focal loss gamma parameter (if focal loss) |
| `focal_alpha` | uniform | [0.1, 0.5] | Focal loss alpha parameter (if focal loss) |

### Optimization Metric

By default, the system optimizes for `macro_f1` averaged across all folds. You can change this:

```yaml
hpo:
  metric:
    name: "macro_f1"  # Options: macro_f1, pos_f1, roc_auc, pr_auc
    fold_aggregation: "mean"  # Options: mean, median, min, max
```

### Pruning

Optuna can prune unpromising trials early to save time:

```yaml
hpo:
  pruner:
    type: "median"  # Prune trials performing worse than median
    n_startup_trials: 5  # Don't prune first 5 trials
    n_warmup_steps: 0  # Start pruning from beginning
    interval_steps: 1  # Check every step
```

## Advanced Usage

### Customizing the Search Space

Edit `configs/hpo/default.yaml` to modify the search space:

```yaml
hpo:
  search_space:
    # Add or modify hyperparameters
    learning_rate:
      type: "log_uniform"
      low: 5.0e-6  # Changed from 1.0e-6
      high: 5.0e-5  # Changed from 1.0e-4
```

### Override Configuration from CLI

```bash
# Change number of trials
python scripts/hpo_search.py hpo.trials.n_trials=20

# Change number of folds
python scripts/hpo_search.py training.n_folds=3

# Search only focal loss
python scripts/hpo_search.py hpo.search_space.loss_function.choices=[focal]

# Combine multiple overrides
python scripts/hpo_search.py \
  hpo.trials.n_trials=30 \
  hpo.metric.name=pos_f1 \
  training.n_folds=3
```

### Resuming a Study

Studies are automatically saved to SQLite. To resume:

```bash
# This will continue from where it left off
python scripts/hpo_search.py
```

To start fresh:

```bash
# Delete the database and start over
rm optuna.db
python scripts/hpo_search.py
```

## MLflow Integration

### Viewing Results

The HPO system logs all trials to MLflow. View results in the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### MLflow Structure

```
Study Run (Parent)
├── Trial 0 (Child)
│   ├── Fold 0 (Grandchild)
│   ├── Fold 1 (Grandchild)
│   ├── ...
│   └── Fold 4 (Grandchild)
├── Trial 1 (Child)
│   └── ...
├── ...
└── Study Summary (Child)
```

### Logged Information

For each trial:
- All hyperparameters
- Objective value (aggregate metric)
- Per-fold metrics
- Training duration
- Best trial indicator

For the study:
- Number of trials (total, completed, pruned, failed)
- Best trial parameters
- Best metric value
- Parameter importance (if available)
- Trials dataframe (CSV artifact)

## Output Files

After HPO completes, the following files are created:

1. `optuna.db` - SQLite database containing all trial data
2. `configs/trainer/hpo_best.yaml` - Best hyperparameters in Hydra format
3. `optuna_study_summary.txt` - Text summary of the study
4. `optuna_trials.csv` - CSV file with all trial results (in MLflow)

## Architecture

### Components

1. **Configuration** (`configs/hpo/`)
   - YAML files defining search space and study settings

2. **HPO Module** (`src/Project/SubProject/hpo/`)
   - `optuna_trial.py` - Objective function and study runner
   - `optuna_utils.py` - Utility functions for creating study and suggesting parameters
   - `optuna_callbacks.py` - Callbacks for MLflow logging and pruning

3. **Entry Point** (`scripts/hpo_search.py`)
   - Main script to run HPO

### Workflow

```
1. Load configuration (Hydra)
   ↓
2. Load and prepare data
   ↓
3. Create Optuna study
   ↓
4. For each trial:
   a. Suggest hyperparameters
   b. Update configuration
   c. Run 5-fold cross-validation
   d. Compute aggregate metric
   e. Log to MLflow
   f. Check for pruning
   ↓
5. Select best trial
   ↓
6. Save best configuration
   ↓
7. Log study summary
```

## Best Practices

### 1. Start with Quick Configuration

```bash
# Test everything works first
python scripts/hpo_search.py hpo=quick
```

### 2. Use Appropriate Number of Trials

- Quick testing: 10 trials
- Development: 20-30 trials
- Production: 50-100 trials

### 3. Monitor Progress

Check MLflow UI regularly to:
- See which hyperparameters work best
- Identify if more trials are needed
- Debug any failed trials

### 4. Parallel Execution

For multiple GPUs:

```bash
python scripts/hpo_search.py hpo.trials.n_jobs=2
```

**Note:** This requires careful GPU management. By default, use `n_jobs=1` for single GPU.

### 5. Focus the Search

If you know some hyperparameters work well, narrow the search:

```yaml
hpo:
  search_space:
    learning_rate:
      type: "log_uniform"
      low: 1.5e-5  # Narrower range
      high: 2.5e-5
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you get OOM errors:

1. Reduce batch size in search space:
   ```yaml
   per_device_train_batch_size:
     type: "categorical"
     choices: [4, 8, 16]  # Removed 32
   ```

2. Reduce number of folds:
   ```bash
   python scripts/hpo_search.py training.n_folds=3
   ```

### Slow Trials

If trials are taking too long:

1. Reduce number of epochs:
   ```yaml
   num_train_epochs:
     type: "int"
     low: 2
     high: 3  # Reduced from 5
   ```

2. Reduce number of folds:
   ```bash
   python scripts/hpo_search.py training.n_folds=3
   ```

3. Enable pruning to stop bad trials early (already enabled by default)

### Failed Trials

Check the MLflow UI or Optuna logs to see error messages. Common causes:
- Invalid hyperparameter combinations
- CUDA errors (GPU issues)
- Data loading errors

### Study Not Resuming

Make sure `load_if_exists: true` in the configuration and the study name matches.

## Example Session

```bash
# 1. Quick test (10 trials)
python scripts/hpo_search.py hpo=quick

# 2. Check results in MLflow
mlflow ui

# 3. Run full HPO (50 trials)
python scripts/hpo_search.py

# 4. Train with best parameters
python scripts/train.py trainer=hpo_best

# 5. Compare results
# Open MLflow UI and compare hpo_best run with baseline
```

## References

- [Optuna Documentation](https://optuna.org/)
- [Hydra Documentation](https://hydra.cc/)
- [MLflow Documentation](https://mlflow.org/)
