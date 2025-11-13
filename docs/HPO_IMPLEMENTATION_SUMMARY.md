# Optuna HPO Implementation Summary

## Overview

This document summarizes the complete Optuna hyperparameter optimization (HPO) implementation for the DeBERTa-v3 Evidence Binding project.

## Implementation Date

2025-11-13

## Files Created

### Configuration Files (2 files)
1. `configs/hpo/default.yaml` - Full HPO configuration with 50 trials
2. `configs/hpo/quick.yaml` - Quick HPO configuration with 10 trials for testing

### Source Code (4 files)
3. `src/Project/SubProject/hpo/__init__.py` - Module initialization and exports
4. `src/Project/SubProject/hpo/optuna_utils.py` - Utility functions for study creation and parameter suggestion
5. `src/Project/SubProject/hpo/optuna_callbacks.py` - MLflow and pruning callbacks
6. `src/Project/SubProject/hpo/optuna_trial.py` - Objective function and study runner

### Scripts (1 file)
7. `scripts/hpo_search.py` - Main entry point for running HPO

### Documentation (2 files)
8. `docs/HPO_GUIDE.md` - Comprehensive user guide
9. `docs/HPO_IMPLEMENTATION_SUMMARY.md` - This file

### Modified Files (1 file)
10. `configs/config.yaml` - Added HPO configuration group to defaults

## Key Features

### 1. Comprehensive Search Space

The implementation optimizes 8 hyperparameters:
- **learning_rate**: Log-uniform distribution [1e-6, 1e-4]
- **per_device_train_batch_size**: Categorical [8, 16, 32]
- **num_train_epochs**: Integer [2, 5]
- **warmup_ratio**: Uniform [0.0, 0.1]
- **weight_decay**: Uniform [0.0, 0.05]
- **loss_function**: Categorical [weighted_ce, focal]
- **focal_gamma**: Uniform [1.0, 3.0] (conditional on focal loss)
- **focal_alpha**: Uniform [0.1, 0.5] (conditional on focal loss)

### 2. Optuna Integration

- **Sampler**: Tree-structured Parzen Estimator (TPE)
- **Pruner**: MedianPruner for early stopping of unpromising trials
- **Storage**: SQLite database for persistent study storage
- **Resumable**: Studies can be resumed from previous runs

### 3. MLflow Integration

- **Nested Runs**: Each trial is logged as a nested MLflow run
- **Hierarchical Tracking**: Study → Trials → Folds
- **Comprehensive Logging**: Parameters, metrics, artifacts, and study summary
- **Parameter Importance**: Automatically computed and logged if available

### 4. Cross-Validation

- Each trial runs full 5-fold cross-validation
- Metrics are aggregated across folds (mean by default)
- Per-fold metrics are stored as user attributes
- Supports multiple aggregation methods: mean, median, min, max

### 5. Configuration Management

- Fully integrated with existing Hydra configuration system
- CLI overrides supported
- Best parameters automatically saved to YAML format
- Easy to use with existing training pipeline

## Architecture

```
Hydra Config System
        ↓
    HPO Study
        ↓
    ┌──────┴──────┐
    ↓             ↓
Optuna Trial  MLflow Logging
    ↓             ↓
CV Training   Nested Runs
    ↓             ↓
 Metrics      Artifacts
```

## Usage Examples

### Basic Usage

```bash
# Run HPO with default configuration
python scripts/hpo_search.py

# Run quick test
python scripts/hpo_search.py hpo=quick

# Use best parameters
python scripts/train.py trainer=hpo_best
```

### Advanced Usage

```bash
# Customize number of trials
python scripts/hpo_search.py hpo.trials.n_trials=20

# Optimize for different metric
python scripts/hpo_search.py hpo.metric.name=pos_f1

# Search only focal loss
python scripts/hpo_search.py \
  hpo.search_space.loss_function.choices=[focal]

# Combine multiple overrides
python scripts/hpo_search.py \
  hpo.trials.n_trials=30 \
  training.n_folds=3 \
  hpo.metric.name=roc_auc
```

## Integration with Existing Code

The HPO system integrates seamlessly with existing components:

1. **Data Loading**: Uses existing `load_dsm5_criteria`, `load_redsm5_data`, etc.
2. **Training**: Uses existing `run_cross_validation` function
3. **Evaluation**: Uses existing `compute_metrics` function
4. **Configuration**: Extends existing Hydra configuration system
5. **Logging**: Integrates with existing MLflow setup

## Output Files

After running HPO, the following files are generated:

1. **optuna.db**: SQLite database containing all trial data
2. **configs/trainer/hpo_best.yaml**: Best hyperparameters in Hydra format
3. **optuna_study_summary.txt**: Human-readable study summary
4. **optuna_trials.csv**: Detailed trial results (in MLflow)

## Performance Considerations

### Memory Usage
- Each trial runs full 5-fold CV
- Memory usage scales with batch size and model size
- Typical memory: 8-16GB GPU memory per trial

### Time Estimates
- Single trial: ~15-30 minutes (depends on hardware and epochs)
- 10 trials (quick): ~2.5-5 hours
- 50 trials (full): ~12-25 hours

### Parallelization
- Default: 1 trial at a time (sequential)
- Can run multiple trials in parallel with `n_jobs > 1`
- Requires careful GPU management for parallel execution

## Best Practices

1. **Start with quick configuration**: Test with 10 trials first
2. **Monitor via MLflow UI**: Check progress and debug issues
3. **Resume studies**: Use persistent storage to resume interrupted runs
4. **Focus the search**: Narrow search space if you have prior knowledge
5. **Prune early**: MedianPruner saves time by stopping bad trials

## Testing Recommendations

### Phase 1: Verification (Quick Test)
```bash
# Test with minimal configuration
python scripts/hpo_search.py hpo=quick training.n_folds=2
```

### Phase 2: Development
```bash
# Run with moderate number of trials
python scripts/hpo_search.py hpo.trials.n_trials=20
```

### Phase 3: Production
```bash
# Full search with all trials
python scripts/hpo_search.py
```

## Future Enhancements

Possible improvements for future iterations:

1. **Multi-Objective Optimization**: Optimize for multiple metrics simultaneously
2. **Advanced Pruning**: Implement more sophisticated pruning strategies
3. **Distributed Search**: Add support for distributed trial execution
4. **Early Stopping**: Implement early stopping during training
5. **Warm Start**: Initialize search from previous best parameters
6. **Visualization**: Add real-time visualization dashboards
7. **Auto-reporting**: Generate comprehensive reports automatically

## Dependencies

The implementation requires:
- optuna >= 3.4
- mlflow >= 2.8
- transformers >= 4.40
- torch >= 2.2
- hydra-core (already in project)

All dependencies are specified in `pyproject.toml`.

## Maintenance

### Updating Search Space
Edit `configs/hpo/default.yaml` to modify hyperparameter ranges:

```yaml
hpo:
  search_space:
    learning_rate:
      type: "log_uniform"
      low: 1.0e-6  # Modify as needed
      high: 1.0e-4  # Modify as needed
```

### Changing Optimization Metric
Edit the metric configuration:

```yaml
hpo:
  metric:
    name: "macro_f1"  # Change to: pos_f1, roc_auc, pr_auc
    fold_aggregation: "mean"  # Change to: median, min, max
```

### Adding New Hyperparameters

1. Add to search space in `configs/hpo/default.yaml`
2. Update `suggest_hyperparameters()` in `optuna_utils.py`
3. Update objective function in `optuna_trial.py` to use the parameter

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure package is installed with `pip install -e .`
2. **CUDA OOM**: Reduce batch size in search space
3. **Slow Trials**: Reduce epochs or number of folds
4. **Failed Trials**: Check MLflow UI for error messages

### Debug Mode

Add verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

This implementation provides a production-ready, fully-integrated HPO system that:
- Seamlessly extends the existing codebase
- Requires minimal code changes to use
- Provides comprehensive logging and tracking
- Supports resumable and parallelizable execution
- Follows best practices for reproducibility

The system is ready to use for optimizing hyperparameters and should provide significant improvements over manual tuning.
