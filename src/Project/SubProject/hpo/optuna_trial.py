"""Objective function for Optuna HPO."""

import optuna
from optuna.trial import TrialState
from omegaconf import DictConfig, OmegaConf
import copy
from typing import Dict, Any

from ..engine.train_engine import run_cross_validation
from ..data.dataset import (
    load_dsm5_criteria,
    load_redsm5_data,
    stratified_negative_sampling,
    create_folds,
    compute_class_weights,
)
from .optuna_utils import suggest_hyperparameters


def create_objective(base_config: DictConfig, dsm5_criteria: Dict, df_redsm5: Any, folds: Any):
    """
    Create an objective function for Optuna optimization.

    This factory function creates a closure that captures the data and configuration,
    and returns an objective function that Optuna can optimize.

    Args:
        base_config: Base Hydra configuration
        dsm5_criteria: DSM-5 criteria dictionary
        df_redsm5: ReDSM5 dataset dataframe
        folds: Cross-validation folds

    Returns:
        Callable: Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function to optimize.

        Args:
            trial: Optuna trial object

        Returns:
            float: Objective value (metric to maximize/minimize)
        """
        # Suggest hyperparameters for this trial
        trial_params = suggest_hyperparameters(trial, base_config)

        # Create a copy of the base config and update with trial params
        trial_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))

        # Update training arguments
        for key, value in trial_params.items():
            if key in ["learning_rate", "per_device_train_batch_size", "num_train_epochs",
                       "warmup_ratio", "weight_decay"]:
                OmegaConf.update(trial_config, f"training.args.{key}", value)
            elif key == "loss_function":
                OmegaConf.update(trial_config, "loss.type", value)
            elif key == "focal_gamma":
                OmegaConf.update(trial_config, "loss.focal_gamma", value)
            elif key == "focal_alpha":
                OmegaConf.update(trial_config, "loss.focal_alpha", value)

        # Update the experiment name to include trial number
        mlflow_exp_name = f"{base_config.hpo.mlflow.experiment_name}_trial_{trial.number}"
        OmegaConf.update(trial_config, "mlflow.experiment_name", mlflow_exp_name)

        try:
            # Compute class weights if using weighted CE loss
            class_weights = None
            if trial_params.get("loss_function", "weighted_ce") == "weighted_ce":
                class_weights = compute_class_weights(df_redsm5)

            # Run cross-validation with the trial hyperparameters
            fold_results, aggregate_metrics = run_cross_validation(
                config=trial_config,
                df=df_redsm5,
                folds=folds,
                dsm5_criteria=dsm5_criteria,
                class_weights=class_weights,
            )

            # Extract the metric to optimize
            metric_name = base_config.hpo.metric.name
            aggregation_method = base_config.hpo.metric.fold_aggregation

            # Get metric values from all folds
            metric_values = [
                fold_result["aggregate_metrics"][metric_name]
                for fold_result in fold_results
            ]

            # Aggregate across folds
            if aggregation_method == "mean":
                objective_value = sum(metric_values) / len(metric_values)
            elif aggregation_method == "median":
                sorted_values = sorted(metric_values)
                n = len(sorted_values)
                objective_value = (
                    sorted_values[n // 2]
                    if n % 2 != 0
                    else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
                )
            elif aggregation_method == "min":
                objective_value = min(metric_values)
            elif aggregation_method == "max":
                objective_value = max(metric_values)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")

            # Store fold results as user attributes for later analysis
            for fold_idx, metric_value in enumerate(metric_values):
                trial.set_user_attr(f"fold_{fold_idx}_{metric_name}", metric_value)

            # Store aggregate metrics
            for metric_key, metric_value in aggregate_metrics.items():
                if isinstance(metric_value, (int, float)):
                    trial.set_user_attr(f"aggregate_{metric_key}", metric_value)

            return objective_value

        except Exception as e:
            # Log the error and return a failure
            print(f"Trial {trial.number} failed with error: {e}")
            trial.set_user_attr("error", str(e))
            raise optuna.TrialPruned()

    return objective


def run_hpo_study(config: DictConfig) -> optuna.Study:
    """
    Run the complete HPO study.

    Args:
        config: Hydra configuration

    Returns:
        optuna.Study: Completed study object
    """
    from .optuna_utils import create_study
    from .optuna_callbacks import MLflowCallback, log_study_summary
    import mlflow

    print("Loading data...")
    # Load data (same as train.py)
    dsm5_criteria = load_dsm5_criteria(config.data.dsm5_dir)
    df_redsm5 = load_redsm5_data(config.data.csv_path, dsm5_criteria)
    df_redsm5 = stratified_negative_sampling(
        df_redsm5,
        pos_neg_ratio=config.data.pos_neg_ratio,
        strategy=config.data.neg_strategy,
    )
    folds = create_folds(df_redsm5, n_splits=config.training.n_folds)

    print(f"Loaded {len(df_redsm5)} samples with {config.training.n_folds} folds")

    # Create study
    print("Creating Optuna study...")
    study = create_study(config)

    # Create MLflow callback
    mlflow_callback = MLflowCallback(
        experiment_name=config.hpo.mlflow.experiment_name,
        nested=config.hpo.mlflow.nested_runs,
    )

    # Create objective function
    objective = create_objective(config, dsm5_criteria, df_redsm5, folds)

    # Start parent MLflow run
    with mlflow.start_run(run_name=f"hpo_study_{study.study_name}"):
        # Log study configuration
        mlflow.log_params({
            "n_trials": config.hpo.trials.n_trials,
            "n_folds": config.training.n_folds,
            "metric": config.hpo.metric.name,
            "sampler": config.hpo.sampler.type,
            "pruner": config.hpo.pruner.type,
        })

        # Run optimization
        print(f"Starting optimization with {config.hpo.trials.n_trials} trials...")
        study.optimize(
            objective,
            n_trials=config.hpo.trials.n_trials,
            timeout=config.hpo.trials.timeout,
            n_jobs=config.hpo.trials.n_jobs,
            callbacks=[mlflow_callback],
            show_progress_bar=True,
        )

        # Log study summary
        print("\nOptimization complete!")
        log_study_summary(study, output_path="optuna_study_summary.txt")

        # Save best parameters to YAML
        from .optuna_utils import get_best_params_as_yaml
        best_config_path = "configs/trainer/hpo_best.yaml"
        get_best_params_as_yaml(study, best_config_path)
        mlflow.log_artifact(best_config_path)

    return study
