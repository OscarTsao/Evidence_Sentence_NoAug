"""Callbacks for Optuna HPO integration with MLflow."""

import optuna
from optuna.trial import TrialState
from typing import Optional
import mlflow


class MLflowCallback:
    """
    Callback to log Optuna trial information to MLflow.
    """

    def __init__(self, experiment_name: str, nested: bool = True):
        """
        Initialize MLflow callback.

        Args:
            experiment_name: Name of the MLflow experiment
            nested: Whether to create nested runs for each trial
        """
        self.experiment_name = experiment_name
        self.nested = nested
        self.parent_run_id = None

        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """
        Log trial information to MLflow after each trial completes.

        Args:
            study: Optuna study
            trial: Completed trial
        """
        # Only log completed trials
        if trial.state != TrialState.COMPLETE:
            return

        # Create a new run for this trial
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=self.nested):
            # Log parameters
            mlflow.log_params(trial.params)

            # Log trial number
            mlflow.log_param("trial_number", trial.number)

            # Log the objective value
            if trial.value is not None:
                mlflow.log_metric("objective_value", trial.value)

            # Log user attributes (like individual fold scores)
            for key, value in trial.user_attrs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, value)

            # Log best trial info
            if study.best_trial.number == trial.number:
                mlflow.log_param("is_best_trial", True)
                mlflow.log_metric("best_value", study.best_value)

            # Log trial duration
            if trial.duration is not None:
                mlflow.log_metric("trial_duration_seconds", trial.duration.total_seconds())


class PruningCallback:
    """
    Callback for early pruning of unpromising trials.
    This can be integrated into the training loop if needed.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "eval_f1_macro"):
        """
        Initialize pruning callback.

        Args:
            trial: Optuna trial object
            monitor: Metric to monitor for pruning
        """
        self.trial = trial
        self.monitor = monitor

    def __call__(self, epoch: int, metrics: dict) -> bool:
        """
        Check if trial should be pruned based on intermediate metrics.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics

        Returns:
            bool: True if trial should be pruned
        """
        # Report intermediate value to Optuna
        if self.monitor in metrics:
            self.trial.report(metrics[self.monitor], epoch)

            # Check if trial should be pruned
            if self.trial.should_prune():
                return True

        return False


def log_study_summary(study: optuna.Study, output_path: Optional[str] = None) -> None:
    """
    Log a summary of the study to MLflow and optionally save to file.

    Args:
        study: Completed Optuna study
        output_path: Optional path to save summary as text file
    """
    import pandas as pd

    # Get study statistics
    best_trial = study.best_trial
    n_trials = len(study.trials)
    n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
    n_failed = len([t for t in study.trials if t.state == TrialState.FAIL])

    summary = f"""
Optuna Study Summary
====================
Study Name: {study.study_name}
Direction: {study.direction.name}

Trial Statistics:
- Total Trials: {n_trials}
- Completed: {n_complete}
- Pruned: {n_pruned}
- Failed: {n_failed}

Best Trial:
- Trial Number: {best_trial.number}
- Value: {best_trial.value:.4f}
- Parameters:
"""

    for param, value in best_trial.params.items():
        summary += f"  - {param}: {value}\n"

    print(summary)

    # Save to file if requested
    if output_path:
        with open(output_path, "w") as f:
            f.write(summary)

    # Log to MLflow
    with mlflow.start_run(run_name="study_summary", nested=True):
        mlflow.log_param("study_name", study.study_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_complete", n_complete)
        mlflow.log_param("n_pruned", n_pruned)
        mlflow.log_param("n_failed", n_failed)
        mlflow.log_metric("best_value", best_trial.value)
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})

        # Create and log trials dataframe
        trials_df = study.trials_dataframe()
        trials_csv = "optuna_trials.csv"
        trials_df.to_csv(trials_csv, index=False)
        mlflow.log_artifact(trials_csv)

        # Log parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            importance_dict = {f"importance_{k}": v for k, v in importance.items()}
            mlflow.log_params(importance_dict)
        except Exception as e:
            print(f"Could not compute parameter importance: {e}")
