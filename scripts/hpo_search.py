#!/usr/bin/env python3
"""
Hyperparameter Optimization Script using Optuna

This script performs hyperparameter search using Optuna to find the best
hyperparameters for the DeBERTa-v3 model on the evidence binding task.

Usage:
    # Run with default configuration (50 trials)
    python scripts/hpo_search.py

    # Run with quick configuration (10 trials)
    python scripts/hpo_search.py hpo=quick

    # Override specific parameters
    python scripts/hpo_search.py hpo.trials.n_trials=20 training.n_folds=3

    # Use a specific loss function in search space
    python scripts/hpo_search.py hpo.search_space.loss_function.choices=[focal]
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from Project.SubProject.hpo.optuna_trial import run_hpo_study
from Project.SubProject.utils.seed import set_seed
import mlflow


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main function for hyperparameter optimization.

    Args:
        config: Hydra configuration
    """
    # Print configuration
    print("=" * 80)
    print("Hyperparameter Optimization Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Verify HPO config exists
    if "hpo" not in config:
        raise ValueError(
            "HPO configuration not found. Please ensure you have hpo config group. "
            "Try running with: python scripts/hpo_search.py hpo=default"
        )

    # Set up MLflow
    mlflow_config = config.mlflow
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)

    # Run HPO study
    try:
        study = run_hpo_study(config)

        # Print final results
        print("\n" + "=" * 80)
        print("HPO Study Completed!")
        print("=" * 80)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        for param, value in study.best_trial.params.items():
            print(f"  {param}: {value}")
        print("=" * 80)

        print("\nBest configuration saved to: configs/trainer/hpo_best.yaml")
        print("To train with best parameters, run:")
        print("  python scripts/train.py trainer=hpo_best")

    except KeyboardInterrupt:
        print("\nHPO interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nHPO failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
