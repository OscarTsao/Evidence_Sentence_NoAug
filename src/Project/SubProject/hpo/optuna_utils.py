"""Utility functions for Optuna HPO."""

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, NopPruner
from typing import Dict, Any
from omegaconf import DictConfig


def create_study(config: DictConfig) -> optuna.Study:
    """
    Create an Optuna study based on the configuration.

    Args:
        config: Hydra configuration containing HPO settings

    Returns:
        optuna.Study: Configured Optuna study
    """
    hpo_config = config.hpo

    # Create sampler
    sampler = create_sampler(hpo_config.sampler)

    # Create pruner
    pruner = create_pruner(hpo_config.pruner)

    # Create study
    study = optuna.create_study(
        study_name=hpo_config.study.study_name,
        storage=hpo_config.study.storage,
        direction=hpo_config.study.direction,
        load_if_exists=hpo_config.study.load_if_exists,
        sampler=sampler,
        pruner=pruner,
    )

    return study


def create_sampler(sampler_config: DictConfig) -> optuna.samplers.BaseSampler:
    """
    Create an Optuna sampler based on configuration.

    Args:
        sampler_config: Sampler configuration

    Returns:
        optuna.samplers.BaseSampler: Configured sampler
    """
    if sampler_config.type == "tpe":
        return TPESampler(
            n_startup_trials=sampler_config.n_startup_trials,
            seed=42  # For reproducibility
        )
    elif sampler_config.type == "random":
        return RandomSampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_config.type}")


def create_pruner(pruner_config: DictConfig) -> optuna.pruners.BasePruner:
    """
    Create an Optuna pruner based on configuration.

    Args:
        pruner_config: Pruner configuration

    Returns:
        optuna.pruners.BasePruner: Configured pruner
    """
    if pruner_config.type == "median":
        return MedianPruner(
            n_startup_trials=pruner_config.n_startup_trials,
            n_warmup_steps=pruner_config.n_warmup_steps,
            interval_steps=pruner_config.interval_steps,
        )
    elif pruner_config.type == "none":
        return NopPruner()
    else:
        raise ValueError(f"Unknown pruner type: {pruner_config.type}")


def suggest_hyperparameters(trial: optuna.Trial, config: DictConfig) -> Dict[str, Any]:
    """
    Suggest hyperparameters for the trial based on the search space configuration.

    Args:
        trial: Optuna trial object
        config: Hydra configuration containing search space

    Returns:
        Dict[str, Any]: Dictionary of suggested hyperparameters
    """
    search_space = config.hpo.search_space
    params = {}

    # Learning rate
    if "learning_rate" in search_space:
        lr_config = search_space.learning_rate
        if lr_config.type == "log_uniform":
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", lr_config.low, lr_config.high, log=True
            )

    # Batch size
    if "per_device_train_batch_size" in search_space:
        bs_config = search_space.per_device_train_batch_size
        if bs_config.type == "categorical":
            params["per_device_train_batch_size"] = trial.suggest_categorical(
                "per_device_train_batch_size", bs_config.choices
            )

    # Number of epochs
    if "num_train_epochs" in search_space:
        epochs_config = search_space.num_train_epochs
        if epochs_config.type == "int":
            params["num_train_epochs"] = trial.suggest_int(
                "num_train_epochs", epochs_config.low, epochs_config.high
            )

    # Warmup ratio
    if "warmup_ratio" in search_space:
        warmup_config = search_space.warmup_ratio
        if warmup_config.type == "uniform":
            params["warmup_ratio"] = trial.suggest_float(
                "warmup_ratio", warmup_config.low, warmup_config.high
            )

    # Weight decay
    if "weight_decay" in search_space:
        wd_config = search_space.weight_decay
        if wd_config.type == "uniform":
            params["weight_decay"] = trial.suggest_float(
                "weight_decay", wd_config.low, wd_config.high
            )

    # Loss function
    if "loss_function" in search_space:
        loss_config = search_space.loss_function
        if loss_config.type == "categorical":
            params["loss_function"] = trial.suggest_categorical(
                "loss_function", loss_config.choices
            )

    # Focal loss parameters (only if focal loss is selected)
    if params.get("loss_function") == "focal":
        if "focal_gamma" in search_space:
            gamma_config = search_space.focal_gamma
            if gamma_config.type == "uniform":
                params["focal_gamma"] = trial.suggest_float(
                    "focal_gamma", gamma_config.low, gamma_config.high
                )

        if "focal_alpha" in search_space:
            alpha_config = search_space.focal_alpha
            if alpha_config.type == "uniform":
                params["focal_alpha"] = trial.suggest_float(
                    "focal_alpha", alpha_config.low, alpha_config.high
                )

    return params


def get_best_params_as_yaml(study: optuna.Study, output_path: str) -> None:
    """
    Save the best trial parameters as a YAML configuration file.

    Args:
        study: Optuna study with completed trials
        output_path: Path to save the YAML file
    """
    import yaml

    best_params = study.best_trial.params
    best_value = study.best_trial.value

    # Create a config structure
    config = {
        "training": {
            "args": {
                "learning_rate": best_params.get("learning_rate"),
                "per_device_train_batch_size": best_params.get("per_device_train_batch_size"),
                "num_train_epochs": best_params.get("num_train_epochs"),
                "warmup_ratio": best_params.get("warmup_ratio"),
                "weight_decay": best_params.get("weight_decay"),
            }
        },
        "loss": {
            "type": best_params.get("loss_function", "weighted_ce"),
        },
        "best_metric_value": best_value,
    }

    # Add focal loss params if applicable
    if best_params.get("loss_function") == "focal":
        config["loss"]["focal_gamma"] = best_params.get("focal_gamma")
        config["loss"]["focal_alpha"] = best_params.get("focal_alpha")

    # Save to YAML
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Best parameters saved to {output_path}")
    print(f"Best {study.direction.name} value: {best_value:.4f}")
