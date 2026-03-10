from __future__ import annotations

import logging
from dataclasses import replace

import mlflow
import optuna

from tennisvision.core.experiments import ExperimentConfig, run_experiment
from tennisvision.core.mlflow_utils import setup_mlflow
from tennisvision.core.utils import setup_logging


def objective(trial: optuna.Trial, base_cfg: ExperimentConfig) -> float:
    logger.info(f"=== Starting Trial {trial.number} ===")

    # sample hyperparams
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
    params = {"weight_decay": weight_decay, "label_smoothing": label_smoothing}

    if base_cfg.use_scheduler:
        # Optimize scheduler parameters
        head_patience = trial.suggest_int("head_scheduler_patience", 1, 5)
        head_factor = trial.suggest_float("head_scheduler_factor", 0.3, 0.8)
        ft_patience = trial.suggest_int("ft_scheduler_patience", 1, 5)
        ft_factor = trial.suggest_float("ft_scheduler_factor", 0.3, 0.8)
        params |= {
            "head_scheduler_patience": head_patience,
            "head_scheduler_factor": head_factor,
            "ft_scheduler_patience": ft_patience,
            "ft_scheduler_factor": ft_factor,
        }

        cfg = replace(
            base_cfg,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            head_scheduler_patience=head_patience,
            head_scheduler_factor=head_factor,
            ft_scheduler_patience=ft_patience,
            ft_scheduler_factor=ft_factor,
            seed=base_cfg.seed,
        )
    else:
        # Optimize learning rates
        head_lr = trial.suggest_float("head_lr", 1e-4, 3e-2, log=True)
        finetune_lr = trial.suggest_float("finetune_lr", 1e-6, 3e-4, log=True)
        params |= {"head_lr": head_lr, "finetune_lr": finetune_lr}
        cfg = replace(
            base_cfg,
            head_lr=head_lr,
            finetune_lr=finetune_lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            seed=base_cfg.seed,
        )

    run_name = f"hpo_trial_{trial.number}"

    # nested run, inside the parent run
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(params)

        out = run_experiment(cfg, log_model=False, log_confusion_matrix=False, save_checkpoints=False)

    # run_experiment already logs best/val_metric to mlflow, but we want here also to return it to optuna
    best = out.get("ft_best_val_metric") or out.get("head_best_val_metric") or -1.0
    mlflow.log_metric("hpo/objective", float(best))

    logger.info(f"=== Finished Trial {trial.number} | Score: {best:.4f} ===")
    return float(best)


logger = logging.getLogger(__name__)


def main(trials=6) -> None:

    setup_logging(logging.INFO)
    logger.info("Start training")

    base_cfg = ExperimentConfig(
        image_root="data/Tennis positions/images",
        model_name="mobilenet_v3_large",
        head_epochs=4,
        finetune=False,
        finetune_epochs=8,
        batch_size=32,
        num_workers=0,  # on macOS multiprocessing with DataLoader in a loop (like HPO) often hangs.
        pin_memory=True,
        mlflow_experiment_name="TennisVision",
        enable_explainability=False 
    )

    study = optuna.create_study(direction="maximize")

    # setting an experiment for the whole hpo
    setup_mlflow(
        experiment_name="TennisVisionHPO",
        tracking_uri=base_cfg.mlflow_tracking_uri,
        set_experiment=True,
    )

    with mlflow.start_run(run_name="HPO_parent"):
        mlflow.log_param("n_trials", trials)
        study.optimize(lambda t: objective(t, base_cfg), n_trials=trials)

        mlflow.log_metric("hpo/best_value", float(study.best_value))
        mlflow.log_params({f"hpo_best/{k}": v for k, v in study.best_params.items()})

        print("BEST:", study.best_value, study.best_params)

        # Final run: full logs + checkpoints + explainability turned on
        final_cfg = replace(
            base_cfg,
            **study.best_params,  # head_lr, finetune_lr, weight_decay, label_smoothing
            head_epochs=base_cfg.head_epochs,
            finetune_epochs=base_cfg.finetune_epochs,
            seed=base_cfg.seed,
            enable_explainability = True
        )

        # final run (explainability images are logged to MLflow inside run_experiment)
        run_experiment(final_cfg, log_model=True, log_confusion_matrix=True, save_checkpoints=True)

        # getting the last child run (which is our final model)
        client = mlflow.tracking.MlflowClient()
        run_id = mlflow.active_run().info.run_id
        child_runs = client.search_runs(
            experiment_ids=[mlflow.active_run().info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            order_by=["start_time DESC"],
        )
        final_run_id = child_runs[0].info.run_id if child_runs else None

        # Register final model
        # if final_run_id:
        #     final_model_uri = f"runs:/{final_run_id}/model"
        #     mlflow.register_model(model_uri=final_model_uri, name="TennisVision")


if __name__ == "__main__":
    main()
