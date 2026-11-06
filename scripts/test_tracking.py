import mlflow

from mtlab.experiments.tracking import (
    start_experiment_run,
    log_dataset_info,
)


with start_experiment_run(
    experiment_name="00-mlflow-test",
    run_name="tracking-module-check",
    params={
        "test_type": "tracking_module",
        "dataset_scope": "test",
    },
    tags={
        "research_area": "infrastructure",
    },
):
    log_dataset_info(
        role="train",
        name="Samanantar",
        split="train",
        row_count=2000,
        path="data/processed/samanantar/en-hi/word/",
    )

    mlflow.log_metric("tracking_check", 1.0)

print("Tracking module verified.")