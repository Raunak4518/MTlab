import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACKING_URI = "http://127.0.0.1:5000"

def configure_tracking():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_system_metrics_sampling_interval(10)
def get_git_tags():
    revision = subprocess.run(["git","rev-parse","--short","HEAD"], cwd = PROJECT_ROOT, capture_output=True, text = True, check = False)
    status = subprocess.run(["git","status","--porcelain"], cwd=PROJECT_ROOT, capture_output=True , text= True, check=False)
    return {
        "git_commit":revision.stdout.strip(),
        "git_dirty": str(bool(status.stdout.strip())).lower(),
    }
def start_experiment_run(
    experiment_name: str,
    run_name: str,
    params: Mapping[str, object] | None = None,
    tags: Mapping[str, object] | None = None,
) -> mlflow.ActiveRun:
    """
    Start a tracked MTLab experiment.

    Automatically records:
    - System metrics
    - Python and MLflow versions
    - Operating system
    - CPU count
    - Git commit and working-tree state
    - Experiment-specific parameters and tags
    """
    configure_tracking()

    mlflow.set_experiment(experiment_name)

    run_tags = {
        "project": "MTLab",
        **get_git_tags(),
    }

    if tags:
        run_tags.update({
            key: str(value)
            for key, value in tags.items()
        })

    run_params = {
        "environment.python_version": platform.python_version(),
        "environment.mlflow_version": mlflow.__version__,
        "environment.platform": platform.platform(),
        "environment.cpu_count": os.cpu_count() or 0,
    }

    if params:
        run_params.update(params)

    run = mlflow.start_run(
        run_name=run_name,
        tags=run_tags,
        log_system_metrics=True,
    )

    mlflow.log_params(run_params)

    return run


def log_dataset_info(
    role: str,
    name: str,
    split: str,
    row_count: int,
    path: str | Path,
) -> None:
    """Record the dataset used by the current run."""
    mlflow.log_params({
        f"data.{role}.name": name,
        f"data.{role}.split": split,
        f"data.{role}.rows": row_count,
        f"data.{role}.path": str(path),
    })