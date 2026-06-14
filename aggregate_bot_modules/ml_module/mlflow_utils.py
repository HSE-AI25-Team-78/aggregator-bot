from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency path
    mlflow = None


def mlflow_enabled() -> bool:
    raw_value = os.getenv("ENABLE_MLFLOW", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def get_default_tracking_uri(project_root: Path) -> str:
    return (project_root / "ML" / "mlruns").resolve().as_uri()


def get_git_context(project_root: Path) -> dict[str, str | None]:
    def read_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return result.stdout.strip() or None

    return {
        "branch": read_git(["git", "-C", str(project_root), "branch", "--show-current"]),
        "commit": read_git(["git", "-C", str(project_root), "rev-parse", "HEAD"]),
    }


def configure_mlflow(project_root: Path, experiment_name: str, run_name: str) -> dict[str, str] | None:
    if not mlflow_enabled():
        print("[*] MLflow отключён через ENABLE_MLFLOW=0.")
        return None
    if mlflow is None:
        print("[warn] mlflow не установлен, пропускаем experiment tracking.")
        return None

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", get_default_tracking_uri(project_root))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
    run_name = os.getenv("MLFLOW_RUN_NAME", run_name)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name": run_name,
    }


@contextmanager
def start_mlflow_run(
    *,
    project_root: Path,
    experiment_name: str,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Iterator[dict[str, str] | None]:
    cfg = configure_mlflow(project_root=project_root, experiment_name=experiment_name, run_name=run_name)
    run_context = nullcontext(None)
    if cfg and mlflow is not None:
        run_context = mlflow.start_run(run_name=cfg["run_name"])

    with run_context:
        if cfg and mlflow is not None:
            git_context = get_git_context(project_root)
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            if git_context.get("branch"):
                mlflow.set_tag("git_branch", git_context["branch"])
            if git_context.get("commit"):
                mlflow.set_tag("git_commit", git_context["commit"])
        yield cfg


def log_params(params: dict[str, Any]) -> None:
    if mlflow is None:
        return
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict[str, float]) -> None:
    if mlflow is None:
        return
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_artifact_if_exists(path: Path) -> None:
    if mlflow is None or not path.exists():
        return
    mlflow.log_artifact(str(path))


def log_artifacts(paths: list[Path]) -> None:
    for path in paths:
        log_artifact_if_exists(path)
