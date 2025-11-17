from typing import Optional, Dict, Any


class Loggers:
    def __init__(self, wandb=None, mlflow=None):
        self.wandb = wandb
        self.mlflow = mlflow

    def log_params(self, params: Dict[str, Any]):
        if self.wandb:
            self.wandb.config.update(dict(params), allow_val_change=True)
        if self.mlflow:
            try:
                self.mlflow.log_params(params)
            except Exception:
                for k, v in params.items():
                    try:
                        self.mlflow.log_param(k, v)
                    except Exception:
                        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.wandb:
            data = dict(metrics)
            if step is not None:
                data["step"] = step
            self.wandb.log(data)

        if self.mlflow:
            for k, v in metrics.items():
                try:
                    if step is None:
                        self.mlflow.log_metric(k, v)
                    else:
                        self.mlflow.log_metric(k, v, step=step)
                except Exception:
                    pass

    def finish(self):
        if self.wandb:
            try:
                self.wandb.finish()
            except Exception:
                pass
        if self.mlflow:
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass


def init_loggers(
    project: str = "ml-portfolio",
    run_name: str = "dev",
    use_wandb: bool = False,
    use_mlflow: bool = False,
    tags: Optional[Dict[str, Any]] = None,
) -> Loggers:
    wb = None
    mf = None

    if use_wandb:
        try:
            import wandb

            wb = wandb.init(project=project, name=run_name, config=tags or {})
        except Exception as e:
            print(f"[warn] W&B not enabled: {e}")

    if use_mlflow:
        try:
            import mlflow

            mlflow.set_experiment(project)
            mlflow.start_run(run_name=run_name)
            mf = mlflow
        except Exception as e:
            print(f"[warn] MLflow not enabled: {e}")

    return Loggers(wandb=wb, mlflow=mf)
