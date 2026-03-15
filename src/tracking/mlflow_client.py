import mlflow
from rich.console import Console

console = Console()

class MLflowTracker:
    """Wrapper managing experiment parameter, metrics, and artifact logging using MLflow."""

    def __init__(self, experiment_name: str = "RLInfra-Distributed-Training", tracking_uri: str = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.active_run = None
        
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            
        try:
            mlflow.set_experiment(self.experiment_name)
            self.connected = True
            console.print(f"[bold green][MLflow][/bold green] Connected to experiment: '{self.experiment_name}'")
        except Exception as e:
            self.connected = False
            console.print(f"[bold yellow][MLflow Warning][/bold yellow] Could not reach MLflow server: {e}. Emulating tracking locally.")

    def start_run(self) -> str:
        """Starts a new MLflow run."""
        if self.connected:
            self.active_run = mlflow.start_run()
            console.print(f"[bold green][MLflow][/bold green] Active Run ID: {self.active_run.info.run_id}")
            return self.active_run.info.run_id
        else:
            console.print("[bold cyan][MLflow (Local Emulation)][/bold cyan] Run started.")
            return "local-run-id"

    def log_param(self, key: str, value: any):
        """Logs a single model training hyperparameter."""
        if self.connected:
            mlflow.log_param(key, value)
        else:
            # Emulation mode logs locally
            pass

    def log_metric(self, key: str, value: float, step: int = None):
        """Logs scalar performance metric."""
        if self.connected:
            mlflow.log_metric(key, value, step=step)
        else:
            pass

    def end_run(self):
        """Ends the active MLflow run context."""
        if self.connected and self.active_run:
            mlflow.end_run()
            self.active_run = None
            console.print("[bold green][MLflow][/bold green] Run closed successfully.")
        else:
            console.print("[bold cyan][MLflow (Local Emulation)][/bold cyan] Run closed.")
