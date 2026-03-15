import os
import torch
from pathlib import Path
from rich.console import Console

console = Console()

class PyTorchProfilerWrapper:
    """Wrapper configuring and orchestrating PyTorch Profiler tracing."""

    def __init__(self, export_dir: str = "./profiler_traces", wait: int = 2, warmup: int = 2, active: int = 5, repeat: int = 1):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.profiler = None

    def get_schedule(self) -> torch.profiler.ProfilerActivity:
        """Returns the profiler schedule configuration."""
        return torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat
        )

    def start(self):
        """Initializes and runs the PyTorch Profiler context."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        # Build profiler with tensorboard trace exportation
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=self.get_schedule(),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.export_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.start()
        console.print(f"[bold cyan][Profiler][/bold cyan] PyTorch Profiler started. Traces will export to '{self.export_dir}'")

    def step(self):
        """Notifies the profiler of training iteration boundary."""
        if self.profiler:
            self.profiler.step()

    def stop(self):
        """Stops the profiler execution, flushing records to disk."""
        if self.profiler:
            self.profiler.stop()
            console.print("[bold green][Profiler][/bold green] PyTorch Profiler stopped and traces flushed successfully.")
