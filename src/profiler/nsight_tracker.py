import torch
from rich.console import Console

console = Console()

class NsightTracker:
    """NVTX instrumentation wrapper for Nvidia Nsight Systems and Compute timeline tracing."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available and self.enabled:
            console.print("[bold yellow][Nsight Tracker][/bold yellow] CUDA not available. Running in NVTX dry-run emulation mode.")

    def push_range(self, message: str, color: str = "blue"):
        """Pushes a named NVTX range onto the stack."""
        if not self.enabled:
            return
            
        if self.cuda_available:
            # NVidia custom marking on active CUDA streams
            torch.cuda.nvtx.range_push(message)
        else:
            # Emulated stdout diagnostic trace for development
            pass

    def pop_range(self):
        """Pops the last NVTX range off the stack."""
        if not self.enabled:
            return
            
        if self.cuda_available:
            torch.cuda.nvtx.range_pop()
        else:
            pass

    def mark_event(self, name: str):
        """Creates a single timepoint NVTX marker."""
        if not self.enabled:
            return
            
        if self.cuda_available:
            torch.cuda.nvtx.mark(name)
        else:
            pass
