import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from rich.console import Console

console = Console()

class DDPTrainer:
    """Distributed Data Parallel (DDP) wrapper for robust distributed ML training."""
    
    def __init__(self, rank: int, world_size: int, model: torch.nn.Module, backend: str = "gloo"):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.model = model.to(rank) if torch.cuda.is_available() else model
        
        self._setup_process_group()
        self._wrap_model()

    def _setup_process_group(self):
        """Initialize the distributed process group."""
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
        
        console.print(f"[bold blue][Rank {self.rank}][/bold blue] Initializing process group with backend '{self.backend}'")
        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size
        )
        console.print(f"[bold green][Rank {self.rank}][/bold green] Process group initialized successfully.")

    def _wrap_model(self):
        """Wrap the model in PyTorch's DistributedDataParallel container."""
        if dist.is_initialized():
            if torch.cuda.is_available():
                self.ddp_model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=False)
            else:
                # Gloo CPU-based distributed training support
                self.ddp_model = DDP(self.model, find_unused_parameters=False)
            console.print(f"[bold green][Rank {self.rank}][/bold green] Model successfully wrapped in PyTorch DDP.")
        else:
            raise RuntimeError("Process group must be initialized before wrapping model.")

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
        """Run a single DDP training step."""
        self.ddp_model.train()
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            x, y = x.to(self.rank), y.to(self.rank)
            
        outputs = self.ddp_model(x)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        # Average the loss across all processes for logging
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        avg_loss = loss.item() / self.world_size
        
        return avg_loss

    def cleanup(self):
        """Clean up the distributed environment."""
        console.print(f"[bold blue][Rank {self.rank}][/bold blue] Destroying process group.")
        dist.destroy_process_group()
