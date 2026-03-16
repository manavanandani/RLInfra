import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from rich.console import Console

console = Console()

# Patch PyTorch FSDP issue on macOS (MPS backend missing current_device)
if hasattr(torch, "mps") and not hasattr(torch.mps, "current_device"):
    torch.mps.current_device = lambda: 0

class FSDPTrainer:
    """Fully Sharded Data Parallel (FSDP) trainer for large models."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        model: torch.nn.Module,
        cpu_offload: bool = True,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        backend: str = "gloo"
    ):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.cpu_offload = cpu_offload
        self.sharding_strategy = sharding_strategy
        self.model = model
        
        self._setup_process_group()
        self._wrap_model()

    def _setup_process_group(self):
        """Initialize process group for distributed scaling."""
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
        
        console.print(f"[bold blue][Rank {self.rank}][/bold blue] Initializing process group for FSDP.")
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size
            )
        console.print(f"[bold green][Rank {self.rank}][/bold green] Process group initialized for FSDP.")

    def _wrap_model(self):
        """Wrap model with FSDP, configuring offload and precision policies."""
        if not dist.is_initialized():
            raise RuntimeError("Process group must be initialized before FSDP wrapping.")

        # Size-based wrapping policy (e.g., wrap modules larger than 2M params)
        auto_wrap_policy = size_based_auto_wrap_policy

        # Offload parameters to CPU to bypass GPU memory limits if needed
        offload_config = CPUOffload(offload_params=self.cpu_offload)

        # Mixed precision setup (FP16 or BF16)
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
            device = torch.device(f"cuda:{self.rank}")
            self.fsdp_model = FSDP(
                self.model.to(device),
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=offload_config,
                sharding_strategy=self.sharding_strategy,
                mixed_precision=mixed_precision,
                device_id=torch.cuda.current_device()
            )
        else:
            # CPU Fallback configuration
            self.fsdp_model = FSDP(
                self.model,
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=offload_config,
                sharding_strategy=self.sharding_strategy
            )
        console.print(f"[bold green][Rank {self.rank}][/bold green] Model successfully sharded and wrapped in FSDP.")

    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
        """Execute a sharded training step."""
        self.fsdp_model.train()
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            x, y = x.to(device), y.to(device)
            
        outputs = self.fsdp_model(x)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        # Average sharded loss
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        avg_loss = loss.item() / self.world_size
        
        return avg_loss

    def cleanup(self):
        """Cleanup process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
