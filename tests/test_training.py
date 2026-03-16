import pytest
import torch
import torch.nn as nn
from src.training.ddp_trainer import DDPTrainer
from src.training.fsdp_trainer import FSDPTrainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

def test_ddp_trainer_init():
    model = SimpleModel()
    try:
        # Initialize locally with gloo backend
        trainer = DDPTrainer(rank=0, world_size=1, model=model, backend="gloo")
        assert trainer.world_size == 1
        assert isinstance(trainer.ddp_model, nn.parallel.DistributedDataParallel)
        trainer.cleanup()
    except Exception as e:
        pytest.fail(f"DDP initialization failed: {e}")

def test_fsdp_trainer_init():
    model = SimpleModel()
    try:
        trainer = FSDPTrainer(rank=0, world_size=1, model=model, cpu_offload=False, backend="gloo")
        assert trainer.world_size == 1
        # FSDP wrapping can be assert checked
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        assert isinstance(trainer.fsdp_model, FSDP)
        trainer.cleanup()
    except Exception as e:
        pytest.fail(f"FSDP initialization failed: {e}")
