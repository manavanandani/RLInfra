import time
import yaml
import torch
import torch.nn as nn
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Local module imports
from src.training.ddp_trainer import DDPTrainer
from src.training.fsdp_trainer import FSDPTrainer
from src.training.data_pipeline import AsyncEpisodePipeline
from src.kernels.fused_attention import FusedAttention
from src.profiler.torch_profiler import PyTorchProfilerWrapper
from src.profiler.nsight_tracker import NsightTracker
from src.tracking.mlflow_client import MLflowTracker

console = Console()

class PolicyModel(nn.Module):
    """Simple policy network simulating LLM layers for test executions."""
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attn = FusedAttention()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [batch_size, seq_len]
        h = self.embedding(x)
        # Reshape to project heads for attention [batch_size, 1, seq_len, head_dim]
        # In a real transformer, we project Q, K, V separately.
        q = h.unsqueeze(1)
        k = h.unsqueeze(1)
        v = h.unsqueeze(1)
        attn_out = self.attn(q, k, v)
        # Squeeze heads and project logits
        attn_out = attn_out.squeeze(1)
        return self.linear(attn_out)

@click.group()
def main():
    """RLInfra: Distributed RL Training Infrastructure CLI."""
    pass

@main.command()
@click.option("--config-path", default="config/training_config.yaml", help="Path to config file.")
@click.option("--rank", default=0, help="Distributed process rank ID.")
@click.option("--world-size", default=1, help="Distributed total process rank size.")
@click.option("--steps", default=10, help="Total training steps to run for demo.")
def run_training(config_path: str, rank: int, world_size: int, steps: int):
    """Launches a localized/mocked distributed training run using specified configuration."""
    console.print(Panel("[bold green]RLInfra Distributed Training Platform[/bold green]\nStarting execution pipeline...", expand=False))
    
    # Load configuration file
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Extract configs
    model_cfg = cfg["model"]
    dist_cfg = cfg["distributed"]
    pipe_cfg = cfg["pipeline"]
    prof_cfg = cfg["profiler"]
    mlflow_cfg = cfg["mlflow"]
    
    # 1. Initialize MLflow Experiment Tracking
    mlflow_client = MLflowTracker(
        experiment_name=mlflow_cfg["experiment_name"],
        tracking_uri=mlflow_cfg.get("tracking_uri")
    )
    mlflow_client.start_run()
    mlflow_client.log_param("rank", rank)
    mlflow_client.log_param("world_size", world_size)
    mlflow_client.log_param("sharding_strategy", dist_cfg["sharding_strategy"])
    
    # 2. Instantiate Model
    console.print("[bold cyan][Model][/bold cyan] Initializing PolicyModel architecture...")
    model = PolicyModel(vocab_size=model_cfg["vocab_size"], hidden_dim=model_cfg["hidden_dim"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 3. Instantiate Distributed wrapper (DDP/FSDP fallback logic)
    if world_size > 1 or dist_cfg["backend"] == "gloo":
        try:
            if dist_cfg["sharding_strategy"] == "full_shard":
                trainer = FSDPTrainer(
                    rank=rank,
                    world_size=world_size,
                    model=model,
                    cpu_offload=dist_cfg["cpu_offload"],
                    backend=dist_cfg["backend"]
                )
                model = trainer.fsdp_model
            else:
                trainer = DDPTrainer(
                    rank=rank,
                    world_size=world_size,
                    model=model,
                    backend=dist_cfg["backend"]
                )
                model = trainer.ddp_model
        except Exception as e:
            console.print(f"[bold red]Failed to init distributed training wrapper:[/bold red] {e}. Running in single-device CPU mode.")
            trainer = None
    else:
        trainer = None
        
    # 4. Initialize Data Loader Pipeline
    pipeline = AsyncEpisodePipeline(
        episode_length=pipe_cfg["episode_length"],
        feature_dim=model_cfg["hidden_dim"],
        num_workers=pipe_cfg["num_workers"],
        queue_capacity=pipe_cfg["queue_capacity"]
    )
    pipeline.start()
    
    # 5. Initialize PyTorch & Nsight Profilers
    profiler = None
    if prof_cfg["enabled"]:
        profiler = PyTorchProfilerWrapper(
            export_dir=prof_cfg["export_path"],
            wait=prof_cfg["wait_steps"],
            warmup=prof_cfg["warmup_steps"],
            active=prof_cfg["active_steps"]
        )
        profiler.start()
        
    nsight = NsightTracker(enabled=True)
    
    # 6. Primary Training Loop Execution
    try:
        for step in range(1, steps + 1):
            nsight.push_range(f"training_step_{step}")
            t_start = time.perf_counter()
            
            # Fetch batch asynchronously
            nsight.push_range("data_fetch")
            obs, targets = pipeline.next_batch(pipe_cfg["batch_size"])
            nsight.pop_range() # data_fetch
            
            # Simulated token preprocessing (targets need shape matching sequence predictions)
            # Input: [batch_size, seq_len], Targets: [batch_size, seq_len]
            # Convert float inputs to integer tokens for the Embedding layer
            obs_tokens = torch.clamp((obs[:, :, 0] * 1000).long().abs(), 0, model_cfg["vocab_size"] - 1)
            target_tokens = torch.clamp((targets[:, :]).long().abs(), 0, model_cfg["vocab_size"] - 1)
            
            nsight.push_range("compute_forward_backward")
            if trainer:
                # Perform distributed backward step
                loss_val = trainer.train_step(obs_tokens, target_tokens, optimizer, criterion)
            else:
                # Local single process fallback execution
                optimizer.zero_grad()
                outputs = model(obs_tokens)
                # Reshape to calculate CrossEntropy over vocabulary predictions
                loss = criterion(outputs.view(-1, model_cfg["vocab_size"]), target_tokens.view(-1))
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
            nsight.pop_range() # compute_forward_backward
            
            # Step the profiler
            if profiler:
                profiler.step()
                
            t_end = time.perf_counter()
            step_time = t_end - t_start
            throughput = (pipe_cfg["batch_size"] * pipe_cfg["episode_length"]) / step_time
            
            # Fetch diagnostics
            stats = pipeline.get_diagnostics()
            
            # Log metrics
            mlflow_client.log_metric("loss", loss_val, step=step)
            mlflow_client.log_metric("step_time_sec", step_time, step=step)
            mlflow_client.log_metric("tokens_per_sec", throughput, step=step)
            mlflow_client.log_metric("gpu_utilization_percent", stats["simulated_gpu_utilization_percent"], step=step)
            
            if step % 2 == 0 or step == steps:
                console.print(f"[Step {step}/{steps}] Loss: {loss_val:.4f} | Time: {step_time*1000:.1f}ms | Throughput: {throughput:.1f} tok/s | GPU Util: {stats['simulated_gpu_utilization_percent']:.1f}% (Wait: {stats['average_stall_time_ms']:.1f}ms)")
                
            nsight.pop_range() # training_step
            
    finally:
        # Cleanup Resources
        pipeline.stop()
        if profiler:
            profiler.stop()
        if trainer:
            trainer.cleanup()
        mlflow_client.end_run()
        
        # Display Diagnostics Summary
        summary = pipeline.get_diagnostics()
        table = Table(title="RLInfra Optimization Diagnostics Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_row("Asynchronous Loading Status", "ENABLED")
        table.add_row("Total Batches Loaded", str(summary["total_batches_fetched"]))
        table.add_row("Average Thread Stall Delay", f"{summary['average_stall_time_ms']:.3f} ms")
        table.add_row("Optimized GPU Utilization", f"{summary['simulated_gpu_utilization_percent']:.1f} %")
        console.print(table)
        console.print("[bold green]RLInfra training run finalized successfully![/bold green]")

if __name__ == "__main__":
    main()
