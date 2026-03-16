import pytest
import time
from src.training.data_pipeline import AsyncEpisodePipeline

def test_async_data_pipeline():
    # Instantiate with small dimensions
    pipeline = AsyncEpisodePipeline(episode_length=32, feature_dim=64, num_workers=2, queue_capacity=4)
    
    try:
        pipeline.start()
        time.sleep(0.5) # Give background workers time to buffer
        
        # Load a few batches
        for _ in range(3):
            obs, actions = pipeline.next_batch(batch_size=2)
            assert obs.shape == (2, 32, 64)
            assert actions.shape == (2, 32)
            
        diagnostics = pipeline.get_diagnostics()
        assert diagnostics["total_batches_fetched"] == 3
        assert "average_stall_time_ms" in diagnostics
        assert diagnostics["simulated_gpu_utilization_percent"] > 0
        
    finally:
        pipeline.stop()
