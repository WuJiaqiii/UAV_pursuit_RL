#!/usr/bin/env python3
"""Measure batch-1 actor inference overhead; NOT a hardware-in-loop deadline proof."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import statistics
import time
from pathlib import Path
import numpy as np
from pursuit_rl.learning.checkpoints import load_model
from pursuit_rl.simulation.observations import observation_dimensions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--iterations', type=int, default=2000)
    p.add_argument('--out')
    a = p.parse_args()
    if a.iterations < 1 or a.warmup < 0:
        p.error('positive iterations and nonnegative warmup required')
    model, data = load_model(a.checkpoint, device=a.device)
    actor_dim, _ = observation_dimensions(data['config'])
    obs = np.zeros((1, actor_dim), np.float32)
    import torch
    def sync():
        if a.device == 'cuda':
            torch.cuda.synchronize()
    for _ in range(a.warmup):
        model.predict(obs)
    sync()
    samples = np.zeros(a.iterations)
    for i in range(a.iterations):
        sync()
        start = time.perf_counter_ns()
        action = model.predict(obs)
        sync()
        samples[i] = (time.perf_counter_ns()-start)/1e6
    result = {
      'device': a.device, 'iterations': a.iterations,
      'mean_ms': float(np.mean(samples)), 'p50_ms': float(np.percentile(samples, 50)),
      'p95_ms': float(np.percentile(samples, 95)), 'p99_ms': float(np.percentile(samples, 99)),
      'max_ms': float(np.max(samples)), 'deadline_misses_over_1ms': int((samples > 1.).sum()),
      'deadline_miss_ratio': float((samples > 1.).mean()),
      'output_dim': int(action.shape[-1]),
      'note':'Measures Python NumPy -> PyTorch -> NumPy inference only. Excludes sensing, vehicle bus, RT scheduling, and flight controller.'
    }
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2), encoding='utf8')


if __name__=='__main__':
    main()
