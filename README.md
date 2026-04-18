# Sublinear MPC MST

A simulation of Boruvka's minimum spanning tree (MST) algorithm in the Massively Parallel Computation (MPC) model with sublinear memory per machine.

## Key Features

- **Sublinear memory**: `S = n^α` words per machine, where `α ∈ (0, 1)`
- **Boruvka's algorithm**: Fragment-based MST construction with aggregation tree communication
- **Distributed termination**: OR-aggregated has_inter_fragment_edge check (no centralized decision)
- **Sub-round mechanism**: Automatic fallback for bandwidth violations (Chernoff tail events)
- **Benchmarking**: Alpha sweeps and scalability tests

## Usage

```bash
python main.py --n 100 --m 300 --alpha 0.5
python main.py --benchmark
python main.py --alpha-sweep
```

## Theory

Per-machine load at level `i`:
```
E[load_i] = f * (4/S)^(i-2)
```

Peak: `4n/S` at level 3, phase 1. With Chernoff 2× safety:
```
max_indegree = 8n/S messages
recv_limit = 40n/S words
```

Sublinearity holds for all `α ∈ (0, 1)`:
- `α ≥ 1/2`: Strictly `O(S)` per machine
- `α < 1/2`: `O(n^(1-α))` per machine (still sublinear in n)
