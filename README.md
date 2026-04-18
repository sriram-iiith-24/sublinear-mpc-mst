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
