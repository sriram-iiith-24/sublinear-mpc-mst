# Sublinear MPC MST

A simulation of Borůvka's Minimum Spanning Tree (MST) algorithm in the Massively Parallel Computation (MPC) model with **strongly sublinear memory** per machine (`S = n^α` words, `α ∈ (0, 1)`).

Based on Exercise Sheet 12 (Fabian Kuhn) and the MPC lecture slides.

---

## Algorithm Overview

The algorithm runs iterative Borůvka phases. Each phase:

1. **Level-1 local min** — each machine finds the minimum outgoing edge per fragment (0 rounds)
2. **Upward pass** — aggregate minimums up a multi-level tree (L−1 rounds)
3. **Merge decision** — top-level machines apply red/blue coloring; red→blue merges selected (0 rounds)
4. **Downward delivery** — decisions sent directly from top to all responsible L1 machines (1 round)
5. **Apply at L1** — update fragment IDs, mark MST edges (0 rounds)
6. **Distributed termination** — S/4-ary upward reduction + downward broadcast of fragment count

Terminates when only one fragment remains. Correctness verified against Kruskal's reference MST.

---

## Key Design Decisions

| # | Decision | Effect |
|---|---|---|
| 1 | **Uncapped tree depth** (`⌈log_{S/4}(m)⌉`, no `4/α` cap) | Eliminates L_top concentration spike at low α |
| 2 | **Strict O(S) bandwidth limits** (recv=10·S, send/mem=20·S) | Provably sublinear per-round bandwidth |
| 3 | **Two-choice parent hashing** in aggregation tree | Reduces per-level max recv by 10–20% at α≥0.5 |
| 4 | **Distributed S/4-ary termination** at top level | No centralized coordinator for termination |
| 5 | **Direct top→L1 downward** (not tree-shaped) | 1 round, 2m total messages, O(S) per L1 machine |

---

## Parameters (all derived from `n`, `m`, `α`)

| Parameter | Formula | Meaning |
|---|---|---|
| `S` | `⌊n^α⌋` | Words of memory per machine |
| `M` | `⌈m / ⌊S/5⌋⌉` | Machines per level |
| `L` | `⌈log_{S/4}(m)⌉` | Aggregation tree depth |
| `recv_limit` | `10·S` words | Per-round receive cap (hard sub-round trigger) |
| `max_phases` | `10·log₂(n)` | Safety bound on outer Borůvka loop |

---

## Usage

### Single run
```bash
python main.py --n 200 --alpha 0.5
python main.py --n 200 --m 400 --alpha 0.5   # custom edge count
python main.py --n 100 --alpha 0.3 --log-level DEBUG
```

### Graph types
```bash
python main.py --n 50 --complete              # complete graph K_n
python main.py --n 100 --path                 # path graph
python main.py --n 100 --cycle                # cycle graph
python main.py --load-graph graph.txt         # from edge-list file
```

### Correctness benchmark (9 cases, α=0.3/0.5/0.7, n=50..5000)
```bash
python main.py --benchmark
```

### Alpha sweep (varies α from 0.3–0.8 at fixed n)
```bash
python main.py --alpha-sweep --n 500
python main.py --alpha-sweep --n 1000
```

### Per-level bandwidth probe (fast, phase-1 only)
```bash
python bandwidth_probe.py --n 500 --alpha 0.5
python bandwidth_probe.py --n 200 --alpha 0.3 --phases 1
python bandwidth_probe.py --n 1000 --alpha 0.5 --verbose   # per-machine detail
```

### Full load sweep across (n, α) cases
```bash
python load_sweep.py 2>/dev/null
# Edit `cases` list in load_sweep.py (line 55) to change test parameters
```
