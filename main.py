import argparse
import logging
import sys
import time

from src.config import MPCConfig
from src.graph_generator import (
    generate_connected_graph,
    generate_complete_graph,
    generate_sparse_path,
    generate_sparse_cycle,
    save_graph,
    load_graph,
)
from src.boruvka import run_algorithm
from src.validator import validate_mst



def setup_logging(level: str):
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(name)-20s %(levelname)-8s %(message)s"
    logging.basicConfig(level=numeric, format=fmt, stream=sys.stdout)



def run_single(n: int, edges: list, alpha: float, label: str = ""):
    config = MPCConfig(n, len(edges), alpha)
    log = logging.getLogger('mpc.main')

    log.info(f"{'='*60}")
    log.info(f"RUN  {label or f'n={n} m={len(edges)} alpha={alpha}'}")
    log.info(f"CONFIG  {config}")
    log.info(f"{'='*60}")

    t0 = time.perf_counter()
    mst_edges, coordinator = run_algorithm(n, edges, config)
    elapsed = time.perf_counter() - t0

    result = validate_mst(n, edges, mst_edges)

    status = "PASS" if result.correct else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {status}  |  {label or f'n={n} m={len(edges)}'}  |  alpha={alpha}")
    print(f"  S={config.S}  M={config.M}  levels={config.num_levels}  "
          f"thr_level={config.threshold_level}  "
          f"edges/machine={config.edges_per_machine}")
    sub_rounds = coordinator.round_counter - coordinator.logical_round_counter
    print(f"  logical_rounds={coordinator.logical_round_counter}  "
          f"actual_rounds={coordinator.round_counter}  "
          f"sub_round_overhead={sub_rounds}  elapsed={elapsed:.3f}s")
    print(f"  MST_edges={result.mst_edge_count}/{result.expected_edge_count}  "
          f"weight={result.mst_weight}  expected={result.expected_weight}")
    if result.errors:
        for err in result.errors:
            print(f"  ERROR: {err}")
    print(f"{'='*60}\n")

    return result



BENCHMARK_CASES = [
    (50,  3,   0.5, "tiny n=50"),
    (100, 3,   0.5, "small n=100"),
    (200, 3,   0.5, "n=200"),
    (500, 3,   0.5, "n=500"),
    (1000, 3,  0.5, "n=1000"),
    (2000, 3,  0.5, "n=2000"),
    (5000, 3,  0.5, "n=5000"),
    (100, 3,   0.3, "alpha=0.3 n=100"),
    (100, 3,   0.7, "alpha=0.7 n=100"),
]


def run_benchmark(seed: int = 42):
    all_pass = True
    results = []
    for n, mult, alpha, label in BENCHMARK_CASES:
        m = min(mult * n, n * (n - 1) // 2)
        edges = generate_connected_graph(n, m, seed=seed)
        result = run_single(n, edges, alpha, label=label)
        results.append((label, result.correct, result.mst_weight, result.expected_weight))
        if not result.correct:
            all_pass = False

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for label, ok, got, exp in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label}  weight={got}/{exp}")
    print("="*60)
    print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print("="*60)
    return all_pass



ALPHA_SWEEP_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def run_alpha_sweep(n: int = 500, seed: int = 42):
    """Run fixed n across alpha values; report violations and sub-round trends."""
    m = min(3 * n, n * (n - 1) // 2)
    edges = generate_connected_graph(n, m, seed=seed)

    w = 78
    print(f"\n{'='*w}")
    print(f"ALPHA SWEEP | n={n} m={m} seed={seed}")
    print(f"{'='*w}")
    print(
        f"{'alpha':>6} {'S':>6} {'thr':>4} {'lvls':>5} "
        f"{'logical':>8} {'actual':>7} {'sub_ovhd':>9} "
        f"{'recv_viol':>10} {'send_viol':>10} {'ok':>5}"
    )
    print(f"{'-'*w}")

    all_pass = True
    for alpha in ALPHA_SWEEP_VALUES:
        config = MPCConfig(n, m, alpha)
        mst_edges, coordinator = run_algorithm(n, edges, config)
        result = validate_mst(n, edges, mst_edges)

        recv_viol = coordinator.violations.get('recv', 0)
        send_viol = coordinator.violations.get('send', 0)
        sub_overhead = coordinator.round_counter - coordinator.logical_round_counter
        ok = "PASS" if result.correct else "FAIL"
        if not result.correct:
            all_pass = False

        print(
            f"{alpha:>6.1f} {config.S:>6} {config.threshold_level:>4} "
            f"{config.num_levels:>5} {coordinator.logical_round_counter:>8} "
            f"{coordinator.round_counter:>7} {sub_overhead:>9} "
            f"{recv_viol:>10} {send_viol:>10} {ok:>5}"
        )

    print(f"{'='*w}")
    print(f"  thr = threshold_level = ceil(1/alpha+1): level where S/2 bound holds")
    print(f"  sub_ovhd = actual_rounds - logical_rounds (sub-round overhead)")
    print(f"{'='*w}\n")
    return all_pass



def build_parser():
    p = argparse.ArgumentParser(
        description="Sublinear MPC MST simulation (Boruvka / aggregation tree)"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--load-graph", metavar="FILE",
                     help="Load graph from edge-list file")
    src.add_argument("--complete", action="store_true",
                     help="Generate complete graph K_n")
    src.add_argument("--path", action="store_true",
                     help="Generate path graph 0-1-2-..")
    src.add_argument("--cycle", action="store_true",
                     help="Generate cycle graph")
    src.add_argument("--benchmark", action="store_true",
                     help="Run full benchmark suite")
    src.add_argument("--alpha-sweep", action="store_true",
                     help="Sweep alpha in {0.3..0.8} at fixed n, show violation trends")

    p.add_argument("--n", type=int, default=100,
                   help="Number of nodes (default: 100)")
    p.add_argument("--m", type=int, default=None,
                   help="Number of edges (default: 3*n for random graphs)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="MPC memory exponent S=n^alpha (default: 0.5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--save-graph", metavar="FILE",
                   help="Save generated graph to file before running")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.benchmark:
        ok = run_benchmark(seed=args.seed)
        sys.exit(0 if ok else 1)

    if args.alpha_sweep:
        ok = run_alpha_sweep(n=args.n, seed=args.seed)
        sys.exit(0 if ok else 1)

    if args.load_graph:
        n, edges = load_graph(args.load_graph)
        label = f"file={args.load_graph}"
    elif args.complete:
        n = args.n
        edges = generate_complete_graph(n, seed=args.seed)
        label = f"complete K_{n}"
    elif args.path:
        n = args.n
        edges = generate_sparse_path(n, seed=args.seed)
        label = f"path n={n}"
    elif args.cycle:
        n = args.n
        edges = generate_sparse_cycle(n, seed=args.seed)
        label = f"cycle n={n}"
    else:
        n = args.n
        m = args.m if args.m is not None else min(3 * n, n * (n - 1) // 2)
        edges = generate_connected_graph(n, m, seed=args.seed)
        label = f"random n={n} m={m}"

    if args.save_graph:
        save_graph(n, edges, args.save_graph)
        print(f"Graph saved to {args.save_graph}")

    result = run_single(n, edges, args.alpha, label=label)
    sys.exit(0 if result.correct else 1)


if __name__ == '__main__':
    main()
