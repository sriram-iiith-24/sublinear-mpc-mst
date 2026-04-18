import random


def generate_connected_graph(n: int, m: int, seed: int = None) -> list:
    if seed is not None:
        random.seed(seed)

    assert n >= 2
    max_edges = n * (n - 1) // 2
    assert n - 1 <= m <= max_edges, f"m={m} must be in [{n-1}, {max_edges}]"

    edge_set = set()

    nodes = list(range(n))
    random.shuffle(nodes)
    for i in range(1, n):
        j = random.randint(0, i - 1)
        u, v = nodes[i], nodes[j]
        edge_set.add((min(u, v), max(u, v)))

    while len(edge_set) < m:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edge_set:
            continue
        edge_set.add(key)

    edges_list = list(edge_set)
    weights = random.sample(range(1, 10 * m + 1), m)
    return [(u, v, w) for (u, v), w in zip(edges_list, weights)]


def generate_complete_graph(n: int, seed: int = None) -> list:
    """Generate K_n with distinct weights."""
    if seed is not None:
        random.seed(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    m = len(edges)
    weights = random.sample(range(1, 10 * m + 1), m)
    return [(u, v, w) for (u, v), w in zip(edges, weights)]


def generate_sparse_path(n: int, seed: int = None) -> list:
    """Generate a path graph: 0-1-2-..-(n-1) with distinct weights."""
    if seed is not None:
        random.seed(seed)
    m = n - 1
    weights = random.sample(range(1, 10 * m + 1), m)
    return [(i, i + 1, w) for i, w in enumerate(weights)]


def generate_sparse_cycle(n: int, seed: int = None) -> list:
    """Generate a cycle graph with distinct weights."""
    if seed is not None:
        random.seed(seed)
    m = n
    weights = random.sample(range(1, 10 * m + 1), m)
    edges = [(i, (i + 1) % n, w) for i, w in enumerate(weights)]
    return edges


def save_graph(n: int, edges: list, filepath: str):
    """Save graph to edge list file."""
    m = len(edges)
    with open(filepath, 'w') as f:
        f.write(f"{n} {m}\n")
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")


def load_graph(filepath: str) -> tuple:
    """Load graph from edge list file. Returns (n, edges)."""
    with open(filepath, 'r') as f:
        first_line = f.readline().strip().split()
        n = int(first_line[0])
        edges = []
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                u, v, w = int(parts[0]), int(parts[1]), int(parts[2])
                edges.append((u, v, w))
    return n, edges
