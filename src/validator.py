import logging
from dataclasses import dataclass, field

log = logging.getLogger('mpc.validator')



class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if x and y were in different components."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def num_components(self) -> int:
        return sum(1 for i in range(len(self.parent)) if self.parent[i] == i)



def kruskal_mst(n: int, edges: list) -> list:
    """Compute MST via Kruskal's algorithm.

    Args:
        n: number of nodes (0 .. n-1)
        edges: list of (u, v, weight) tuples

    Returns:
        list of (u, v, weight) MST edges (n-1 edges for a connected graph).
    """
    log.debug(f"[KRUSKAL] n={n} m={len(edges)}")
    sorted_edges = sorted(edges, key=lambda e: e[2])
    uf = UnionFind(n)
    mst = []
    for u, v, w in sorted_edges:
        if uf.union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break
    log.debug(f"[KRUSKAL] MST size={len(mst)} weight={sum(w for _,_,w in mst)}")
    return mst



@dataclass
class ValidationResult:
    correct: bool
    mst_weight: float
    expected_weight: float
    mst_edge_count: int
    expected_edge_count: int
    weight_match: bool
    size_match: bool
    is_spanning_tree: bool
    errors: list = field(default_factory=list)

    def __str__(self):
        status = "PASS" if self.correct else "FAIL"
        lines = [
            f"[VALIDATION] {status}",
            f"  edges    : {self.mst_edge_count} (expected {self.expected_edge_count})",
            f"  weight   : {self.mst_weight} (expected {self.expected_weight})",
            f"  spanning : {self.is_spanning_tree}",
        ]
        if self.errors:
            lines.append(f"  errors   : {self.errors}")
        return "\n".join(lines)



def validate_mst(n: int, edges: list, mst_edges: list) -> ValidationResult:
    """Validate mst_edges against the Kruskal reference MST.

    Checks:
      (a) mst_edges has exactly n-1 edges
      (b) mst_edges forms a spanning tree (all n nodes connected, no cycle)
      (c) total weight equals Kruskal reference weight

    Args:
        n: number of nodes
        edges: full graph edge list (u, v, weight)
        mst_edges: edges claimed to form the MST

    Returns:
        ValidationResult
    """
    log.info(f"[VALIDATOR] n={n} m={len(edges)} mst_size={len(mst_edges)}")

    errors = []

    expected_count = n - 1
    size_ok = len(mst_edges) == expected_count
    if not size_ok:
        errors.append(
            f"expected {expected_count} MST edges, got {len(mst_edges)}"
        )

    uf = UnionFind(n)
    has_cycle = False
    all_nodes = set()
    for u, v, _w in mst_edges:
        all_nodes.add(u)
        all_nodes.add(v)
        if not uf.union(u, v):
            has_cycle = True
            errors.append(f"cycle detected via edge ({u}, {v})")

    root = uf.find(0)
    isolated = [i for i in range(n) if uf.find(i) != root]
    if isolated:
        errors.append(f"nodes not in MST spanning tree: {isolated[:10]}{'...' if len(isolated)>10 else ''}")

    is_spanning = (not has_cycle) and (not isolated) and size_ok

    reference_mst = kruskal_mst(n, edges)
    expected_weight = sum(w for _, _, w in reference_mst)
    actual_weight = sum(w for _, _, w in mst_edges)
    weight_ok = (actual_weight == expected_weight)
    if not weight_ok:
        errors.append(
            f"weight mismatch: got {actual_weight}, expected {expected_weight} "
            f"(diff={actual_weight - expected_weight})"
        )

    correct = size_ok and is_spanning and weight_ok

    result = ValidationResult(
        correct=correct,
        mst_weight=actual_weight,
        expected_weight=expected_weight,
        mst_edge_count=len(mst_edges),
        expected_edge_count=expected_count,
        weight_match=weight_ok,
        size_match=size_ok,
        is_spanning_tree=is_spanning,
        errors=errors,
    )

    log.info(str(result))
    return result
