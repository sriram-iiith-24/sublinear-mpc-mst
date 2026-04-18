import logging
import math
import random

log = logging.getLogger('mpc.tree')


class AggregationTree:

    def __init__(self, config, fragments: set, seed: int,
                 level1_frag_map: dict):
        """
        Args:
            config: MPCConfig
            fragments: set of active fragment IDs this phase
            seed: shared random seed for deterministic construction
            level1_frag_map: dict mapping level-1 machine index -> set of
                fragment IDs that machine is responsible for (data-driven).
                Built by scanning each machine's edges.
        """
        self.config = config
        self.fragments = fragments
        self.seed = seed
        self.num_levels = config.num_levels
        self.machines_per_level = config.machines_per_level
        self.S = config.S
        self.m = config.m

        self._responsible = {}
        self._parent = {}
        self._children = {}

        self._level1_frag_map = level1_frag_map
        self._build()

        total_l1_pairs = sum(len(frags) for frags in level1_frag_map.values())
        log.info(
            f"[TREE] Built | fragments={len(fragments)} levels={self.num_levels} "
            f"M={self.machines_per_level} seed={seed} "
            f"L1_pairs={total_l1_pairs}"
        )
        for level in range(1, self.num_levels + 1):
            sample_frag = next(iter(fragments))
            r = len(self._responsible[level].get(sample_frag, []))
            log.info(f"[TREE] Level {level} | R(sample_frag,{level})={r}")

    def _build(self):
        for level in range(1, self.num_levels + 1):
            self._responsible[level] = {}
            if level < self.num_levels:
                self._parent[level] = {}
            if level > 1:
                self._children[level] = {}

        for machine_idx, frags in self._level1_frag_map.items():
            for frag in frags:
                if frag in self.fragments:
                    self._responsible[1].setdefault(frag, []).append(machine_idx)

        for frag in self._responsible[1]:
            self._responsible[1][frag].sort()

        base = self.S / 4 if self.S > 4 else 1
        for frag in self.fragments:
            for level in range(2, self.num_levels + 1):
                if level == self.num_levels:
                    r = 1
                else:
                    r = max(1, math.ceil(self.m / (base ** level)))
                    r = min(r, self.machines_per_level)

                rng = random.Random(self._subset_seed(frag, level))
                pool = list(range(self.machines_per_level))
                responsible = sorted(rng.sample(pool, min(r, len(pool))))
                self._responsible[level][frag] = responsible

        for frag in self.fragments:
            for level in range(1, self.num_levels):
                self._parent[level].setdefault(frag, {})
                next_responsible = self._responsible[level + 1][frag]

                self._children.setdefault(level + 1, {})
                self._children[level + 1].setdefault(frag, {idx: [] for idx in next_responsible})

                curr_responsible = self._responsible[level].get(frag, [])
                for machine_idx in curr_responsible:
                    rng = random.Random(self._parent_seed(frag, level, machine_idx))
                    parent = rng.choice(next_responsible)
                    self._parent[level][frag][machine_idx] = parent
                    self._children[level + 1][frag][parent].append(machine_idx)

            log.debug(
                f"[TREE] ASSIGN frag={frag} | L1_machines={len(self._responsible[1].get(frag, []))} "
                f"top={self._responsible[self.num_levels][frag][0]}"
            )

    def _subset_seed(self, frag: int, level: int) -> int:
        return hash((self.seed, 'subset', frag, level)) & 0xFFFFFFFF

    def _parent_seed(self, frag: int, level: int, machine_idx: int) -> int:
        return hash((self.seed, 'parent', frag, level, machine_idx)) & 0xFFFFFFFF

    def get_responsible_machines(self, fragment: int, level: int) -> list:
        return self._responsible.get(level, {}).get(fragment, [])

    def get_parent(self, machine_idx: int, fragment: int, level: int) -> int:
        return self._parent[level][fragment][machine_idx]

    def get_children(self, machine_idx: int, fragment: int, level: int) -> list:
        if level <= 1:
            return []
        return self._children.get(level, {}).get(fragment, {}).get(machine_idx, [])

    def get_top_machine(self, fragment: int) -> int:
        return self._responsible[self.num_levels][fragment][0]
