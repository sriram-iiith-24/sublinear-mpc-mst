import math


class MPCConfig:

    WORDS_PER_EDGE = 5
    WORDS_PER_UP_MSG = 5
    WORDS_PER_DOWN_MSG = 5
    WORDS_PER_TERM_MSG = 1

    def __init__(self, n: int, m: int, alpha: float = 0.5):
        assert 0 < alpha < 1, f"alpha must be in (0,1), got {alpha}"
        assert n >= 2, f"need at least 2 nodes, got {n}"
        assert m >= n - 1, f"need at least n-1 edges for connectivity, got {m}"

        self.n = n
        self.m = m
        self.alpha = alpha

        self.S = max(math.floor(n ** alpha), self.WORDS_PER_EDGE)

        self.edges_per_machine = self.S // self.WORDS_PER_EDGE

        self.machines_per_level = math.ceil(m / self.edges_per_machine)
        self.M = self.machines_per_level

        base = self.S / 4
        if base <= 1:
            self.num_levels = 1
        else:
            raw_levels = math.ceil(math.log(m, base)) if m > 1 else 1
            max_levels = math.ceil(4 / alpha)
            self.num_levels = max(1, min(raw_levels, max_levels))

        self.max_indegree = 2 * n

        self.threshold_level = math.ceil(1.0 / alpha + 1)

        self.level1_send_limit = 2 * self.edges_per_machine * self.WORDS_PER_UP_MSG

        self.upper_recv_limit = self.max_indegree * self.WORDS_PER_UP_MSG
        self.upper_send_limit = self.max_indegree * self.WORDS_PER_UP_MSG
        self.upper_storage_limit = self.max_indegree * self.WORDS_PER_UP_MSG

        self.max_phases = math.ceil(10 * math.log2(n))

    def responsible_count(self, level: int) -> int:
        """R(x, i) = m / (S/4)^i -- number of machines responsible per fragment.

        At the top level, this is forced to 1.
        At other levels, ceil is used (exercise sheet ignores rounding).
        """
        if level == self.num_levels:
            return 1
        base = self.S / 4
        r = self.m / (base ** level)
        return min(max(1, math.ceil(r)), self.machines_per_level)

    def __repr__(self):
        return (f"MPCConfig(n={self.n}, m={self.m}, alpha={self.alpha}, S={self.S}, "
                f"M={self.M}, levels={self.num_levels}, "
                f"threshold_level={self.threshold_level}, "
                f"max_indegree={self.max_indegree}, "
                f"edges/machine={self.edges_per_machine})")
