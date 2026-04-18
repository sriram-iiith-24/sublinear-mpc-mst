import logging
import random
from collections import defaultdict

log = logging.getLogger('mpc.coordinator')


class Coordinator:

    def __init__(self, machines: dict):
        """
        Args:
            machines: dict mapping machine_id (level, index) -> Machine object.
                      Live reference -- boruvka logic can add/remove machines.
        """
        self._machines = machines
        self.round_counter = 0
        self.logical_round_counter = 0
        self._round_log = []
        self._base_seed = random.randint(0, 2**32 - 1)
        log.info(
            f"[COORDINATOR] Initialized | machines={len(machines)} "
            f"base_seed={self._base_seed}"
        )

    def execute_round(self, label: str = ""):
        """Execute one synchronous communication round."""
        for machine in self._machines.values():
            machine.clear_inbox()

        all_messages = []
        for machine in self._machines.values():
            outbox = machine.flush_outbox()
            all_messages.extend(outbox)

        by_recipient = defaultdict(list)
        words_per_recipient = defaultdict(int)
        for msg in all_messages:
            rid = msg['recipient']
            by_recipient[rid].append(msg)
            words_per_recipient[rid] += msg['word_count']

        for rid, messages in by_recipient.items():
            total_words = words_per_recipient[rid]
            if rid not in self._machines:
                raise ValueError(
                    f"Message addressed to non-existent machine {rid}. "
                    f"Sender(s): {[m['sender'] for m in messages]}"
                )
            self._machines[rid].receive(messages, total_words)

        self.round_counter += 1
        self.logical_round_counter += 1
        round_label = f" [{label}]" if label else ""
        self._round_log.append({
            'round': self.round_counter,
            'logical_round': self.logical_round_counter,
            'label': label,
            'messages_delivered': len(all_messages),
            'recipients': len(by_recipient),
            'total_words': sum(words_per_recipient.values()),
        })
        log.info(
            f"[COORDINATOR] ROUND {self.round_counter}{round_label} | "
            f"msgs={len(all_messages)} recipients={len(by_recipient)} "
            f"total_words={sum(words_per_recipient.values())}"
        )
        for rid, msgs in by_recipient.items():
            log.debug(
                f"[COORDINATOR] ROUND {self.round_counter} DETAIL | "
                f"to={rid} msgs={len(msgs)} words={words_per_recipient[rid]} "
                f"from={[m['sender'] for m in msgs]}"
            )

    def generate_shared_seed(self, phase: int) -> int:
        """Deterministic seed per phase for aggregation tree construction."""
        seed = hash((self._base_seed, phase)) & 0xFFFFFFFF
        log.info(f"[COORDINATOR] SEED | phase={phase} seed={seed}")
        return seed

    def get_round_log(self) -> list:
        return self._round_log
