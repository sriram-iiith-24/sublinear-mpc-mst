import logging

log = logging.getLogger('mpc.machine')


class MemoryViolation(Exception):
    pass


class BandwidthViolation(Exception):
    pass


class Machine:

    def __init__(self, machine_id: tuple, memory_limit: int,
                 send_limit: int, recv_limit: int, hard_memory: bool = True):
        """
        Args:
            memory_limit: max words of stored data.
            send_limit: max words sent per round (from indegree analysis).
            recv_limit: max words received per round (from indegree analysis).
            hard_memory: if True, memory violations raise (level-1 edge storage).
                         if False, they are tracked but don't crash (upper-level
                         transient data where bounds are probabilistic).
        """
        self.id = machine_id
        self.memory_limit = memory_limit
        self.send_limit = send_limit
        self.recv_limit = recv_limit
        self.hard_memory = hard_memory

        self._edges = []
        self._local_data = {}
        self._local_data_words = 0

        self._outbox = []
        self._outbox_words = 0
        self._inbox = []
        self._inbox_words = 0

        self.violations = {
            'memory': [],
            'send': [],
            'recv': [],
        }
        self.peak_memory = 0
        self.peak_send = 0
        self.peak_recv = 0

        log.debug(
            f"[MACHINE {self.id}] Created | mem={memory_limit} "
            f"send={send_limit} recv={recv_limit} hard_mem={hard_memory}"
        )


    def words_stored(self) -> int:
        return len(self._edges) * 5 + self._local_data_words

    def words_sent(self) -> int:
        return self._outbox_words

    def words_received(self) -> int:
        return self._inbox_words


    def store_edge(self, u: int, v: int, weight: int, fid_u: int, fid_v: int):
        new_words = self.words_stored() + 5
        if new_words > self.memory_limit:
            raise MemoryViolation(
                f"Machine {self.id}: storing edge would use {new_words} words "
                f"(limit {self.memory_limit})"
            )
        self._edges.append({
            'u': u, 'v': v, 'weight': weight,
            'fid_u': fid_u, 'fid_v': fid_v,
            'is_mst': False,
        })
        log.debug(
            f"[MACHINE {self.id}] STORE_EDGE | edge=({u},{v},w={weight}) "
            f"fids=({fid_u},{fid_v}) | words={new_words}/{self.memory_limit}"
        )

    def get_edges(self) -> list:
        return self._edges

    def get_mst_edges(self) -> list:
        return [(e['u'], e['v'], e['weight']) for e in self._edges if e['is_mst']]

    def mark_mst_edge(self, u: int, v: int, weight: int):
        for e in self._edges:
            if e['weight'] == weight and (
                (e['u'] == u and e['v'] == v) or (e['u'] == v and e['v'] == u)
            ):
                e['is_mst'] = True
                log.info(
                    f"[MACHINE {self.id}] MARK_MST | edge=({u},{v},w={weight})"
                )
                return
        log.warning(
            f"[MACHINE {self.id}] MARK_MST_MISS | edge=({u},{v},w={weight}) not found"
        )

    def update_fids(self, old_fid: int, new_fid: int):
        count = 0
        for e in self._edges:
            if e['fid_u'] == old_fid:
                e['fid_u'] = new_fid
                count += 1
            if e['fid_v'] == old_fid:
                e['fid_v'] = new_fid
                count += 1
        if count > 0:
            log.debug(
                f"[MACHINE {self.id}] UPDATE_FIDS | {old_fid}->{new_fid} "
                f"updated {count} fid entries"
            )

    def classify_edges(self):
        """Return dict: fragment_id -> list of inter-fragment edges for that fragment."""
        result = {}
        for e in self._edges:
            if e['fid_u'] != e['fid_v']:
                result.setdefault(e['fid_u'], []).append(e)
                result.setdefault(e['fid_v'], []).append(e)
        inter_count = sum(len(v) for v in result.values()) // 2
        intra_count = len(self._edges) - inter_count
        log.debug(
            f"[MACHINE {self.id}] CLASSIFY | inter={inter_count} intra={intra_count} "
            f"fragments={sorted(result.keys())}"
        )
        return result

    def find_min_outgoing_per_fragment(self):
        """For each fragment on this machine, find the minimum-weight inter-fragment edge.

        Returns dict: fragment_id -> (u, v, weight, other_fid) or None.
        The 'other_fid' is carried in upward messages so the top-level machine
        can make merge decisions without accessing level-1 data.
        """
        inter = self.classify_edges()
        result = {}
        all_frags = set()
        for e in self._edges:
            all_frags.add(e['fid_u'])
            all_frags.add(e['fid_v'])

        for fid in all_frags:
            if fid not in inter:
                result[fid] = None
                continue
            best = None
            for e in inter[fid]:
                other_fid = e['fid_v'] if e['fid_u'] == fid else e['fid_u']
                candidate = (e['weight'], e['u'], e['v'], other_fid)
                if best is None or candidate < best:
                    best = candidate
            if best:
                w, u, v, other_fid = best
                result[fid] = (u, v, w, other_fid)
                log.debug(
                    f"[MACHINE {self.id}] MIN_OUTGOING | frag={fid} -> "
                    f"edge=({u},{v},w={w}) other_fid={other_fid}"
                )
            else:
                result[fid] = None
        return result


    def store_local_data(self, key: str, data, word_count: int):
        old_wc = self._local_data[key][1] if key in self._local_data else 0
        new_total = self.words_stored() - old_wc + word_count
        self.peak_memory = max(self.peak_memory, new_total)
        if new_total > self.memory_limit:
            self.violations['memory'].append((new_total, self.memory_limit))
            msg = (f"Machine {self.id}: storing local data would use {new_total} words "
                   f"(limit {self.memory_limit})")
            if self.hard_memory:
                raise MemoryViolation(msg)
            else:
                log.debug(f"[MACHINE {self.id}] MEM_EXCEED | {msg}")
        if key in self._local_data:
            self._local_data_words -= self._local_data[key][1]
        self._local_data[key] = (data, word_count)
        self._local_data_words += word_count
        log.debug(
            f"[MACHINE {self.id}] STORE_DATA | key='{key}' "
            f"words={word_count} | total={new_total}/{self.memory_limit}"
        )

    def get_local_data(self, key: str):
        if key in self._local_data:
            return self._local_data[key][0]
        return None

    def clear_local_data(self):
        self._local_data.clear()
        self._local_data_words = 0


    def send(self, recipient_id: tuple, payload: dict, word_count: int):
        new_sent = self._outbox_words + word_count
        self.peak_send = max(self.peak_send, new_sent)
        if new_sent > self.send_limit:
            self.violations['send'].append((new_sent, self.send_limit))
            log.debug(
                f"[MACHINE {self.id}] SEND_EXCEED | {new_sent} words "
                f"(limit {self.send_limit})"
            )
        self._outbox.append({
            'sender': self.id,
            'recipient': recipient_id,
            'payload': payload,
            'word_count': word_count,
        })
        self._outbox_words += word_count
        log.debug(
            f"[MACHINE {self.id}] SEND | to={recipient_id} "
            f"words={word_count} payload={payload} | total_sent={new_sent}/{self.send_limit}"
        )

    def flush_outbox(self) -> list:
        outbox = self._outbox
        self._outbox = []
        self._outbox_words = 0
        return outbox

    def receive(self, messages: list, total_words: int):
        self.peak_recv = max(self.peak_recv, total_words)
        if total_words > self.recv_limit:
            self.violations['recv'].append((total_words, self.recv_limit))
            log.debug(
                f"[MACHINE {self.id}] RECV_EXCEED | {total_words} words "
                f"(limit {self.recv_limit})"
            )
        self._inbox = messages
        self._inbox_words = total_words
        log.debug(
            f"[MACHINE {self.id}] RECEIVE | msgs={len(messages)} "
            f"words={total_words}/{self.recv_limit}"
        )

    def get_inbox(self) -> list:
        return self._inbox

    def clear_inbox(self):
        self._inbox = []
        self._inbox_words = 0
