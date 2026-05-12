import logging
import math
import random
from collections import defaultdict

from src.config import MPCConfig
from src.machine import Machine
from src.aggregation_tree import AggregationTree

log = logging.getLogger('mpc.boruvka')



def create_level1_machines(config: MPCConfig) -> dict:
    machines = {}
    for i in range(config.machines_per_level):
        mid = (1, i)
        machines[mid] = Machine(
            mid,
            memory_limit=config.S,
            send_limit=config.level1_send_limit,
            recv_limit=config.upper_recv_limit,
        )
    log.info(
        f"[SETUP] Created {config.machines_per_level} level-1 machines | "
        f"mem={config.S} send={config.level1_send_limit} "
        f"recv={config.upper_recv_limit}"
    )
    return machines


def create_upper_level_machines(config: MPCConfig, machines: dict):
    """Create machines for levels 2..num_levels with derived limits.

    Upper-level machines hold transient aggregation data.
    Memory = indegree * WORDS_PER_UP_MSG (derived from indegree bound).
    Bandwidth = same (derived from indegree bound).
    """
    for level in range(2, config.num_levels + 1):
        for i in range(config.machines_per_level):
            mid = (level, i)
            machines[mid] = Machine(
                mid,
                memory_limit=config.upper_storage_limit,
                send_limit=config.upper_send_limit,
                recv_limit=config.upper_recv_limit,
                hard_memory=True,
            )
    log.info(
        f"[SETUP] Created upper-level machines | levels 2..{config.num_levels}, "
        f"{config.machines_per_level}/level | mem={config.upper_storage_limit} "
        f"send={config.upper_send_limit} recv={config.upper_recv_limit}"
    )


def remove_upper_level_machines(config: MPCConfig, machines: dict):
    """Remove all machines above level 1 (cleanup between phases)."""
    to_remove = [mid for mid in machines if mid[0] > 1]
    for mid in to_remove:
        del machines[mid]
    log.debug(f"[CLEANUP] Removed {len(to_remove)} upper-level machines")



def distribute_edges(edges: list, config: MPCConfig, machines: dict):
    """Distribute graph edges across level-1 machines (round-robin).

    Each edge stored as (u, v, weight, fid_u=u, fid_v=v) since initially
    every node is its own fragment.
    """
    level1_ids = sorted([mid for mid in machines if mid[0] == 1])
    for i, (u, v, w) in enumerate(edges):
        mid = level1_ids[i % len(level1_ids)]
        machines[mid].store_edge(u, v, w, fid_u=u, fid_v=v)
    log.info(
        f"[DISTRIBUTE] {len(edges)} edges -> {len(level1_ids)} level-1 machines "
        f"(~{len(edges) // len(level1_ids)} edges/machine, "
        f"max {config.edges_per_machine} capacity)"
    )



def color_fragment(fragment_id: int, seed: int) -> str:
    """Deterministically color a fragment red or blue using shared randomness."""
    rng = random.Random(hash((seed, 'color', fragment_id)) & 0xFFFFFFFF)
    return 'red' if rng.random() < 0.5 else 'blue'


def get_current_fragments(machines: dict) -> set:
    """Scan level-1 machines to find all active fragment IDs."""
    frags = set()
    for mid, machine in machines.items():
        if mid[0] != 1:
            continue
        for e in machine.get_edges():
            frags.add(e['fid_u'])
            frags.add(e['fid_v'])
    return frags


def build_level1_frag_map(machines: dict) -> dict:
    """Build mapping: level-1 machine index -> set of fragment IDs it serves.

    A machine is responsible for fragment x if it stores an edge touching x.
    This is the data-driven level-1 responsibility from the exercise sheet.
    """
    frag_map = {}
    for mid, machine in machines.items():
        if mid[0] != 1:
            continue
        frags = set()
        for e in machine.get_edges():
            frags.add(e['fid_u'])
            frags.add(e['fid_v'])
        if frags:
            frag_map[mid[1]] = frags
    return frag_map



def _flush_and_batch(machines: dict, recv_limit: int, pending_msgs: list = None):
    """Split messages into sub-round batches respecting recv_limit.

    If pending_msgs is provided, use those directly (avoids Machine.send
    overhead for intermediate forwarding). Otherwise flush all outboxes.

    Each sub-round batch guarantees no recipient receives more than
    recv_limit words, so the MPC per-round receive constraint is respected.

    Returns (sub_rounds, total_msg_count) where sub_rounds is a list:
        sub_rounds[i] = {recipient_id: (messages_list, total_words)}
    """
    if pending_msgs is not None:
        all_msgs = pending_msgs
    else:
        all_msgs = []
        for machine in machines.values():
            all_msgs.extend(machine.flush_outbox())

    if not all_msgs:
        return [], 0

    by_recipient = defaultdict(list)
    for msg in all_msgs:
        by_recipient[msg['recipient']].append(msg)

    recipient_chunks = {}
    max_chunks = 0
    for rid, msgs in by_recipient.items():
        chunks = []
        cur_msgs = []
        cur_words = 0
        for msg in msgs:
            if cur_words + msg['word_count'] > recv_limit and cur_msgs:
                chunks.append((cur_msgs, cur_words))
                cur_msgs = [msg]
                cur_words = msg['word_count']
            else:
                cur_msgs.append(msg)
                cur_words += msg['word_count']
        if cur_msgs:
            chunks.append((cur_msgs, cur_words))
        recipient_chunks[rid] = chunks
        max_chunks = max(max_chunks, len(chunks))

    sub_rounds = []
    for s in range(max_chunks):
        batch = {}
        for rid, chunks in recipient_chunks.items():
            if s < len(chunks):
                batch[rid] = chunks[s]
        sub_rounds.append(batch)

    log.debug(
        f"[BATCH] {len(all_msgs)} msgs -> {len(sub_rounds)} sub-round(s) | "
        f"recipients={len(by_recipient)} recv_limit={recv_limit}"
    )
    return sub_rounds, len(all_msgs)


def _deliver_subround(machines: dict, coordinator, batch: dict,
                      label: str, sub_idx: int, num_subs: int):
    """Deliver one sub-round batch: clear recipient inboxes, then deliver.

    Each recipient receives at most recv_limit words (guaranteed by _flush_and_batch).
    Updates coordinator round counter and log.
    """
    for rid in batch:
        if rid in machines:
            machines[rid].clear_inbox()

    total_words = 0
    total_msgs = 0
    for rid, (msgs, words) in batch.items():
        if rid not in machines:
            raise ValueError(
                f"Message to non-existent machine {rid}. "
                f"Senders: {[m['sender'] for m in msgs]}"
            )
        machines[rid].receive(msgs, words)
        total_words += words
        total_msgs += len(msgs)

    coordinator.round_counter += 1
    if sub_idx == 0:
        coordinator.logical_round_counter += 1
    sub_label = f"{label} sub={sub_idx+1}/{num_subs}" if num_subs > 1 else label
    coordinator._round_log.append({
        'round': coordinator.round_counter,
        'logical_round': coordinator.logical_round_counter,
        'label': sub_label,
        'messages_delivered': total_msgs,
        'recipients': len(batch),
        'total_words': total_words,
    })
    log.info(
        f"[COORDINATOR] ROUND {coordinator.round_counter} [{sub_label}] | "
        f"msgs={total_msgs} recipients={len(batch)} words={total_words}"
    )
    for rid, (msgs, words) in batch.items():
        log.debug(
            f"[COORDINATOR] ROUND {coordinator.round_counter} DETAIL | "
            f"to={rid} msgs={len(msgs)} words={words} "
            f"from={[m['sender'] for m in msgs]}"
        )



def _compute_top_machine_contributions(machines, tree, config, fragments, merge_decisions):
    num_levels = config.num_levels
    top_owned = defaultdict(list)
    for frag in fragments:
        top_idx = tree.get_top_machine(frag)
        top_owned[top_idx].append(frag)
    contributions = {}
    for mid in machines:
        if mid[0] != num_levels:
            continue
        owned = top_owned.get(mid[1], [])
        count = sum(1 for fid in owned if fid not in merge_decisions)
        contributions[mid[1]] = count
    return contributions


def _termination_upward_reduction(machines, num_levels, M, k, coordinator, config):
    WTERM = config.WORDS_PER_TERM_MSG
    if M <= 1:
        return
    step = 1
    while step < M:
        for j in range(0, M, k * step):
            for i in range(1, k):
                sender_j = j + i * step
                if sender_j >= M:
                    break
                sender_mid = (num_levels, sender_j)
                receiver_mid = (num_levels, j)
                if sender_mid not in machines or receiver_mid not in machines:
                    continue
                val = machines[sender_mid].get_local_data('term_partial') or 0
                machines[sender_mid].send(
                    receiver_mid, {'partial_sum': val}, word_count=WTERM
                )
        coordinator.execute_round(label=f"term_up_step{step}")
        for j in range(0, M, k * step):
            mid = (num_levels, j)
            if mid not in machines:
                continue
            machine = machines[mid]
            current = machine.get_local_data('term_partial') or 0
            for msg_item in machine.get_inbox():
                current += msg_item['payload']['partial_sum']
            machine.store_local_data('term_partial', current, word_count=WTERM)
        step *= k


def _termination_downward_broadcast(machines, num_levels, M, k, coordinator, config):
    WTERM = config.WORDS_PER_TERM_MSG
    if M <= 1:
        return
    steps = []
    s = 1
    while s < M:
        steps.append(s)
        s *= k
    for step in reversed(steps):
        for j in range(0, M, k * step):
            sender_mid = (num_levels, j)
            if sender_mid not in machines:
                continue
            val = machines[sender_mid].get_local_data('term_f_new') or 0
            for i in range(1, k):
                receiver_j = j + i * step
                if receiver_j >= M:
                    break
                receiver_mid = (num_levels, receiver_j)
                if receiver_mid in machines:
                    machines[sender_mid].send(
                        receiver_mid, {'f_new': val}, word_count=WTERM
                    )
        coordinator.execute_round(label=f"term_down_step{step}")
        for j in range(0, M, k * step):
            for i in range(1, k):
                receiver_j = j + i * step
                if receiver_j >= M:
                    break
                mid = (num_levels, receiver_j)
                if mid not in machines:
                    continue
                machine = machines[mid]
                inbox = machine.get_inbox()
                if inbox:
                    val = inbox[0]['payload']['f_new']
                    machine.store_local_data('term_f_new', val, word_count=WTERM)


def run_distributed_termination(machines, tree, coordinator, config,
                                fragments, merge_decisions) -> int:
    num_levels = config.num_levels
    M = config.machines_per_level
    k = max(2, config.S // 4)
    WTERM = config.WORDS_PER_TERM_MSG

    contributions = _compute_top_machine_contributions(
        machines, tree, config, fragments, merge_decisions
    )

    for mid, machine in machines.items():
        if mid[0] != num_levels:
            continue
        ps = contributions.get(mid[1], 0)
        machine.store_local_data('term_partial', ps, word_count=WTERM)

    _termination_upward_reduction(machines, num_levels, M, k, coordinator, config)

    root_mid = (num_levels, 0)
    f_new = 0
    if root_mid in machines:
        f_new = machines[root_mid].get_local_data('term_partial') or 0
        machines[root_mid].store_local_data('term_f_new', f_new, word_count=WTERM)

    _termination_downward_broadcast(machines, num_levels, M, k, coordinator, config)

    log.info(
        f"[TERMINATION] S/4-ary reduction | M={M} k={k} "
        f"f_new={f_new} fragments_before={len(fragments)}"
    )
    return f_new


def run_phase(machines: dict, tree: AggregationTree, coordinator,
              config: MPCConfig, seed: int) -> bool:
    """Execute one complete Boruvka phase.

    Steps:
    1. Level-1: classify edges, find local min outgoing per fragment (0 rounds)
    2. Upward pass: aggregate min edges through tree (L-1 rounds)
    3. Top-level decision: red/blue coloring determines merges (0 rounds)
    4. Downward pass through tree: each machine forwards only to children
       that contributed in the upward pass (L-1 rounds)
    5. Level-1: apply decisions -- update FIDs, mark MST edges (0 rounds)
    6. Termination: S/4-ary fragment-count reduction at top level

    The downward pass mirrors the upward path exactly: decisions flow back
    through the same tree edges used on the way up (get_children at each
    level). Collected from the batch dict directly to avoid reading stale
    upward-pass messages left in machine inboxes.

    Returns True if more than one fragment remains, False otherwise.
    """
    num_levels = tree.num_levels
    fragments = tree.fragments
    WUP = config.WORDS_PER_UP_MSG
    WDOWN = config.WORDS_PER_DOWN_MSG
    WTERM = config.WORDS_PER_TERM_MSG

    log.info(f"[PHASE] === START | fragments={len(fragments)} levels={num_levels} seed={seed} ===")

    colors = {frag: color_fragment(frag, seed) for frag in fragments}
    red_count = sum(1 for c in colors.values() if c == 'red')
    log.info(f"[PHASE] COLORS | red={red_count} blue={len(colors) - red_count}")
    log.debug(f"[PHASE] COLOR_DETAIL | {colors}")

    if num_levels == 1:
        log.info("[PHASE] SINGLE-LEVEL fast path")

        global_mins = {}
        for mid, machine in machines.items():
            if mid[0] != 1:
                continue
            for frag, result in machine.find_min_outgoing_per_fragment().items():
                if frag not in fragments or result is None:
                    continue
                u, v, w, other_fid = result
                if frag not in global_mins or w < global_mins[frag]['weight']:
                    global_mins[frag] = {
                        'fragment_id': frag, 'u': u, 'v': v,
                        'weight': w, 'other_fid': other_fid,
                    }

        had_merge = False
        for fid, ed in global_mins.items():
            my_color = colors[fid]
            other_fid = ed['other_fid']
            other_color = colors.get(other_fid, 'blue')
            if my_color == 'red' and other_color == 'blue':
                had_merge = True
                log.info(
                    f"[DECIDE] MERGE | {fid}(red) -> {other_fid}(blue) "
                    f"via ({ed['u']},{ed['v']},w={ed['weight']})"
                )
                for mid, machine in machines.items():
                    if mid[0] == 1:
                        machine.update_fids(fid, other_fid)
                        machine.mark_mst_edge(ed['u'], ed['v'], ed['weight'])

        has_inter = any(
            any(e['fid_u'] != e['fid_v'] for e in machine.get_edges())
            for mid, machine in machines.items() if mid[0] == 1
        )
        new_frags = get_current_fragments(machines)
        log.info(
            f"[PHASE] === END (single-level) | "
            f"fragments: {len(fragments)} -> {len(new_frags)} | "
            f"has_inter={has_inter} ==="
        )
        return has_inter

    log.info("[PHASE] STEP 1: Level-1 local min outgoing edge computation")

    pending_up = []

    for mid, machine in machines.items():
        if mid[0] != 1:
            continue
        min_outgoing = machine.find_min_outgoing_per_fragment()

        for frag, result in min_outgoing.items():
            if frag not in fragments:
                continue
            if result is None:
                continue

            u, v, w, other_fid = result
            msg = {'fragment_id': frag, 'u': u, 'v': v,
                   'weight': w, 'other_fid': other_fid}

            parent_idx = tree.get_parent(mid[1], frag, 1)
            pending_up.append({
                'sender': mid,
                'recipient': (2, parent_idx),
                'payload': msg,
                'word_count': WUP,
            })

    log.info("[PHASE] STEP 2: Upward pass (sub-rounded)")

    for level in range(2, num_levels + 1):
        sub_rounds, total = _flush_and_batch(
            machines, config.upper_recv_limit, pending_msgs=pending_up
        )
        pending_up = []

        if not sub_rounds:
            log.info(f"[UPWARD L{level}] No messages to deliver")
            continue

        log.info(
            f"[UPWARD L{level-1}->L{level}] {total} msgs in "
            f"{len(sub_rounds)} sub-round(s)"
        )

        for sub_idx, batch in enumerate(sub_rounds):
            _deliver_subround(
                machines, coordinator, batch,
                f"upward_L{level-1}->L{level}", sub_idx, len(sub_rounds),
            )

            for mid, machine in machines.items():
                if mid[0] != level:
                    continue
                inbox = machine.get_inbox()
                if not inbox:
                    continue

                running = machine.get_local_data('frag_mins') or {}
                for msg_item in inbox:
                    p = msg_item['payload']
                    fid = p['fragment_id']
                    if fid not in running or p['weight'] < running[fid]['weight']:
                        running[fid] = p
                machine.store_local_data(
                    'frag_mins', running, word_count=len(running) * WUP
                )

        for mid, machine in machines.items():
            if mid[0] != level:
                continue
            frag_mins = machine.get_local_data('frag_mins')
            if frag_mins:
                for fid, ed in frag_mins.items():
                    log.debug(
                        f"[UPWARD L{level}] Machine {mid} | frag={fid} "
                        f"min=({ed['u']},{ed['v']},w={ed['weight']}) "
                        f"other_fid={ed['other_fid']}"
                    )

        if level < num_levels:
            for mid, machine in machines.items():
                if mid[0] != level:
                    continue
                frag_mins = machine.get_local_data('frag_mins')
                if not frag_mins:
                    continue
                for fid, ed in frag_mins.items():
                    parent_idx = tree.get_parent(mid[1], fid, level)
                    pending_up.append({
                        'sender': mid,
                        'recipient': (level + 1, parent_idx),
                        'payload': ed,
                        'word_count': WUP,
                    })

    log.info("[PHASE] STEP 3: Merge decisions at top level")

    merge_decisions = {}

    for mid, machine in machines.items():
        if mid[0] != num_levels:
            continue
        frag_mins = machine.get_local_data('frag_mins')
        if frag_mins is None:
            continue

        for fid, ed in frag_mins.items():
            my_color = colors[fid]
            other_fid = ed['other_fid']
            other_color = colors.get(other_fid, 'blue')

            log.debug(
                f"[DECIDE] frag={fid}({my_color}) -> {other_fid}({other_color}) "
                f"via ({ed['u']},{ed['v']},w={ed['weight']})"
            )

            if my_color == 'red' and other_color == 'blue':
                merge_decisions[fid] = {
                    'old_fid': fid, 'new_fid': other_fid,
                    'u': ed['u'], 'v': ed['v'], 'weight': ed['weight'],
                }
                log.info(
                    f"[DECIDE] MERGE | {fid}(red) -> {other_fid}(blue) "
                    f"via ({ed['u']},{ed['v']},w={ed['weight']})"
                )

    log.info(f"[PHASE] STEP 3 RESULT | {len(merge_decisions)} merges decided")

    log.info("[PHASE] STEP 4: Top-level -> L1 direct decision delivery (1 round)")

    pending_down = []
    for mid, machine in machines.items():
        if mid[0] != num_levels:
            continue
        frag_mins = machine.get_local_data('frag_mins')
        if frag_mins is None:
            continue
        for fid in frag_mins:
            if fid in merge_decisions:
                payload, wc = merge_decisions[fid], WDOWN
            else:
                payload, wc = {'old_fid': fid, 'no_merge': True}, WTERM
            for l1_idx in tree.get_responsible_machines(fid, 1):
                pending_down.append({
                    'sender': mid,
                    'recipient': (1, l1_idx),
                    'payload': payload,
                    'word_count': wc,
                })

    log.info("[PHASE] STEP 5: Deliver decisions to L1 and apply")

    had_merge = False
    l1_had_merge = set()

    sub_rounds, total = _flush_and_batch(
        machines, config.upper_recv_limit, pending_msgs=pending_down
    )
    if sub_rounds:
        log.info(f"[DOWNWARD top->L1] {total} msgs in {len(sub_rounds)} sub-round(s)")
        for sub_idx, batch in enumerate(sub_rounds):
            _deliver_subround(
                machines, coordinator, batch,
                "downward_top->L1", sub_idx, len(sub_rounds),
            )
            for rid, (msgs, _wc) in batch.items():
                if rid[0] != 1:
                    continue
                machine = machines[rid]
                for msg in msgs:
                    p = msg['payload']
                    if 'new_fid' in p:
                        had_merge = True
                        l1_had_merge.add(rid)
                        machine.update_fids(p['old_fid'], p['new_fid'])
                        machine.mark_mst_edge(p['u'], p['v'], p['weight'])

    log.info(
        f"[PHASE] STEP 5 RESULT | had_merge={had_merge} "
        f"L1_machines_with_merge={len(l1_had_merge)}"
    )

    log.info("[PHASE] STEP 6: Distributed fragment-count termination (S/4-ary)")

    f_new = run_distributed_termination(
        machines, tree, coordinator, config, fragments, merge_decisions
    )

    for mid, machine in machines.items():
        if mid[0] > 1:
            machine.clear_local_data()

    new_frags = get_current_fragments(machines)
    continue_phases = f_new > 1
    log.info(
        f"[PHASE] === END | fragments: {len(fragments)} -> {len(new_frags)} "
        f"| merges={len(merge_decisions)} f_new={f_new} "
        f"continue={continue_phases} ==="
    )
    return continue_phases



def run_algorithm(n: int, edges: list, config: MPCConfig) -> tuple:
    """Run the full Boruvka MST algorithm.

    Returns (mst_edges, coordinator) where mst_edges is a list of (u, v, weight).
    """
    log.info(
        f"[ALGORITHM] === START | n={n} m={len(edges)} alpha={config.alpha} "
        f"S={config.S} M={config.M} levels={config.num_levels} "
        f"edges/machine={config.edges_per_machine} ==="
    )

    machines = create_level1_machines(config)
    coordinator = Coordinator(machines)

    distribute_edges(edges, config, machines)

    for phase in range(config.max_phases):
        fragments = get_current_fragments(machines)
        log.info(f"[ALGORITHM] PHASE {phase} | fragments={len(fragments)}")

        if len(fragments) <= 1:
            log.info("[ALGORITHM] Single fragment -- termination guard")
            break

        seed = coordinator.generate_shared_seed(phase)
        level1_frag_map = build_level1_frag_map(machines)
        tree = AggregationTree(config, fragments, seed, level1_frag_map)

        create_upper_level_machines(config, machines)
        continue_phases = run_phase(machines, tree, coordinator, config, seed)
        remove_upper_level_machines(config, machines)

        if not continue_phases:
            log.info("[ALGORITHM] Distributed termination: only one fragment remains")
            break
    else:
        log.warning(f"[ALGORITHM] Hit max_phases={config.max_phases} without converging!")

    mst_edges = []
    seen = set()
    for mid, machine in machines.items():
        if mid[0] == 1:
            for u, v, w in machine.get_mst_edges():
                key = (min(u, v), max(u, v), w)
                if key not in seen:
                    seen.add(key)
                    mst_edges.append((u, v, w))

    total_violations = {'memory': 0, 'send': 0, 'recv': 0}
    peak_stats = {'memory': 0, 'send': 0, 'recv': 0}
    for mid, machine in machines.items():
        for vtype in ('memory', 'send', 'recv'):
            total_violations[vtype] += len(machine.violations[vtype])
        peak_stats['memory'] = max(peak_stats['memory'], machine.peak_memory)
        peak_stats['send'] = max(peak_stats['send'], machine.peak_send)
        peak_stats['recv'] = max(peak_stats['recv'], machine.peak_recv)

    sub_rounds = coordinator.round_counter - coordinator.logical_round_counter
    log.info(
        f"[ALGORITHM] === DONE | "
        f"logical_rounds={coordinator.logical_round_counter} "
        f"actual_rounds={coordinator.round_counter} "
        f"(sub_rounds_overhead={sub_rounds}) "
        f"MST_edges={len(mst_edges)} (expected {n-1}) ==="
    )
    log.info(
        f"[CONSTRAINTS] Level-1 memory (hard sublinear): S={config.S} words | "
        f"violations={total_violations['memory']}"
    )
    log.info(
        f"[CONSTRAINTS] Bandwidth (uniform worst-case, asymptotic O(S)): "
        f"send_violations={total_violations['send']} "
        f"recv_violations={total_violations['recv']}"
    )
    log.info(
        f"[CONSTRAINTS] Peaks: memory={peak_stats['memory']} "
        f"send={peak_stats['send']} recv={peak_stats['recv']}"
    )

    coordinator.violations = total_violations
    coordinator.peak_stats = peak_stats

    return mst_edges, coordinator


from src.coordinator import Coordinator
