import torch
import numpy as np
from collections import deque
from math import comb
import random
from typing import Literal

def set_random_seed(seed=42, device = 'cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True     
        torch.backends.cudnn.benchmark = False
        
def get_anneal_ramp_param(epoch, ramp_start, ramp_end, max_param, method: Literal['cosine', 'linear'] = 'linear'):
    t = np.clip((epoch - ramp_start) / (ramp_end - ramp_start), a_min = 0, a_max = 1) if ramp_end != ramp_start else 1
    if method == 'linear':
        return max_param * t
    elif method == 'cosine':
        return max_param * 1/2 * (1 - np.cos(np.pi * t))
    else:
        raise ValueError(f"Annealing method must be one of ['cosine', 'linear']. Got {method}")


# Vibe coded ways to get minimum coalition iterations for N players in coalition game
def hopcroft_karp(adj):
    left_nodes = list(adj)
    right_nodes = {v for nbrs in adj.values() for v in nbrs}

    pair_u = {u: None for u in left_nodes}
    pair_v = {v: None for v in right_nodes}
    dist = {}

    INF = 10**18

    def bfs():
        q = deque()
        found_free_right = False

        for u in left_nodes:
            if pair_u[u] is None:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF

        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu is None:
                    found_free_right = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    q.append(pu)

        return found_free_right

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu is None or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True

        dist[u] = INF
        return False

    while bfs():
        for u in left_nodes:
            if pair_u[u] is None:
                dfs(u)

    return {u: v for u, v in pair_u.items() if v is not None}

def minimal_ablation_permutations(players):
    """
    Returns the minimum number of ablation input sequences.

    Each returned sequence represents one chain of coalitions.
    The prefixes across all sequences cover every nonempty coalition exactly once.
    The empty coalition should be handled separately.
    """
    n = len(players)

    subsets_by_rank = [[] for _ in range(n + 1)]
    for mask in range(1, 1 << n):  # exclude empty coalition
        subsets_by_rank[mask.bit_count()].append(mask)

    successor = {}
    predecessor = {}

    # Match rank k to rank k+1 among nonempty coalitions.
    for k in range(1, n):
        adj = {}

        for mask in subsets_by_rank[k]:
            supersets = []
            for i in range(n):
                if not (mask >> i) & 1:
                    supersets.append(mask | (1 << i))
            adj[mask] = supersets

        matching = hopcroft_karp(adj)

        for u, v in matching.items():
            successor[u] = v
            predecessor[v] = u

    # Chain starts are nonempty masks with no predecessor.
    starts = [
        mask
        for mask in range(1, 1 << n)
        if mask not in predecessor
    ]

    orders = []

    for start in starts:
        chain = [start]

        while chain[-1] in successor:
            chain.append(successor[chain[-1]])

        order_indices = []

        # Add all players in the starting coalition.
        current = 0
        for i in range(n):
            if (start >> i) & 1:
                order_indices.append(i)
                current |= 1 << i

        # Add one new player at each step.
        for nxt in chain[1:]:
            added = nxt ^ current
            i = (added & -added).bit_length() - 1
            order_indices.append(i)
            current = nxt

        orders.append([players[i] for i in order_indices])

    expected = comb(n, n // 2)
    assert len(orders) == expected

    return orders