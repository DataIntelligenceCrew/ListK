#!/usr/bin/env python3
"""
Monte Carlo simulation for multi-pivot quicksort and quickselect.

Estimates the expected number of oracle (listwise ranker) calls needed to sort
or select top-k items from N items, where the oracle can rank up to W items
per call and P pivots are used for partitioning.
"""

import argparse
import csv
import os
import random
from typing import List


class Oracle:
    """Listwise ranker that can rank up to W items per call."""

    def __init__(self, W: int):
        self.W = W
        self.calls = 0

    def rank(self, items: List[float]) -> List[float]:
        """Rank items (up to W) in descending order. Costs 1 oracle call."""
        if not items:
            return []
        if len(items) > self.W:
            raise ValueError(f"Cannot rank {len(items)} items, max is {self.W}")
        self.calls += 1
        return sorted(items, reverse=True)


def quicksort(items: List[float], P: int, oracle: Oracle) -> List[float]:
    """
    Multi-pivot quicksort.

    1. Base case: if |items| <= W, sort in one oracle call
    2. Randomly select P pivots, sort them (1 call)
    3. Partition remaining N-P items (each call: P pivots + up to W-P items)
    4. Recurse on each bucket
    """
    n = len(items)
    if n <= 1:
        return items[:]
    if n <= oracle.W:
        return oracle.rank(items)

    # Select and sort P random pivots
    pivot_indices = set(random.sample(range(n), P))
    pivots = [items[i] for i in pivot_indices]
    sorted_pivots = oracle.rank(pivots)

    # Partition remaining items into P+1 buckets
    buckets = [[] for _ in range(P + 1)]
    remaining = [items[i] for i in range(n) if i not in pivot_indices]
    items_per_call = oracle.W - P

    for i in range(0, len(remaining), items_per_call):
        batch = remaining[i:i + items_per_call]
        oracle.rank(list(sorted_pivots) + batch)

        for item in batch:
            bucket_idx = sum(1 for pv in sorted_pivots if item < pv)
            buckets[bucket_idx].append(item)

    # Place pivots at bucket boundaries
    for i, pv in enumerate(sorted_pivots):
        buckets[i].append(pv)

    # Recurse
    result = []
    for bucket in buckets:
        if bucket:
            result.extend(quicksort(bucket, P, oracle))
    return result


def quickselect(items: List[float], k: int, P: int, oracle: Oracle) -> List[float]:
    """
    Multi-pivot quickselect to find top-k items.

    Same as quicksort, but only recurses into buckets needed for top-k.
    """
    n = len(items)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return items[:]
    if n <= oracle.W:
        return oracle.rank(items)[:k]

    # Select and sort P random pivots
    pivot_indices = set(random.sample(range(n), P))
    pivots = [items[i] for i in pivot_indices]
    sorted_pivots = oracle.rank(pivots)

    # Partition remaining items
    buckets = [[] for _ in range(P + 1)]
    remaining = [items[i] for i in range(n) if i not in pivot_indices]
    items_per_call = oracle.W - P

    for i in range(0, len(remaining), items_per_call):
        batch = remaining[i:i + items_per_call]
        oracle.rank(list(sorted_pivots) + batch)

        for item in batch:
            bucket_idx = sum(1 for pv in sorted_pivots if item < pv)
            buckets[bucket_idx].append(item)

    for i, pv in enumerate(sorted_pivots):
        buckets[i].append(pv)

    # Select from buckets until we have k items
    result = []
    remaining_k = k
    for bucket in buckets:
        if remaining_k <= 0 or not bucket:
            continue
        if len(bucket) <= remaining_k:
            result.extend(bucket)
            remaining_k -= len(bucket)
        else:
            result.extend(quickselect(bucket, remaining_k, P, oracle))
            break
    return result


def simulate_quicksort(N: int, W: int, P: int, trials: int) -> float:
    """Run trials and return average oracle calls for quicksort."""
    total = 0
    for _ in range(trials):
        oracle = Oracle(W)
        quicksort([random.random() for _ in range(N)], P, oracle)
        total += oracle.calls
    return total / trials


def simulate_quickselect(N: int, W: int, P: int, k: int, trials: int) -> float:
    """Run trials and return average oracle calls for quickselect."""
    total = 0
    for _ in range(trials):
        oracle = Oracle(W)
        quickselect([random.random() for _ in range(N)], k, P, oracle)
        total += oracle.calls
    return total / trials


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo simulation for multi-pivot quicksort/quickselect',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-a', '--algorithm', choices=['quicksort', 'quickselect', 'both'],
                        default='both', help='Algorithm to simulate')
    parser.add_argument('-n', '--N', type=int, default=1000, help='Number of items')
    parser.add_argument('-w', '--W', type=int, default=20, help='Max items per oracle call')
    parser.add_argument('-t', '--trials', type=int, default=5000, help='Monte Carlo trials')
    parser.add_argument('--P-min', type=int, default=1, help='Minimum pivot count')
    parser.add_argument('--P-max', type=int, default=None, help='Maximum pivot count')
    parser.add_argument('--k-values', type=str, default=None,
                        help='Comma-separated k values for quickselect')
    parser.add_argument('-o', '--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Random seed')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    P_max = args.P_max or (args.W - 1)
    P_range = range(args.P_min, P_max + 1)

    k_values = ([int(k) for k in args.k_values.split(',')] if args.k_values
                else sorted(set(list(range(1, args.N + 1, 100)) + [args.N])))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.algorithm in ('quicksort', 'both'):
        if not args.quiet:
            print(f"Simulating quicksort (N={args.N}, W={args.W}, trials={args.trials})")
        results = {P: simulate_quicksort(args.N, args.W, P, args.trials)
                   for P in P_range if P < args.W}
        if not args.quiet:
            for P, avg in sorted(results.items()):
                print(f"  P={P:2d}: {avg:.4f}")

        path = os.path.join(args.output_dir,
                            f"T_N{args.N}_W{args.W}_algo-quicksort_trials-{args.trials}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['algo', 'P', 'T_est'])
            for P in sorted(results):
                writer.writerow(['quicksort', P, f"{results[P]:.4f}"])
        print(f"Saved to {path}")

    if args.algorithm in ('quickselect', 'both'):
        if not args.quiet:
            print(f"Simulating quickselect (N={args.N}, W={args.W}, trials={args.trials})")
        results = {}
        for P in P_range:
            if P >= args.W:
                continue
            for k in k_values:
                if not (1 <= k <= args.N):
                    continue
                avg = simulate_quickselect(args.N, args.W, P, k, args.trials)
                results[(P, k)] = avg
                if not args.quiet:
                    print(f"  P={P:2d}, k={k:4d}: {avg:.4f}")

        path = os.path.join(args.output_dir,
                            f"T_N{args.N}_W{args.W}_algo-quickselect_trials-{args.trials}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['algo', 'P', 'k', 'T_est'])
            for (P, k) in sorted(results):
                writer.writerow(['quickselect', P, k, f"{results[(P, k)]:.4f}"])
        print(f"Saved to {path}")


if __name__ == '__main__':
    main()
