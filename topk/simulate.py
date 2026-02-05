
#!/usr/bin/env python3
"""
Monte Carlo simulation for estimating oracle (listwise ranker) call costs
of quicksort and quickselect algorithms with multi-pivot partitioning.

This is a true simulation - oracle call counts emerge from actually running
the algorithm, not from theoretical formulas.

Model:
- We have N items to sort/select from
- An oracle can rank up to W items per call
- We use P pivots to partition items into P+1 buckets
- Each call to the oracle that includes items costs 1 call
"""

import argparse
import csv
import os
import random
from typing import List, Tuple


class Oracle:
    """
    Simulates a listwise ranker oracle.
    Each call ranks up to W items and returns their sorted order.
    """

    def __init__(self, W: int):
        self.W = W
        self.calls = 0

    def reset(self):
        self.calls = 0

    def rank(self, items: List[float]) -> List[float]:
        """
        Rank items (up to W) and return sorted in descending order.
        This costs exactly 1 oracle call.
        """
        if not items:
            return []
        if len(items) > self.W:
            raise ValueError(f"Cannot rank {len(items)} items, max is {self.W}")
        self.calls += 1
        return sorted(items, reverse=True)


def quicksort_sim(items: List[float], P: int, oracle: Oracle) -> List[float]:
    """
    Multi-pivot quicksort simulation.

    Algorithm:
    1. Base case: if |items| <= W, make one oracle call to sort
    2. Otherwise:
       a. Randomly select P items as pivots
       b. Make 1 oracle call to sort the pivots (establishes bucket boundaries)
       c. Partition remaining (N-P) items via oracle calls (each includes
          P pivots + up to (W-P) items)
       d. Recurse on each bucket
    """
    n = len(items)

    if n == 0:
        return []
    if n == 1:
        return items[:]

    W = oracle.W

    # Base case: small enough to sort in one call
    if n <= W:
        return oracle.rank(items)

    # Step 1: Randomly select P pivots
    pivot_indices = set(random.sample(range(n), P))
    pivots = [items[i] for i in pivot_indices]

    # Step 2: Sort the pivots (1 oracle call)
    sorted_pivots = oracle.rank(pivots)  # Returns pivots in descending order

    # Step 3: Partition remaining items into P+1 buckets
    # Each oracle call includes P pivots + up to (W-P) items to partition
    buckets = [[] for _ in range(P + 1)]
    items_per_call = W - P

    # Get non-pivot items
    remaining = [items[i] for i in range(n) if i not in pivot_indices]

    # Partition remaining items via oracle calls
    for i in range(0, len(remaining), items_per_call):
        batch = remaining[i:i + items_per_call]
        combined = list(sorted_pivots) + batch
        oracle.rank(combined)  # 1 oracle call

        # Assign batch items to buckets based on pivot boundaries
        for item in batch:
            bucket_idx = 0
            for pv in sorted_pivots:
                if item < pv:
                    bucket_idx += 1
                else:
                    break
            buckets[bucket_idx].append(item)

    # Add pivots to their bucket boundaries (pivot[i] goes to bucket[i])
    for i, pv in enumerate(sorted_pivots):
        buckets[i].append(pv)

    # Step 4: Recursively sort each bucket
    result = []
    for bucket in buckets:
        if bucket:
            result.extend(quicksort_sim(bucket, P, oracle))

    return result


def quickselect_sim(items: List[float], k: int, P: int, oracle: Oracle) -> List[float]:
    """
    Multi-pivot quickselect simulation to find top-k items.

    Same as quicksort, but only recurses into buckets needed to find top-k.
    """
    n = len(items)

    if n == 0:
        return []
    if k <= 0:
        return []
    if k >= n:
        return items[:]

    W = oracle.W

    # Base case: small enough to handle in one call
    if n <= W:
        ranked = oracle.rank(items)
        return ranked[:k]

    # Step 1: Randomly select P pivots
    pivot_indices = set(random.sample(range(n), P))
    pivots = [items[i] for i in pivot_indices]

    # Step 2: Sort the pivots (1 oracle call)
    sorted_pivots = oracle.rank(pivots)

    # Step 3: Partition remaining items into P+1 buckets
    buckets = [[] for _ in range(P + 1)]
    items_per_call = W - P

    remaining = [items[i] for i in range(n) if i not in pivot_indices]

    for i in range(0, len(remaining), items_per_call):
        batch = remaining[i:i + items_per_call]
        combined = list(sorted_pivots) + batch
        oracle.rank(combined)  # 1 oracle call

        for item in batch:
            bucket_idx = 0
            for pv in sorted_pivots:
                if item < pv:
                    bucket_idx += 1
                else:
                    break
            buckets[bucket_idx].append(item)

    # Add pivots to their bucket boundaries
    for i, pv in enumerate(sorted_pivots):
        buckets[i].append(pv)

    # Step 4: Select from buckets until we have k items
    result = []
    remaining_k = k

    for bucket in buckets:
        if remaining_k <= 0:
            break
        if not bucket:
            continue

        if len(bucket) <= remaining_k:
            result.extend(bucket)
            remaining_k -= len(bucket)
        else:
            top_items = quickselect_sim(bucket, remaining_k, P, oracle)
            result.extend(top_items)
            remaining_k = 0

    return result


def run_quicksort_trial(N: int, P: int, W: int) -> int:
    """Run one quicksort trial and return oracle call count."""
    oracle = Oracle(W)
    items = [random.random() for _ in range(N)]
    quicksort_sim(items, P, oracle)
    return oracle.calls


def run_quickselect_trial(N: int, k: int, P: int, W: int) -> int:
    """Run one quickselect trial and return oracle call count."""
    oracle = Oracle(W)
    items = [random.random() for _ in range(N)]
    quickselect_sim(items, k, P, oracle)
    return oracle.calls


def simulate_quicksort(N: int, W: int, trials: int, P_range: range,
                       verbose: bool = True) -> dict:
    """Run quicksort simulation for all P values."""
    results = {}
    for P in P_range:
        if P >= W:
            continue
        total_calls = sum(run_quicksort_trial(N, P, W) for _ in range(trials))
        avg_calls = total_calls / trials
        results[P] = avg_calls
        if verbose:
            print(f"  P={P:2d}: {avg_calls:.4f} avg calls")
    return results


def simulate_quickselect(N: int, W: int, trials: int, P_range: range,
                         k_values: List[int], verbose: bool = True) -> dict:
    """Run quickselect simulation for all P and k values."""
    results = {}
    for P in P_range:
        if P >= W:
            continue
        for k in k_values:
            if k > N or k < 1:
                continue
            total_calls = sum(run_quickselect_trial(N, k, P, W) for _ in range(trials))
            avg_calls = total_calls / trials
            results[(P, k)] = avg_calls
            if verbose:
                print(f"  P={P:2d}, k={k:4d}: {avg_calls:.4f} avg calls")
    return results


def save_quicksort_csv(results: dict, N: int, W: int, trials: int, output_dir: str):
    """Save quicksort results to CSV."""
    filename = f"T_N{N}_W{W}_algo-quicksort_trials-{trials}.csv"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algo', 'P', 'T_est'])
        for P in sorted(results.keys()):
            writer.writerow(['quicksort', P, f"{results[P]:.4f}"])
    print(f"Saved to {output_path}")
    return output_path


def save_quickselect_csv(results: dict, N: int, W: int, trials: int, output_dir: str):
    """Save quickselect results to CSV."""
    filename = f"T_N{N}_W{W}_algo-quickselect_trials-{trials}.csv"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algo', 'P', 'k', 'T_est'])
        for (P, k) in sorted(results.keys()):
            writer.writerow(['quickselect', P, k, f"{results[(P, k)]:.4f}"])
    print(f"Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Monte Carlo simulation for quicksort/quickselect oracle call costs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--algorithm', '-a', choices=['quicksort', 'quickselect', 'both'],
                        default='both', help='Algorithm to simulate')
    parser.add_argument('--N', '-n', type=int, default=1000,
                        help='Number of items')
    parser.add_argument('--W', '-w', type=int, default=20,
                        help='Window size (max items per oracle call)')
    parser.add_argument('--trials', '-t', type=int, default=5000,
                        help='Number of Monte Carlo trials')
    parser.add_argument('--P-max', type=int, default=None,
                        help='Maximum pivot count (default: W-1)')
    parser.add_argument('--P-min', type=int, default=1,
                        help='Minimum pivot count')
    parser.add_argument('--k-values', type=str, default=None,
                        help='Comma-separated k values for quickselect')
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                        help='Output directory for CSV files')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress per-parameter output')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    P_max = args.P_max if args.P_max else args.W - 1
    P_range = range(args.P_min, P_max + 1)

    # Parse k values for quickselect
    if args.k_values:
        k_values = [int(k.strip()) for k in args.k_values.split(',')]
    else:
        # Default: k from 1 to N in steps of 100, plus N
        k_values = list(range(1, args.N + 1, 100))
        if args.N not in k_values:
            k_values.append(args.N)
        k_values = sorted(k_values)

    os.makedirs(args.output_dir, exist_ok=True)
    verbose = not args.quiet

    if args.algorithm in ['quicksort', 'both']:
        print(f"\n{'='*60}")
        print(f"Simulating QUICKSORT (N={args.N}, W={args.W}, trials={args.trials})")
        print(f"{'='*60}")
        results = simulate_quicksort(args.N, args.W, args.trials, P_range, verbose)
        save_quicksort_csv(results, args.N, args.W, args.trials, args.output_dir)

    if args.algorithm in ['quickselect', 'both']:
        print(f"\n{'='*60}")
        print(f"Simulating QUICKSELECT (N={args.N}, W={args.W}, trials={args.trials})")
        print(f"k values: {k_values}")
        print(f"{'='*60}")
        results = simulate_quickselect(args.N, args.W, args.trials, P_range, k_values, verbose)
        save_quickselect_csv(results, args.N, args.W, args.trials, args.output_dir)


if __name__ == '__main__':
    main()
