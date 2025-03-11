import random
import itertools

import numpy as np
from scipy import optimize
from tqdm import tqdm


class Item:
    def __init__(self, weight=None, value=None):
        self.weight = weight
        self.value = value

    def __repr__(self):
        return f"Item with w: {self.weight}, v: {self.value}"


def items_from_csv(file_path):
    """Loads items from a CSV file (value,weight)."""
    data = np.loadtxt(file_path, delimiter=",", dtype=np.int32)
    return [Item(w, v) for v, w in zip(data[:, 0], data[:, 1])]


def gen_random_items(n: int, value_range=(1, 10), weight_range=(1, 10)):
    weight = [random.randint(*weight_range) for _ in range(n)]
    val = [random.randint(*value_range) for _ in range(n)]
    return [Item(w, v) for w, v in zip(weight, val)]


class Knapsack:
    def __init__(self, items: [Item], capacity: int):
        self.items = items
        self.capacity = capacity
        self.n = len(items)

    def __repr__(self):
        """String representation of the Knapsack object."""
        return f"Knapsack(capacity={self.capacity}, items={self.n})"

    def get_weights(self):
        return [i.weight for i in self.items]

    def get_values(self):
        return [i.value for i in self.items]

    def brute_force(self, use_tqdm=False):
        combinations_of_size_r = (
            itertools.combinations(self.items, r) for r in range(self.n + 1)
        )
        iterator = tqdm(
            (c for comb in combinations_of_size_r for c in comb),
            desc="Solving Knapsack brute force",
            disable=not use_tqdm,
            total=2**self.n,
        )
        best_val = 0
        selected_items = None
        for c in iterator:
            tot_weight = sum([i.weight for i in c])
            tot_value = sum([i.value for i in c])
            if tot_weight <= self.capacity and tot_value > best_val:
                best_val = tot_value
                selected_items = c

        return best_val, selected_items

    def greedy(self):
        # ratio = [i.value / i.weight for i in self.items]

        sorted_items = sorted(
            self.items, key=lambda x: x.value / x.weight, reverse=True
        )

        weight = 0
        best_val = 0
        selected_items = []

        while weight < self.capacity and sorted_items:
            it = sorted_items.pop(0)
            if weight + it.weight > self.capacity:
                return best_val, selected_items
            weight += it.weight
            best_val += it.value
            selected_items.append(it)

        return best_val, selected_items

    def dynamic_programming_basepy(self, use_tqdm=False):
        # dp is the dynamic program matrix with rows beeing the items and columns beeing the weight
        # Note that dp[i][w] should be the best possible value for weight limit w by using the first i items.

        dp = [[0 for _ in range(self.capacity + 1)] for _ in range(self.n + 1)]

        # Start iteration at 1
        with tqdm(total=self.n, disable=not use_tqdm) as pbar:
            for i in range(1, self.n + 1):
                pbar.update(1)
                for w in range(1, self.capacity + 1):
                    if self.items[i - 1].weight <= w:
                        dp[i][w] = max(
                            self.items[i - 1].value
                            + dp[i - 1][w - self.items[i - 1].weight],
                            dp[i - 1][w],
                        )
                    else:
                        dp[i][w] = dp[i - 1][w]

        best_val = dp[self.n][self.capacity]
        # Find selected items
        selected_items = []
        w = self.capacity
        for i in range(self.n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(self.items[i - 1])
                w -= self.items[i - 1].weight
        return best_val, selected_items

    def solve_scipy(self):
        sizes = np.array(self.get_weights())
        values = np.array(self.get_values())
        bounds = optimize.Bounds(0, 1)
        integrality = np.full_like(values, True)
        capacity = self.capacity
        constraints = optimize.LinearConstraint(A=sizes, lb=0, ub=capacity)
        res = optimize.milp(
            c=-values, constraints=constraints, integrality=integrality, bounds=bounds
        )
        if not res.success:
            print("Opimizations failed")
            return (None, None)

        selected_items = []
        best_val = 0
        for i in range(self.n):
            if res.x[i]:
                selected_items.append(self.items[i])
                best_val += self.items[i].value

        return best_val, selected_items

    def dynamic_programming_fast(self, use_tqdm=False):
        """Optimized DP using NumPy"""
        weights = self.get_weights()
        values = self.get_values()
        dp = np.zeros(self.capacity + 1, dtype=np.int32)

        with tqdm(total=self.n, disable=not use_tqdm) as pbar:
            for i in range(self.n):
                pbar.update(1)
                w, v = weights[i], values[i]
                if w <= self.capacity:
                    # NumPy array slice assignment (vectorized update)
                    dp[w : self.capacity + 1] = np.maximum(
                        dp[w : self.capacity + 1], dp[: self.capacity + 1 - w] + v
                    )

        # Retrieve best value
        best_val = dp[self.capacity]

        return best_val, None

    def dynamic_programming(self, use_tqdm=False):
        """Optimized DP using NumPy with Backtracking"""
        weights = self.get_weights()
        values = self.get_values()
        dp = np.zeros(self.capacity + 1, dtype=np.int32)
        item_selected = np.zeros(
            (self.n, self.capacity + 1), dtype=bool
        )  # Track selections
        with tqdm(total=self.n, disable=not use_tqdm) as pbar:
            for i in range(self.n):
                pbar.update(1)
                w, v = weights[i], values[i]
                if w <= self.capacity:
                    # Identify positions where value is updated
                    update_positions = dp[w : self.capacity + 1] < (
                        dp[: self.capacity + 1 - w] + v
                    )

                    # Update DP table
                    dp[w : self.capacity + 1] = np.where(
                        update_positions,
                        dp[: self.capacity + 1 - w] + v,
                        dp[w : self.capacity + 1],
                    )

                    # Mark selections in tracking array
                    item_selected[i, w : self.capacity + 1] = update_positions

        # Retrieve best value
        best_val = dp[self.capacity]

        # Backtracking to find selected items
        selected_items = []
        w = self.capacity
        for i in range(self.n - 1, -1, -1):
            if w >= weights[i] and item_selected[i, w]:  # Check if item i was selected
                selected_items.append(self.items[i])
                w -= weights[i]

        return best_val, selected_items
