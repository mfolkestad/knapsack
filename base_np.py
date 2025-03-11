import numpy as np
import random
import itertools
from tqdm import tqdm


class Item:
    def __init__(self, weight=None, value=None):
        self.weight = weight
        self.value = value

    def __repr__(self):
        return f"Item with w: {self.weight}, v: {self.value}"


def gen_random_items(n: int, value_range=(1, 10), weight_range=(1, 10)):
    weight = np.random.randint(*weight_range, size=n)
    val = np.random.randint(*value_range, size=n)
    return [Item(w, v) for w, v in zip(weight, val)]


class Knapsack:
    def __init__(self, items: [Item], capacity: int):
        self.items = items
        self.capacity = capacity
        self.n = len(items)
        self.weights = np.array([i.weight for i in items], dtype=int)
        self.values = np.array([i.value for i in items], dtype=int)

    def __repr__(self):
        return f"Knapsack(capacity={self.capacity}, items={self.n})"

    def brute_force(self, use_tqdm=False):
        best_val = 0
        best_combination = None
        all_indices = np.arange(self.n, dtype=int)  # Ensure integer type

        # Iterate through all subsets using indices
        for r in range(self.n + 1):
            for c in itertools.combinations(all_indices, r):
                c = np.array(c, dtype=int)  # Ensure valid NumPy indexing
                tot_weight = self.weights[c].sum()  # Vectorized sum
                tot_value = self.values[c].sum()  # Vectorized sum

                if tot_weight <= self.capacity and tot_value > best_val:
                    best_val = tot_value
                    best_combination = c

        selected_items = (
            [self.items[i] for i in best_combination]
            if best_combination is not None
            else []
        )
        return best_val, selected_items
