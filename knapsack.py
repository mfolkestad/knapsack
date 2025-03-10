import numpy as np
from tqdm import tqdm


class Knapsack:
    def __init__(self, capacity: int, items=None):
        """Initialize the knapsack with a given capacity and an optional set of items."""
        self.capacity = capacity
        self.items = items if items is not None else Items(None, None)

    def add_items(self, items):
        """Adds an Items object to the knapsack."""
        self.items = self.items.add_items(items.get_values(), items.get_weights())

    def __repr__(self):
        """String representation of the Knapsack object."""
        return f"Knapsack(capacity={self.capacity}, items={len(self.items)})"

    def _solve_without_selected(self, use_tqdm=False):
        """Solves the 0/1 knapsack problem using an optimized DP approach (1D array)."""
        values = self.items.get_values()
        weights = self.items.get_weights()
        n = len(self.items)

        # Edge case: No items
        if n == 0:
            return 0, []

        # Optimized DP: 1D array instead of 2D
        dp = np.zeros(self.capacity + 1, dtype=np.int32)

        # Initialize tqdm if enabled
        iterator = tqdm(range(n), desc="Solving Knapsack", disable=not use_tqdm)

        # Compute DP values
        for i in iterator:
            value, weight = values[i], weights[i]

            # Loop **backwards** to prevent overwriting necessary values
            for w in range(self.capacity, weight - 1, -1):
                dp[w] = max(dp[w], dp[w - weight] + value)

        # The maximum value is now at dp[self.capacity]
        max_value = dp[self.capacity]

        return max_value, []

    def _solve_with_selected(self, use_tqdm=False):
        """Solves the 0/1 knapsack problem using an optimized DP approach (1D array)."""
        values = self.items.get_values()
        weights = self.items.get_weights()
        n = len(self.items)

        # Edge case: No items
        if n == 0:
            return 0, []

        # Optimized DP: 1D array instead of 2D
        dp = np.zeros(self.capacity + 1, dtype=np.int32)
        keep = np.zeros((n, self.capacity + 1), dtype=bool)  # Track chosen items

        # Initialize tqdm if enabled
        iterator = tqdm(range(n), desc="Solving Knapsack", disable=not use_tqdm)

        # Compute DP values
        for i in iterator:
            value, weight = values[i], weights[i]

            # Loop **backwards** to prevent overwriting necessary values
            for w in range(self.capacity, weight - 1, -1):
                if dp[w - weight] + value > dp[w]:
                    dp[w] = dp[w - weight] + value
                    keep[i, w] = True  # Mark that we picked this item

        # The maximum value is now at dp[self.capacity]
        max_value = dp[self.capacity]

        # Backtrack to find selected items
        selected_items = []
        w = self.capacity
        for i in range(n - 1, -1, -1):
            if keep[i, w]:  # If item `i` was taken at weight `w`
                selected_items.append(i)
                w -= weights[i]

        return max_value, selected_items

    def solve_dp(self, return_selcted=True, use_tqdm=False):
        if return_selcted:
            return self._solve_with_selected(use_tqdm=use_tqdm)
        else:
            return self._solve_without_selected(use_tqdm=use_tqdm)

    def solve_greedy(self, use_tqdm=False):
        """
        Solves the knapsack problem using a greedy heuristic:
        - Picks items based on highest value/weight ratio first.
        - Runs in O(n log n) time.
        """
        values = self.items.get_values()
        weights = self.items.get_weights()
        n = len(self.items)

        # Compute value-to-weight ratio and sort by it (descending order)
        ratios = values / weights  # Element-wise division
        sorted_indices = np.argsort(-ratios)  # Sort in descending order
        total_value = 0
        total_weight = 0
        selected_items = []

        iterator = tqdm(
            sorted_indices, desc="Solving Knapsack greedily", disable=not use_tqdm
        )

        # Pick items greedily
        for i in iterator:
            if total_weight + weights[i] <= self.capacity:
                selected_items.append(i)
                total_weight += weights[i]
                total_value += values[i]

        return total_value, selected_items

    def solve_fptas(self, epsilon=0.1, use_tqdm=False):
        """
        Solves the knapsack problem using Fully Polynomial-Time Approximation Scheme (FPTAS).
        - epsilon (ε) controls accuracy vs. speed. Smaller ε gives better accuracy but increases runtime.
        - Runs in O(n^2 / ε), which is much faster than O(nW) for large cases.
        """
        values = self.items.get_values()
        weights = self.items.get_weights()
        n = len(self.items)

        # Edge case: No items
        if n == 0:
            return 0, []

        # Get the max value
        V_max = np.max(values)

        # Compute scaling factor K
        K = (epsilon * V_max) / n
        if K == 0:
            K = 1  # Avoid division by zero

        # Scale down values
        # NB astype(int) works as floor.
        scaled_values = (values / K).astype(int)

        # Approximate maximum possible total value
        V_sum = np.sum(scaled_values)

        # DP table using scaled values
        dp = np.full(V_sum + 1, np.inf, dtype=np.int32)
        dp[0] = 0

        # Track item choices
        keep = np.full((n, V_sum + 1), False, dtype=bool)

        # Initialize tqdm if enabled
        iterator = tqdm(range(n), desc="Solving Knapsack (FPTAS)", disable=not use_tqdm)

        for i in iterator:
            value, weight = scaled_values[i], weights[i]

            # Loop backwards over value sums
            for v in range(V_sum, value - 1, -1):
                if dp[v - value] + weight <= self.capacity:
                    new_weight = dp[v - value] + weight
                    if new_weight < dp[v]:
                        dp[v] = new_weight
                        keep[i, v] = True  # Track selection

        # Find the best valid value
        max_value_scaled = max(v for v in range(V_sum + 1) if dp[v] <= self.capacity)
        max_value = int(max_value_scaled * K)  # Rescale back

        # Backtrack to find selected items
        selected_items = []
        v = max_value_scaled
        for i in range(n - 1, -1, -1):
            if keep[i, v]:  # If item `i` was selected at value sum `v`
                selected_items.append(i)
                v -= scaled_values[i]

        return max_value, selected_items


class Items(np.ndarray):
    def __new__(cls, values=None, weights=None):
        """
        Creates a np array with items where each row represent an item and the columns are weights and values
        """
        if values is None or weights is None:
            obj = super().__new__(cls, (0, 2), dtype=np.int32)
        else:
            assert len(values) == len(weights), (
                "Values and weights must have the same length."
            )
            obj = np.array(list(zip(values, weights)), dtype=np.int32).view(cls)
        return obj

    def add_items(self, values, weights):
        """Adds multiple items at once."""
        new_items = np.array(list(zip(values, weights)), dtype=np.int32)
        return np.vstack((self, new_items)).view(Items)

    @staticmethod
    def from_csv(file_path):
        """Loads items from a CSV file (value,weight)."""
        data = np.loadtxt(file_path, delimiter=",", dtype=np.int32)
        return Items(values=data[:, 0], weights=data[:, 1])

    def to_csv(self, file_path):
        """Saves items to a CSV file."""
        np.savetxt(file_path, self.items, delimiter=",", fmt="%d")

    @staticmethod
    def generate_random_items(n, value_range=(1, 100), weight_range=(1, 50)):
        """Generates a large number of random items efficiently."""
        values = np.random.randint(
            value_range[0], value_range[1] + 1, size=n, dtype=np.int32
        )
        weights = np.random.randint(
            weight_range[0], weight_range[1] + 1, size=n, dtype=np.int32
        )
        return Items(values, weights)

    def get_values(self):
        return self[:, 0]  # Extract values

    def get_weights(self):
        return self[:, 1]  # Extract weights
