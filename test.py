import importlib
import knapsack as ks
import numpy as np

_ = importlib.reload(ks)

n = 100_000
c = 100
s2 = ks.Items().generate_random_items(n, value_range=(1, 100))
s3 = ks.Items().generate_random_items(n, value_range=(1, 1000))
s6 = ks.Items().generate_random_items(n, value_range=(1, 1_000_000))
for s in [s2, s3, s6]:
    knapsack = ks.Knapsack(c, s)
    m_dp, selected_dp = knapsack.solve_dp(use_tqdm=True)

knapsack = ks.Knapsack(c, s2)
m_dp, selected_dp = knapsack.solve_dp(use_tqdm=True)

knapsack = ks.Knapsack(c, s2)
m_dp, selected_dp = knapsack.solve_dp(use_tqdm=True)


m_gr, selected_greedy = knapsack.solve_greedy(use_tqdm=False)
m_ftpas, _ = knapsack.solve_fptas(0.1, True)

weights = s.get_weights()
values = s.get_values()
epsilon = 0.1
# Get the max value
V_max = np.max(values)

# Compute scaling factor K
K = (epsilon * V_max) / n
if K == 0:
    K = 1  # Avoid division by zero

# Scale down values
scaled_values = (values / K).astype(int)

# Approximate maximum possible total value
V_sum = np.sum(scaled_values)
