import importlib
import base_py as ks
import time

_ = importlib.reload(ks)

n = 100_000
capacity = 20_000
k = ks.Knapsack(
    ks.gen_random_items(n, value_range=(1, 10_000), weight_range=(1, 10_000)), capacity
)

start = time.perf_counter()
v, s = k.greedy()
print(f"Greedy: {time.perf_counter() - start}, v: {v}, selected: {len(s)}")

start = time.perf_counter()
v, s = k.dynamic_programming_fast(True)
print(f"DP (only best val) : {time.perf_counter() - start}, v: {v}")

start = time.perf_counter()
v, s = k.dynamic_programming(True)
print(f"DP: {time.perf_counter() - start}, v: {v}, selected: {len(s)}")

start = time.perf_counter()
v, s = k.solve_scipy()
print(f"SCIPY: {time.perf_counter() - start}, v: {v}, selected: {len(s)}")

start = time.perf_counter()
v, s = k.dynamic_programming_basepy(True)
print(f"DP (basepy): {time.perf_counter() - start}, v: {v}, selected: {len(s)}")
