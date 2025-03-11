import importlib
import knapsack as ks
import numpy as np

_ = importlib.reload(ks)


n = 100
c = 100
s = ks.Items().generate_random_items(n, value_range=(1, 100), weight_range=(1, 10))

knapsack = ks.Knapsack(c, s)
knapsack.pseudo_polynimial_algorithm(True)
knapsack._pseudo_polynimial_algorithm(True)
