"""
@file   : x.py
@time   : 2026-03-21
"""
import numpy as np


data = np.random.randint(0, 10, size=(8, 8))
print(data)

res = np.where(data >= 5)
print(res)



