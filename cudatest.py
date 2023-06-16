import numpy as np
import cupy as cp

x_gpu =  cp.array([1,2,3])
norm = cp.linalg.norm(x_gpu)
print(norm)

x_matrix = cp.array([[1,2,3],[4,5,6],[7,8,10]])
inverse = cp.linalg.inv(x_matrix)
print(inverse)

print(inverse*x_gpu)