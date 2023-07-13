from time import time
import numpy as np

def cross_prod(a, b):
    result = [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

    return result

a = [1,2,3]
b = [4,5,6]

s = time()
for i in range(100000): np.cross(a,b)
e = time()
print("Numpy computation took",e-s)

s = time()
for i in range(100000): cross_prod(a,b)
e = time()
print("Manual cross product took",e-s)


