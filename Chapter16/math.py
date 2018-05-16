import numpy as np
import statistics

data1 = [1, 2, 3, 4]
M1=statistics.mean(data1)
print("Mean data1",M1)



data2 = [1, 2, 3, 5]
M2=statistics.mean(data2)
print("Mean data2",M2)

#var = mean(abs(x - x.mean())**2).
print("Variance 1", np.var(data1))
print("Variance 2", np.var(data2))


x=np.array([[1, 2, 3, 4],
            [1, 2, 3, 5]])

a=np.cov(x)
print(a)

from numpy import linalg as LA
w, v = LA.eigh(a)
print("eigenvalue(s)",w)
print("eigenvector(s)",v)



