import pandas as pd
import os
import numpy as np

root = os.path.join('data', 'CelebA')
annotations = pd.read_csv('{}/list_attr_celeba.txt'.format(root),
                          delimiter=" ", skiprows=1, header=None)

a = annotations.iloc[:, 1:].to_numpy()

sens = 20
ot = 2

print(a[:, sens])

b1 = a[:, ot] == 1
c1 = a[:, sens] == 1
d1 = a[:, sens] == -1
e1 = np.logical_and(b1, c1)
f1 = np.logical_and(b1, d1)

print(c1.sum(), d1.sum())
print(e1.sum(), f1.sum())
print(e1.sum()/c1.sum()-f1.sum()/d1.sum())


b2 = a[:180000, ot] == 1
c2 = a[:180000, sens] == 1
d2 = a[:180000, sens] == -1
e2 = np.logical_and(b2, c2)
f2 = np.logical_and(b2, d2)

print(c2.sum(), d2.sum())
print(e2.sum(), f2.sum())
print(e2.sum()/c2.sum()-f2.sum()/d2.sum())

b3 = a[180000:, ot] == 1
c3 = a[180000:, sens] == 1
d3 = a[180000:, sens] == -1
e3 = np.logical_and(b3, c3)
f3 = np.logical_and(b3, d3)

print(c3.sum(), d3.sum())
print(e3.sum(), f3.sum())


b3 = a[180000:, ot] == 1
c3 = a[180000:, 20] == -1
d3 = a[180000:, 39] == 1
e3 = np.logical_and(d3, c3)
f3 = np.logical_and(e3, b3)

print(22599-e3.sum(), e3.sum())
print(b3.sum()-f3.sum(), f3.sum())