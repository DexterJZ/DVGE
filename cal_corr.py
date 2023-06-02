import os
import numpy as np


root = 'outputs/beta_celeba/z_and_y'
train_root = os.path.join(root, 'train')
val_root = os.path.join(root, 'val')

train_data_names = sorted(os.listdir(train_root))
val_data_names = sorted(os.listdir(val_root))

num_train_names = len(train_data_names)
num_val_names = len(val_data_names)

z_all = np.ndarray(shape=(num_train_names+num_val_names, 10))
y_all = np.ndarray(shape=(num_train_names+num_val_names, 1))

for i in range(num_train_names):
    data_name = os.path.join(train_root, train_data_names[i])

    with np.load(data_name) as data:
        z = data['z']
        y = data['y']

    z_all[i, :] = z
    y_all[i, 0] = int(~((y[20] == 0) & (y[39] == 1)))

for i in range(num_val_names):
    data_name = os.path.join(val_root, val_data_names[i])

    with np.load(data_name) as data:
        z = data['z']
        y = data['y']

    z_all[num_train_names+i, :] = z
    y_all[num_train_names+i, 0] = int(~((y[20] == 0) & (y[39] == 1)))

c_0 = np.corrcoef(z_all[:, 0], y_all[:, 0])
c_1 = np.corrcoef(z_all[:, 1], y_all[:, 0])
c_2 = np.corrcoef(z_all[:, 2], y_all[:, 0])
c_3 = np.corrcoef(z_all[:, 3], y_all[:, 0])
c_4 = np.corrcoef(z_all[:, 4], y_all[:, 0])
c_5 = np.corrcoef(z_all[:, 5], y_all[:, 0])
c_6 = np.corrcoef(z_all[:, 6], y_all[:, 0])
c_7 = np.corrcoef(z_all[:, 7], y_all[:, 0])
c_8 = np.corrcoef(z_all[:, 8], y_all[:, 0])
c_9 = np.corrcoef(z_all[:, 9], y_all[:, 0])

print(c_0[0, 1])
print(c_1[0, 1])
print(c_2[0, 1])
print(c_3[0, 1])
print(c_4[0, 1])
print(c_5[0, 1])
print(c_6[0, 1])
print(c_7[0, 1])
print(c_8[0, 1])
print(c_9[0, 1])
