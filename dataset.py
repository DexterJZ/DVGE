"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CelebADataset(ImageFolder):
    def __init__(self, root, transform=None):
        super(CelebADataset, self).__init__(root, transform)
        self.indices = range(len(self))
        self.annotations = pd.read_csv('{}/list_attr_celeba.txt'.format(root),
                                       delimiter=" ", skiprows=1, header=None)

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        fn1 = path1.split('/')[-1].strip()
        fn2 = path2.split('/')[-1].strip()

        idx1 = int(fn1.split('.')[0]) - 1
        idx2 = int(fn2.split('.')[0]) - 1

        label1 = torch.FloatTensor(
            np.array(list(self.annotations.iloc[idx1, 1:])).clip(min=0))

        label2 = torch.FloatTensor(
            np.array(list(self.annotations.iloc[idx2, 1:])).clip(min=0))

        return img1, label1, img2, label2


class ZAndYDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.data_names = sorted(os.listdir(self.root))

    def __getitem__(self, index):
        data_name = os.path.join(self.root, self.data_names[index])

        with np.load(data_name) as data:
            z = data['z']
            y = data['y']

        z = torch.from_numpy(z)
        y = torch.from_numpy(y)

        return z, y

    def __len__(self):
        return len(self.data_names)


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    """For VAE use."""
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    if name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root': root, 'transform': transform}
        dset = CelebADataset
    elif name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root': root, 'transform': transform}
        dset = CustomImageFolder
    elif name.lower() == 'dsprites':
        root = os.path.join(
            dset_dir,
            'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset
    else:
        # raise NotImplementedError
        return None

    train_data = dset(**train_kwargs)

    if args.train:
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=False)

    data_loader = train_loader
    return data_loader


def return_train_data(args):
    name = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers

    root = os.path.join('outputs', name, 'z_and_y/train')
    train_data = ZAndYDataset(root)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    return train_loader


def return_val_data(args):
    name = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers

    root = os.path.join('outputs', name, 'z_and_y/val')
    val_data = ZAndYDataset(root)

    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)

    return val_loader
