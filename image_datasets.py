import jax.numpy as np
from jax import random

import torch
import torch.nn.functional as F
import torchvision


def get_image_dataset(name, n_train=None, n_test=None, classes=None, subkey=None, flattened=True, normalized=False, downsampling_factor=None):

    if name == 'mnist':
        train_Xy = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        test_Xy = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
    if name == 'fmnist':
        train_Xy = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
        test_Xy = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)
    if name == 'cifar10':
        train_Xy = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        test_Xy = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    if name == 'cifar100':
        train_Xy = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
        test_Xy = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

    train_X = train_Xy.data.numpy() if name not in ['cifar10','cifar100'] else train_Xy.data
    train_y_labels = train_Xy.targets if name not in ['cifar10','cifar100'] else torch.Tensor(train_Xy.targets).long()
    train_y = np.array(F.one_hot(train_y_labels))

    test_X = test_Xy.data.numpy() if name not in ['cifar10','cifar100'] else test_Xy.data
    test_y_labels = test_Xy.targets if name not in ['cifar10','cifar100'] else torch.Tensor(test_Xy.targets).long()
    test_y = np.array(F.one_hot(test_y_labels))

    if classes is not None:
        classes = np.array(classes)
        idxs_tr, idxs_te = (train_y[:,classes].sum(axis=1) > 0), (test_y[:,classes].sum(axis=1) > 0)
        train_X, train_y, test_X, test_y = train_X[idxs_tr], train_y[idxs_tr][:,classes], test_X[idxs_te], test_y[idxs_te][:,classes]
        if len(classes) == 2:
            train_y, test_y = train_y[:,:1] * 2 - 1, test_y[:,:1] * 2 - 1

    if downsampling_factor is not None:
        train_X, test_X = train_X[:,::downsampling_factor,::downsampling_factor], test_X[:,::downsampling_factor,::downsampling_factor]

    # normalize globally (correct for the overall mean and std)
    train_mean = train_X.mean()
    train_std = train_X.std()
    train_X = (train_X - train_mean) / train_std
    test_X = (test_X - train_mean) / train_std

    # normalize locally (normalize each image vector independently)
    if normalized:
        train_X /= (train_X ** 2).mean(axis=(1,2,3))[:, None] ** .5
        test_X /= (test_X ** 2).mean(axis=(1,2,3))[:, None] ** .5

    # convert to jax.np
    train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

    if n_train is not None:
        idxs = np.arange(0, n_train) if subkey is None else random.choice(subkey, np.arange(0, len(train_X)), shape=[n_train], replace=False)
        train_X, train_y = train_X[idxs], train_y[idxs]
        assert len(train_X) == n_train

    if n_test is not None:
        idxs = np.arange(0, n_test) if subkey is None else random.choice(subkey, np.arange(0, len(test_X)), shape=[n_test], replace=False)
        test_X, test_y = test_X[idxs], test_y[idxs]
        assert len(test_X) == n_test

    # add a dummy channel dimension to MNIST and FMNIST
    if name in ['mnist', 'fmnist']:
        train_X, test_X = train_X[:,:,:,None], test_X[:,:,:,None]

    # # swap dimensions for CIFAR10 so they're (n, ch, x, y) <- dropping this for now
    # if name in ['cifar10']:
    #     train_X, test_X = np.transpose(train_X, (0, 3, 1, 2)), np.transpose(test_X, (0, 3, 1, 2))

    if flattened:
        train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))

    return train_X, train_y, test_X, test_y