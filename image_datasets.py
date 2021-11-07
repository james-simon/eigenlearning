import jax.numpy as np
from jax import random

import numpy as base_np

import torchvision
import torch.nn.functional as F


def get_image_dataset(name, n_train=None, n_test=None, classes=None, subkey=None, flattened=True):

    if name == 'mnist':
        train_Xy = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        test_Xy = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
    if name == 'fmnist':
        train_Xy = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
        test_Xy = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)
    if name == 'cifar10':
        train_Xy = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        test_Xy = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    train_X = train_Xy.data.numpy() if name not in ['cifar10'] else train_Xy.data
    train_y_labels = base_np.array(train_Xy.targets)
    train_y = F.one_hot(train_y_labels)

    test_X = test_Xy.data.numpy() if name not in ['cifar10'] else test_Xy.data
    test_y_labels = base_np.array(test_Xy.targets)
    test_y = F.one_hot(test_y_labels)

    # num_classes = len(np.unique(train_y_labels))

    # train_y = base_np.zeros((train_y_labels.size, num_classes))  # One-hot class labels
    # train_y[base_np.arange(train_y_labels.size), train_y_labels] = 1
    # test_y = base_np.zeros((test_y_labels.size, num_classes))
    # test_y[base_np.arange(test_y_labels.size), test_y_labels] = 1

    if classes is not None:
        idxs_tr, idxs_te = (train_y[:,np.array(classes)].sum(axis=1) > 0), (test_y[:,np.array(classes)].sum(axis=1) > 0)
        train_X, train_y, test_X, test_y = train_X[idxs_tr], train_y[idxs_tr], test_X[idxs_te], test_y[idxs_te]

    # normalize globally
    train_mean = train_X.mean()
    train_std = train_X.std()
    train_X = (train_X - train_mean) / train_std
    test_X = (test_X - train_mean) / train_std

    # convert to jax.np
    train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

    if n_train is not None:
        idxs = np.arange(0, n_train) if subkey is None else random.choice(subkey, np.arange(0, len(train_X)), shape=[n_train], replace=False)
        train_X, train_y = train_X[idxs], train_y[idxs]
    if n_test is not None:
        idxs = np.arange(0, n_test) if subkey is None else random.choice(subkey, np.arange(0, len(test_X)), shape=[n_test], replace=False)
        test_X, test_y = test_X[idxs], test_y[idxs]

    if name in ['cifar10']:
        train_X, test_X = np.transpose(train_X, (0, 3, 1, 2)), np.transpose(test_X, (0, 3, 1, 2))

    if flattened:
        train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))

    return train_X, train_y, test_X, test_y