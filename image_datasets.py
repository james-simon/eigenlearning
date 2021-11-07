import jax.numpy as np
from jax import random

import numpy as base_np

import torchvision


def select_subset(X, y, n, subkey=None):
  if n is None:
    return X, y
  if subkey is None:
    return X[:n], y[:n]
  indices = np.arange(0, len(X))
  indices = random.choice(subkey, indices, shape=[n], replace=False)
  return X[indices], y[indices]

def select_classes(X, y, y_labels, classes):
    idxs = base_np.isin(y_labels, np.array(classes))
    X, y = X[idxs], y[idxs]
    if len(classes) == 2:
        y = y[:,:1] * 2 - 1
    return X, y

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

    train_X = train_Xy.data.numpy()
    train_y_labels = base_np.array(train_Xy.targets)
    test_X = test_Xy.data.numpy()
    test_y_labels = base_np.array(test_Xy.targets)

    num_classes = len(np.unique(train_y_labels))

    train_y = base_np.zeros((train_y_labels.size, num_classes))  # One-hot class labels
    train_y[base_np.arange(train_y_labels.size), train_y_labels] = 1
    test_y = base_np.zeros((test_y_labels.size, num_classes))
    test_y[base_np.arange(test_y_labels.size), test_y_labels] = 1

    if classes is not None:
        train_X, train_y = select_classes(train_X, train_y, train_y_labels, classes)
        test_X, test_y = select_classes(test_X, test_y, train_y_labels, classes)

    # normalize globally
    train_mean = train_X.mean()
    train_std = train_X.std()
    train_X = (train_X - train_mean) / train_std
    test_X = (test_X - train_mean) / train_std

    # convert to jax.np
    train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

    if n_train is not None:
        train_X, train_y = select_subset(train_X, train_y, n_train, subkey)
    if n_test is not None:
        test_X, test_y = select_subset(test_X, test_y, n_test, subkey)

    if name in ['cifar10']:
        train_X, test_X = np.transpose(train_X, (0, 3, 1, 2)), np.transpose(test_X, (0, 3, 1, 2))

    if flattened:
        train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))
    # else:
    #     if name in ['mnist', 'fmnist']:
    #         train_X, test_X = train_X.reshape((-1, 28, 28)), test_X.reshape((-1, 28, 28))
    #     if name in ['cifar10']:
    #         train_X, test_X = np.transpose(train_X, (0, 3, 1, 2)), np.transpose(test_X, (0, 3, 1, 2))

    return train_X, train_y, test_X, test_y
#
#
# def get_mnist_dataset(n_train=None, n_test=None, classes=None, subkey=None, flattened=True):
#     train_Xy = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
#     test_Xy = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
#     train_X = train_Xy.data.numpy()
#     train_y_labels = base_np.array(train_Xy.targets)
#     test_X = test_Xy.data.numpy()
#     test_y_labels = base_np.array(test_Xy.targets)
#
#     num_classes = len(np.unique(train_y_labels))
#
#     train_X = train_X.reshape((len(train_X), -1))
#     test_X = test_X.reshape((len(test_X), -1))
#     train_y = base_np.zeros((train_y_labels.size, num_classes))  # One-hot class labels
#     train_y[base_np.arange(train_y_labels.size), train_y_labels] = 1
#     test_y = base_np.zeros((test_y_labels.size, num_classes))
#     test_y[base_np.arange(test_y_labels.size), test_y_labels] = 1
#
#     if classes is not None:
#         train_X, train_y = select_classes(train_X, train_y, train_y_labels, classes)
#         test_X, test_y = select_classes(test_X, test_y, train_y_labels, classes)
#
#     # normalize globally
#     train_mean = train_X.mean()
#     train_std = train_X.std()
#     train_X = (train_X - train_mean) / train_std
#     test_X = (test_X - train_mean) / train_std
#
#     # convert to jax.np
#     train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
#
#     if n_train is not None:
#         train_X, train_y = select_subset(train_X, train_y, n_train, subkey)
#     if n_test is not None:
#         test_X, test_y = select_subset(test_X, test_y, n_test, subkey)
#
#     if flattened:
#         train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))
#     else:
#         if name in ['mnist', 'fmnist']:
#             train_X, test_X = np.transpose(train_X, (0, 3, 1, 2)), np.transpose(test_X, (0, 3, 1, 2))
#
#     return train_X, train_y, test_X, test_y
#
#
# def get_fashion_mnist_dataset(n_train=None, n_test=None, classes=None, subkey=None, flattened=True):
#     train_Xy = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
#     test_Xy = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)
#     train_X = train_Xy.data.numpy()
#     train_y_labels = base_np.array(train_Xy.targets)
#     test_X = test_Xy.data.numpy()
#     test_y_labels = base_np.array(test_Xy.targets)
#
#     num_classes = len(np.unique(train_y_labels))
#
#     train_X = train_X.reshape((len(train_X), -1))
#     test_X = test_X.reshape((len(test_X), -1))
#     train_y = base_np.zeros((train_y_labels.size, num_classes))  # One-hot class labels
#     train_y[base_np.arange(train_y_labels.size), train_y_labels] = 1
#     test_y = base_np.zeros((test_y_labels.size, num_classes))
#     test_y[base_np.arange(test_y_labels.size), test_y_labels] = 1
#
#     if classes is not None:
#         train_X, train_y = select_classes(train_X, train_y, train_y_labels, classes)
#         test_X, test_y = select_classes(test_X, test_y, train_y_labels, classes)
#
#     # normalize globally
#     train_mean = train_X.mean()
#     train_std = train_X.std()
#     train_X = (train_X - train_mean) / train_std
#     test_X = (test_X - train_mean) / train_std
#
#     # convert to jax.np
#     train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
#
#     if n_train is not None:
#         train_X, train_y = select_subset(train_X, train_y, n_train, subkey)
#     if n_test is not None:
#         test_X, test_y = select_subset(test_X, test_y, n_test, subkey)
#
#     if not flattened:
#         train_X = train_X.reshape((-1, 28, 28))
#         test_X = test_X.reshape((-1, 28, 28))
#
#     return train_X, train_y, test_X, test_y
#
#
# def get_cifar10_dataset(n_train=None, n_test=None, classes=None, subkey=None, flattened=True):
#   train_Xy = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
#   test_Xy = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
#   train_X = train_Xy.data
#   train_y_labels = base_np.array(train_Xy.targets)
#   test_X = test_Xy.data
#   test_y_labels = base_np.array(test_Xy.targets)
#
#   num_classes = len(np.unique(train_y_labels))
#
#   train_y = base_np.zeros((train_y_labels.size, num_classes))   # One-hot class labels
#   train_y[base_np.arange(train_y_labels.size),train_y_labels] = 1
#   test_y = base_np.zeros((test_y_labels.size, num_classes))
#   test_y[base_np.arange(test_y_labels.size),test_y_labels] = 1
#
#   if classes is not None:
#       train_X, train_y = select_classes(train_X, train_y, train_y_labels, classes)
#       test_X, test_y = select_classes(test_X, test_y, train_y_labels, classes)
#
#   # normalize globally
#   train_mean = train_X.mean()
#   train_std = train_X.std()
#   train_X = (train_X - train_mean)/train_std
#   test_X = (test_X - train_mean)/train_std
#
#   # convert to jax.np
#   train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
#
#   if n_train is not None:
#     train_X, train_y = select_subset(train_X, train_y, n_train, subkey)
#   if n_test is not None:
#     test_X, test_y = select_subset(test_X, test_y, n_test, subkey)
#
#   if flattened:
#     train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))
#   else:
#     train_X, test_X = np.transpose(train_X, (0,3,1,2)), np.transpose(test_X, (0,3,1,2))
#
#   return train_X, train_y, test_X, test_y