import numpy as np
import jax.numpy as jnp
from jax import random

import torch
import torch.nn.functional as F
import torchvision


def get_image_dataset(name, n_train=None, n_test=None, classes=None, subkey=None, flattened=True, normalized=False, downsampling_factor=None):

    dataset_dict = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist': torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
    }

    def get_xy(dataset):
        x = dataset.data.numpy() if name not in ['cifar10','cifar100'] else dataset.data
        y = dataset.targets.numpy() if name not in ['cifar10','cifar100'] else dataset.targets
        n_classes = int(max(y)) + 1

        if classes is not None:
            # convert old class labels to new
            converter = -1 * np.ones(n_classes)
            for new_class, group in enumerate(classes):
                group = [group] if type(group)==int else group
                for old_class in group:
                    converter[old_class] = new_class
            # remove datapoints not in new classes
            mask = (converter[y] >= 0)
            x = x[mask]
            y = converter[y][mask]
            # update n_classes
            n_classes = int(max(y)) + 1

        if downsampling_factor is not None:
            x = x[:,::downsampling_factor,::downsampling_factor]
        # normalize globally (correct for the overall mean and std)
        x = (x - x.mean())/x.std()
        # normalize locally (normalize each image vector independently)
        if normalized:
            x /= (x ** 2).mean(axis=(1,2,3))[:, None] ** .5
            
        # onehot encoding, unless binary classification (+1,-1)
        if n_classes != 2:
            y = F.one_hot(torch.Tensor(y).long())
        else:
            y = 2*y - 1
            y = y[:, None] #reshape

        # convert to immutable jax arrays
        x, y = jnp.array(x), jnp.array(y)
        return x, y

    train = dataset_dict[name](root='./data', train=True, download=True, transform=None)
    train_X, train_y = get_xy(train)
    
    test = dataset_dict[name](root='./data', train=False, download=True, transform=None)
    test_X, test_y = get_xy(test)

    if n_train is not None:
        idxs = jnp.arange(0, n_train) if subkey is None else random.choice(subkey, jnp.arange(0, len(train_X)), shape=[n_train], replace=False)
        train_X, train_y = train_X[idxs], train_y[idxs]
        assert len(train_X) == n_train

    if n_test is not None:
        idxs = jnp.arange(0, n_test) if subkey is None else random.choice(subkey, jnp.arange(0, len(test_X)), shape=[n_test], replace=False)
        test_X, test_y = test_X[idxs], test_y[idxs]
        assert len(test_X) == n_test

    # add a dummy channel dimension to MNIST and FMNIST
    if name in ['mnist', 'fmnist']:
        train_X, test_X = train_X[:,:,:,None], test_X[:,:,:,None]

    # # swap dimensions for CIFAR10 so they're (n, ch, x, y) <- dropping this for now
    # if name in ['cifar10']:
    #     train_X, test_X = jnp.transpose(train_X, (0, 3, 1, 2)), jnp.transpose(test_X, (0, 3, 1, 2))

    if flattened:
        train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))

    return train_X, train_y, test_X, test_y