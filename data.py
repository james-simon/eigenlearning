import jax.numpy as jnp
from jax import random
import jax

import scipy as sp
from scipy.special import eval_gegenbauer, roots_gegenbauer, gamma

from abc import ABC, abstractmethod

from theory import Spectrum


def sample_rotn_invar_kernel(kernel_fn, cosines, d, k_type):
    """Sample different angles of a rotationally-invariant kernel.
    kernel_fn -- the JAX kernel function to sample; assumed to be rotationally-invariant
    cosines -- the cosines of the angles at which to sample the kernel
    d -- the input dimension
    k_type -- either 'ntk' or 'nngp'
    """
    norm = jnp.sqrt(d)
    sines = (1 - cosines ** 2) ** .5
    u0 = jnp.array([1*(i == 0) for i in range(d)]) # [1, 0, 0, 0, ...]
    u1 = jnp.array([1*(i == 1) for i in range(d)]) # [0, 1, 0, 0, ...]
    xs = norm * (jnp.outer(cosines, u0) + jnp.outer(sines, u1))

    # get the kernel between these points and an endpoint
    Ks = kernel_fn(xs[0:1], xs, k_type)[0][::-1]

    return Ks


class SyntheticDomain(ABC):
    
    @abstractmethod
    def __init__(self, dim):
        pass
    
    @abstractmethod
    def get_spectrum(self, kernel_fn, k_type='ntk'):
        pass
    
    @abstractmethod
    def get_dataset(self, target, n_train, n_test, subkey):
        pass

class Hypersphere(SyntheticDomain):
    
    def __init__(self, dim):
        """ dim (int): embedding dimension of hypersphere (corresponds to S^(d-1)).
        """
        assert dim >= 3
        self.dim = int(dim)
    
    def eval_eigenfn(self, k, zs, normalization='norm 1'):
        """ Return eigenfunction Y_k0 outputs on points with z-coordinates zs
        """
        assert normalization in ['norm 1', 'max 1', None]
        d = self.dim
        if normalization == 'norm 1':
            norm_factor = self.get_mode_multiplicity(k) ** .5 / eval_gegenbauer(k, d / 2 - 1, 1)
        elif normalization == 'max 1':
            norm_factor = 1 / eval_gegenbauer(k, d / 2 - 1, 1)
        else:
            norm_factor = 1

        return eval_gegenbauer(k, d / 2 - 1, zs) * norm_factor

    def get_mode_multiplicity(self, k):
        """ Return the degeneracy of the k-th level of eigenmodes
        """
        if k == 0:
            return 1
        d = self.dim
        return (2 * k + d - 2) / k * sp.special.comb(k + d - 3, k - 1)

    def get_spectrum(self, kernel_fn, k_max=50, k_type='ntk'):
        """ Return the eigenvalues (+ multiplicities) of the given rotationally-invariant kernel
        on the hypersphere
        kernel_fn -- The neural_tangents kernel function. Assumed to be rotation-invariant
        k_max (int) -- The max k for which to compute eigenvalues and multiplicities.
        k_type (str) -- Either 'ntk' or 'nngp'
        """
        
        n_sample_pts = 10 ** 3
        zs, ws = roots_gegenbauer(n_sample_pts, self.dim / 2 - 1)
        
        cosines = zs
        Ks = sample_rotn_invar_kernel(kernel_fn, cosines, self.dim, k_type)

        # note: this is only approximate normalization since zs[-1] isn't quite 1
        # but this scale factor doesn't affect any downstream predictions anyways
        Ks /= Ks[-1]

        Ks = jnp.array(Ks)

        eigenvalues = []
        multiplicities = []
        kk = range(k_max + 1)

        for k in kk:
            fs = self.eval_eigenfn(k, zs, normalization='max 1')

            # integrate the eigenfn against the kernel
            prefactor = (1 / jnp.pi ** .5) * gamma(self.dim / 2) / gamma((self.dim - 1) / 2)
            lambda_k = prefactor * (fs * Ks * ws).sum().item()
            mult_k = self.get_mode_multiplicity(k)

            eigenvalues.append(lambda_k)
            multiplicities.append(mult_k)

        return Spectrum(eigenvalues, multiplicities, kk)

    def get_dataset(self, target, n_train, n_test, subkey=None):
        """ Generate a dataset on the hypersphere.
        target (dict): The target function from which samples are drawn. Must be a dictionary mapping
                        eigenmodes to their coefficients, with zero coefficients omitted. For example,
                        {1:3, 2:7} denotes 3*Y_10 + 7*Y_20. To get eigenfunctions besides the m=0 mode, 
                        one can instead supply a (k, unit-vector) pair, and the eigenfunction will be
                        rotated to align with that unit vector instead of the z-axis.
        n_train (int) -- the trainset size, must be nonzero
        n_test (int) -- the testset size
        subkey (int) -- jax prng subkey (one-time use) for random sampling.
        """
        if subkey is None:
            subkey = jnp.array([0, 42], dtype='uint32')
        assert n_train > 0
        N = n_train + n_test

        X = random.normal(subkey, shape=(N, self.dim))
        X = X / jnp.linalg.norm(X, axis=1)[:, None]

        y = jnp.zeros(shape=(N, 1))
        for component in target:
            if isinstance(component, int):
                k = component
                vec = jnp.array([int(i == 0) for i in range(self.dim)])[:, None]
            else:
                k = component[0]
                vec = jnp.array(component[1])[:, None]

            zs = X.dot(vec)
            y += self.eval_eigenfn(k, zs) * target[component]
        
        train_X, train_y, test_X, test_y =  X[:n_train], y[:n_train], X[n_train:], y[n_train:]
        assert len(train_X) == len(train_y) == n_train
        assert len(test_X) == len(test_y) == n_test
        return train_X, train_y, test_X, test_y


class Hypercube(SyntheticDomain):
    
    def __init__(self, dim):
        """ dim (int): dimension of hypercube with 2^dim corners.
        """
        assert dim >= 1
        self.dim = int(dim)
    
    def get_spectrum(self, kernel_fn, k_type='ntk'):
        """ Return the eigenvalues (+ multiplicities) of the given rotationally-invariant kernel
        on the hypercube
        kernel_fn -- The neural_tangents kernel function. Assumed to be rotation-invariant
        k_type (str) -- Either 'ntk' or 'nngp'
        """
        cosines = jnp.linspace(1, -1, self.dim + 1)
        Ks = sample_rotn_invar_kernel(kernel_fn, cosines, self.dim, k_type)

        # normalize so all eigenvalues sum to one
        Ks /= Ks[-1]
        Ks /= 2 ** self.dim

        eigenvalues = []
        multiplicities = []
        
        n_bits = self.dim
        for n_sensitive_bits in range(0, n_bits+1, 1):
            eigenvalue = 0

            for n_flips in range(0, n_bits+1, 1):
                for n_sensitive_flips in range(0, min(n_flips, n_sensitive_bits) + 1, 1):
                    # (which sensitive bits are flipped) * (which insensitive bits are flipped)
                    sensitive_flips = sp.special.comb(n_sensitive_bits, n_sensitive_flips)
                    insensitive_flips = sp.special.comb(n_bits - n_sensitive_bits,
                                                        n_flips - n_sensitive_flips)
                    mult = sensitive_flips * insensitive_flips

                    eigenvalue += mult * Ks[n_bits - n_flips] * (-1)**n_sensitive_flips

            eigenvalues.append(float(eigenvalue))
            multiplicities.append(sp.special.comb(n_bits, n_sensitive_bits))

        return Spectrum(eigenvalues, multiplicities)
    
    def get_dataset(self, target, n_train=None, n_test=None, subkey=None):
        """ Generate a dataset on the hypercube.
        target (dict): The target function from which samples are drawn. Must be a dictionary mapping
                        eigenmodes to their coefficients, with zero coefficients omitted. For example,
                        {1:3, 2:7} denotes 3*phi_1 + 7*phi_2, where phi_1 and phi_2 are sensitive to
                        the first 1 and 2 bits, respectively. To choose which spins are sensitive
                        instead of using the first k, a binary vector can be given instead of k.
        n_train (int) -- the trainset size, must be at least 2. Default: 2^dim (full sample space)
        n_test (int) -- the testset size. Testset may overlap trainset. Default: 2^dim (full sample space)
        subkey (int) -- jax prng subkey (one-time use) for random sampling.
        """
        if subkey is None:
            subkey = jnp.array([0, 42], dtype='uint32')
        n_bits = self.dim
        # generate the integers 0 ... 2^n_bits
        M = 2 ** n_bits
        all_packed_bitsets = jnp.arange(0, M, 1, dtype=jnp.uint32)
        all_packed_bitsets = all_packed_bitsets[:, None].view(jnp.uint8)[:, ::-1]
        n_train = n_train if n_train else M
        n_test = n_test if n_test else M
        # jax throws an error when optimizing on one data point
        assert n_train > 1        
        indices = jnp.arange(0, M, 1)
        
        def get_data(n, sk):
            chosen = random.choice(sk, indices, shape=[n], replace=False)
            packed_bitsets = all_packed_bitsets[chosen]

            # expand them into bitsets and slice off the zeros
            bitsets = jnp.unpackbits(packed_bitsets, axis=1)
            bitsets = bitsets[:, 32 - n_bits:]

            y = jnp.zeros(shape=(len(packed_bitsets), 1))

            for s in target:
                sensitive_bits = jnp.array([j < s for j in range(n_bits)]) if isinstance(s, int) else jnp.array(s)
                bitset_sums = bitsets.dot(jnp.diag(sensitive_bits)).sum(axis=1)[:, None]
                parities = bitset_sums % 2
                parities = 2 * parities.astype(jnp.int8) - 1
                y += parities * target[s]

            # switch from {0,1} to {-1,1}, converting to int8's so they can be negative
            X = 2 * bitsets.astype(jnp.int8) - 1
            return X, y
        
        sk1, sk2 = random.split(subkey, 2)
        train_X, train_y = get_data(n_train, sk1)
        assert len(train_X) == len(train_y) == n_train
        test_X, test_y = get_data(n_test, sk2)
        assert len(test_X) == len(test_y) == n_test
        return train_X, train_y, test_X, test_y


class UnitCircle(SyntheticDomain):
    
    def __init__(self, M):
        """ M (int): the number of points into which we discretize the unit circle.
        """
        assert M >= 2
        self.M = int(M)
    
    def get_spectrum(self, kernel_fn, k_type='ntk'):
        """Return the eigenvalues (+ multiplicities) of the given rotationally-invariant kernel
        on the discretized unit circle.
        kernel_fn -- The neural_tangents kernel function. Assumed to be rotation-invariant
        k_type (str) -- Either 'ntk' or 'nngp'
        """
        thetas = jnp.linspace(0, 2* jnp.pi, self.M, endpoint=False)
        coords = jnp.vstack([jnp.cos(thetas), jnp.sin(thetas)]).T
        Ks = kernel_fn(coords[0:1], coords, k_type)[0]

        # normalize so all eigenvalues to sum to one
        Ks /= Ks[0]
        Ks /= self.M

        kk = range(self.M)
        eigenvalues = [(jnp.cos(k * thetas) * Ks).sum().item() for k in kk]
        return Spectrum(eigenvalues, kk=kk)
    
    def get_dataset(self, target, n_train=None, n_test=None, subkey=None):
        """Generate a dataset on the hypersphere.
        target (dict): The target function from which samples are drawn. Must be a dictionary
                        mapping eigenmodes to their coefficients, with zero coefficients omitted. For
                        example, [{(1,'c'):1, (2,'s'):7}] denotes one target function of
                        sqrt(2)*1*cos(theta) + sqrt(2)*7*sin(2*theta).
        n_train (int) -- the trainset size, must be at least 2. Default: M (full sample space)
        n_test (int) -- the testset size. Testset may overlap trainset. Default: M (full sample space)
        subkey (int) -- jax prng subkey (one-time use) for random sampling.
        """
        if subkey is None:
            subkey = jnp.array([0, 42], dtype='uint32')
        all_thetas = jnp.linspace(0, 2* jnp.pi, self.M, endpoint=False)
        n_train = n_train if n_train else self.M
        n_test = n_test if n_test else self.M
        # jax throws an error when optimizing on one data point
        assert n_train > 1
        
        def get_data(n, sk):
            thetas = random.choice(sk, all_thetas, shape=[n], replace=False)

            thetas = thetas[:, None]
            X = jnp.concatenate([jnp.cos(thetas), jnp.sin(thetas)], axis=1)

            y = jnp.zeros_like(thetas)

            for (k, s_or_c) in target:
                assert k == int(k)
                assert 0 <= k and k <= self.M / 2
                assert s_or_c in [None, 's', 'c']

                if k > 0:
                    assert s_or_c in 'sc'
                    if s_or_c == 'c':
                        y += jnp.cos(k * thetas) * 2 ** .5 * target[(k, s_or_c)]
                    else:
                        y += jnp.sin(k * thetas) * 2 ** .5 * target[(k, s_or_c)]
                else:
                    y += jnp.cos(0 * thetas) * target[(k, s_or_c)]
            return X, y
        
        sk1, sk2 = random.split(subkey, 2)
        train_X, train_y = get_data(n_train, sk1)
        assert len(train_X) == len(train_y) == n_train
        test_X, test_y = get_data(n_test, sk2)
        assert len(test_X) == len(test_y) == n_test
        return train_X, train_y, test_X, test_y


import torch
import torch.nn.functional as F
import torchvision



def kernel_eigendecomposition(kernel_fn, x_data):
    """
    Eigendecomposition of a data kernel matrix
    Args:
        kernel_fn: The kernel function returned by stax
        x_data: ndarray of input data, length M
    Returns:
        tuple(lambdas, U)
        lambdas: Mx1 jax ndarray eigenvalues, increasing order
        U: MxM jax ndarray, columns are corresponding eigenvectors
    """
    K = kernel_fn(x_data, get='ntk')
    M = len(x_data)
    K = jax.device_put(K)
    lambdas, U = jnp.linalg.eigh(K)
    
    lambdas /= M
    return lambdas, U


class ImageData():

    dataset_dict = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist': torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
    }
    
    def __init__(self, dataset_name):
        assert dataset_name in self.dataset_dict
        self.name = dataset_name
        self.dataset = self.dataset_dict[dataset_name]
    
    def get_dataset(self, n_train, n_test=None, classes=None, subkey=None):
        
        def get_xy(dataset):
            import numpy as np
            x = dataset.data.numpy() if self.name not in ['cifar10','cifar100'] else dataset.data
            y = dataset.targets.numpy() if self.name not in ['cifar10','cifar100'] else dataset.targets
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

            # normalize globally (correct for the overall mean and std)
            x = (x - x.mean())/x.std()
            # # normalize locally (normalize each image vector independently)
            # x /= (x ** 2).mean(axis=(1,2,3))[:, None] ** .5
                
            # onehot encoding, unless binary classification (+1,-1)
            if n_classes != 2:
                y = F.one_hot(torch.Tensor(y).long())
            else:
                y = 2*y - 1
                y = y[:, None] #reshape

            # convert to immutable jax arrays
            x, y = jnp.array(x), jnp.array(y)
            return x, y

        train = self.dataset_dict[self.name](root='./data', train=True, download=True, transform=None)
        train_X, train_y = get_xy(train)
        
        test = self.dataset_dict[self.name](root='./data', train=False, download=True, transform=None)
        test_X, test_y = get_xy(test)
        
        # get training and test subset
        if subkey is None:
            train_X, train_y = train_X[:n_train], train_y[:n_train]
            if n_test is not None:
                test_X, test_y = test_X[:n_test], test_y[:n_test]
        else:
            sk_train, sk_test = random.split(subkey, 2)
            train_idxs = random.choice(sk_train, len(train_X),
                                       shape=(int(n_train),), replace=False)
            train_X, train_y = train_X[train_idxs], train_y[train_idxs]
            if n_test is not None:
                test_idxs = random.choice(sk_test, len(test_X),
                                          shape=(int(n_test),), replace=False)
                test_X, test_y = test_X[test_idxs], test_y[test_idxs]
        assert len(train_X) == n_train
        if n_test:
            assert len(test_X) == n_test

        # add a dummy channel dimension to MNIST and FMNIST
        if self.name in ['mnist', 'fmnist']:
            train_X, test_X = train_X[:,:,:,None], test_X[:,:,:,None]

        # flatten
        train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))

        return train_X, train_y, test_X, test_y

    def get_eigendata(self, kernel_fn, X, y):
        eigenvalues, eigenvectors = kernel_eigendecomposition(kernel_fn, X)
        spectrum = Spectrum(eigenvalues)
        y_eigencoeffs = jnp.matmul(eigenvectors.T, y).reshape(-1)
        y_eigencoeffs = y_eigencoeffs / jnp.linalg.norm(y_eigencoeffs)
        eigenvectors = eigenvectors[spectrum.sort_order]
        y_eigencoeffs = y_eigencoeffs[spectrum.sort_order]
        return {
            "spectrum": spectrum,
            "eigenvecs": eigenvectors,
            "eigenlevel_coeffs": y_eigencoeffs,
        }
