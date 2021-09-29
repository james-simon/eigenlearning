import jax.numpy as np
from jax import random

import scipy as sp
from scipy.special import gegenbauer, eval_gegenbauer, roots_gegenbauer, gamma

from utils import sample_kernel

def hyp_har_eigenfn(d, k, zs, normalization='norm 1'):
    """ The eigenfunction Y_k0 evaluated on points on the hypersphere with z-coordinates zs """
    assert normalization in ['norm 1', 'max 1', None]
    if normalization == 'norm 1':
        norm_factor = hyp_har_multiplicity(d, k) ** .5 / eval_gegenbauer(k, d / 2 - 1, 1)
    elif normalization == 'max 1':
        norm_factor = 1 / eval_gegenbauer(k, d / 2 - 1, 1)
    else:
        norm_factor = 1

    return eval_gegenbauer(k, d / 2 - 1, zs) * norm_factor

def hyp_har_multiplicity(d, k):
    """ The degeneracy of the k-th level of eigenmodes on the hypersphere in dimension d """
    return 1 if k == 0 else (2 * k + d - 2) / k * sp.special.comb(k + d - 3, k - 1)

def get_hypersphere_dataset(d, target_terms, n, subkey):
    """Generate a dataset on the hypersphere.

    d -- The dimensionality of the hypersphere. d here is the embedding dimension, so this really corresponds to S^(d-1).
    target_terms -- The target functions of the dataset to return. Must be a dictionary mapping eigenmodes to their
                    coefficients, with zero coefficients omitted. For example, {1:3, 2:7} denotes 3*Y_10 + 7*Y_20.
                    To get eigenfunctions at a given k besides the m=0 mode, one can instead supply a (k, unit-vector)
                    pair, and the eigenfunction will be rotated to align with that unit vector instead of the z-axis.
    n -- the dataset size.
    subkey -- the subkey with which to choose random points.
    """
    if n == 0:
        return np.array([]), [np.array([]) for terms in target_terms]

    xs = random.normal(subkey, shape=(n, d))
    xs = xs / np.linalg.norm(xs, axis=1)[:, None]

    targets = []

    for terms in target_terms:
        y = np.zeros(shape=(n, 1))
        for component in terms:
            if isinstance(component, int):
                k = component
                vec = np.array([int(i == 0) for i in range(d)])[:, None]
            else:
                k = component[0]
                vec = np.array(component[1])[:, None]

            zs = xs.dot(vec)

            y += hyp_har_eigenfn(d, k, zs) * terms[component]

        targets += [y]

    return xs, targets


def hypersphere_eigenvalues(kernel_fn, d, k_max=50, normalized=True, k_type='ntk', n_sample_pts=10 ** 3):
    """Return the eigenvalues of the given rotationally-invariant kernel on the hypersphere, plus their multiplicities.

    kernel_fn -- The neural_tangents kernel function. Assumed to be rotation-invariant
    d -- The dimensionality of the hypersphere. d here is the embedding dimension, so this really corresponds to S^(d-1).
    k_max -- The max k for which to compute eigenvalues and multiplicities.
    normalized -- If True, normalize all eigenvalues to sum to one
    k_type -- Either 'ntk' or 'nngp'
    n_sample_pts -- the number of sample points to use when computing the eigenvalue integral
    """
    assert d >= 3

    zs, ws = roots_gegenbauer(n_sample_pts, d / 2 - 1)

    Ks = sample_kernel(kernel_fn, cosines=zs, d=d, norm=d ** .5, k_type=k_type)

    # note: this is only approximate normalization since zs[-1] isn't quite 1
    # but this scale factor doesn't affect any downstream predictions anyways
    if normalized:
        Ks /= Ks[-1]

    # Ks = zs

    Ks = np.array(Ks)

    eigenvalues = []
    multiplicities = []

    for k in range(k_max + 1):
        fs = hyp_har_eigenfn(d, k, zs, normalization='max 1')

        # integrate the eigenfn against the kernel
        prefactor = (1 / np.pi ** .5) * gamma(d / 2) / gamma((d - 1) / 2)
        lambda_k = prefactor * (fs * Ks * ws).sum().item()
        mult_k = hyp_har_multiplicity(d, k)

        eigenvalues += [lambda_k]
        multiplicities += [mult_k]

    return np.array(eigenvalues), np.array(multiplicities)