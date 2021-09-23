import jax.numpy as np
from jax import random

import scipy as sp
from scipy.special import gegenbauer, eval_gegenbauer, roots_gegenbauer, gamma

from utils import sample_kernel

def hyp_har_eigenfn(d, k, zs, normalization='norm 1'):
    assert normalization in ['norm 1', 'max 1', None]
    if normalization == 'norm 1':
        norm_factor = hyp_har_multiplicity(d, k) ** .5 / eval_gegenbauer(k, d / 2 - 1, 1)
    elif normalization == 'max 1':
        norm_factor = 1 / eval_gegenbauer(k, d / 2 - 1, 1)
    else:
        norm_factor = 1

    return eval_gegenbauer(k, d / 2 - 1, zs) * norm_factor

def hyp_har_multiplicity(d, k):
    return 1 if k == 0 else (2 * k + d - 2) / k * sp.special.comb(k + d - 3, k - 1)

def get_hypersphere_dataset(d, target_terms, n, subkey):
    if n == 0:
        return np.array([]), np.array([])

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


def hypersphere_eigenvalues(kernel_fn, d, k_max=10, normalized=True, k_type='ntk', n_sample_pts=10 ** 3):
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