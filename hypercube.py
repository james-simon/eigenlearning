import jax.numpy as np
from jax import random

import scipy as sp

from utils import sample_kernel

def get_hypercube_dataset(n_bits, target_terms, full=True, n=None, subkey=None):
    """Generate a dataset on the hypercube.

    n_bits -- The dimensionality of the hypercube, giving M = 2^n_bits
    target_terms -- The target functions of the dataset to return. Must be a dictionary mapping eigenmodes
                    to their coefficients, with zero coefficients omitted. For example, {1:3, 2:7} denotes
                    3*phi_1 + 7*phi_2, where phi_1 and phi_2 are sensitive to the first 1 and 2 bits,
                    respectively. To choose which spins are sensitive instead of using the first k, a
                    binary vector can be given instead of k.
    full -- if True, return all M datapoints. If False, return n randomly-chosen points.
    n -- the dataset size.
    subkey -- the subkey with which to choose random points.
    """
    if not full and n == 0:
        return np.array([]), [np.array([]) for terms in target_terms]

    # generate the integers 0 ... 2^n_bits
    packed_bitsets = np.arange(0, 2** n_bits, 1, dtype=np.uint32)[:, None].view(np.uint8)[:, ::-1]

    if not full:
        assert (n is not None) and (subkey is not None)
        indices = np.arange(0, 2 ** n_bits, 1)
        indices = random.choice(subkey, indices, shape=[n], replace=False)
        packed_bitsets = packed_bitsets[indices]
        # jax throws an error when optimizing on one data point, so add a duplicate
        if n == 1:
            packed_bitsets = np.concatenate([packed_bitsets, packed_bitsets])

    # expand them into bitsets and slice off the zeros
    bitsets = np.unpackbits(packed_bitsets, axis=1)
    bitsets = bitsets[:, 32 - n_bits:]

    targets = []
    for terms in target_terms:
        y = np.zeros(shape=(len(packed_bitsets), 1))

        for s in terms:
            sensitive_bits = np.array([j < s for j in range(n_bits)]) if isinstance(s, int) else np.array(s)
            bitset_sums = bitsets.dot(np.diag(sensitive_bits)).sum(axis=1)[:, None]
            parities = bitset_sums % 2
            parities = 2 * parities.astype(np.int8) - 1
            y += parities * terms[s]

        targets += [y]

    # switch from {0,1} to {-1,1}, converting to int8's so they can be negative
    bitsets = 2 * bitsets.astype(np.int8) - 1

    return bitsets, targets


def hypercube_eigenvalues(kernel_fn, n_bits, normalized=True, k_type='ntk'):
    """Return the eigenvalues of the given rotationally-invariant kernel on the hypercube, plus their multiplicities.

    kernel_fn -- The neural_tangents kernel function. Assumed to be rotation-invariant
    n_bits -- The dimensionality of the hypercube
    normalized -- If True, normalize all eigenvalues to sum to one
    k_type -- Either 'ntk' or 'nngp'
    """
    Ks = sample_kernel(kernel_fn, cosines=np.linspace(1, -1, n_bits + 1), d=n_bits, norm=n_bits ** .5, k_type=k_type)

    if normalized:
        Ks /= Ks[-1]
        Ks /= 2 ** n_bits

    eigenvalues = []
    multiplicities = []

    for n_sensitive_bits in range(0, n_bits + 1, 1):
        eigenvalue = 0

        for n_flips in range(0, n_bits + 1, 1):
            for n_sensitive_flips in range(0, min(n_flips, n_sensitive_bits) + 1, 1):
                # (which sensitive bits are flipped) * (which insensitive bits are flipped)
                mult = sp.special.comb(n_sensitive_bits, n_sensitive_flips) * sp.special.comb(n_bits - n_sensitive_bits,
                                                                                              n_flips - n_sensitive_flips)

                eigenvalue += mult * Ks[n_bits - n_flips] * (-1) ** n_sensitive_flips

        eigenvalues += [eigenvalue]
        multiplicities += [sp.special.comb(n_bits, n_sensitive_bits)]

    return np.array(eigenvalues), np.array(multiplicities)