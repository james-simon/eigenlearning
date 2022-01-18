import jax.numpy as np
from jax import random


def get_unit_circle_dataset(M, target_terms, full=True, n=None, subkey=None):
  """Generate a dataset on the discretized unit circle.

  M -- The number of points the unit circle is discretized into
  target_terms -- The target functions of the dataset to return. Must be a list of dictionaries
                  mapping eigenmodes to their coefficients, with zero coefficients omitted. For example,
                  [{(1,'c'):1, (2,'s'):7}] denotes one target function of sqrt(2)*1*cos(theta) + sqrt(2)*7*sin(2*theta).
  full -- if True, return all M datapoints. If False, return n randomly-chosen points.
  n -- the dataset size.
  subkey -- the subkey with which to choose random points.
  """
  thetas = np.linspace(0, 2* np.pi, M, endpoint=False)

  if not full:
    assert (n is not None) and (subkey is not None)
    thetas = random.choice(subkey, thetas, shape=[n], replace=False)
    # jax throws an error when optimizing on one data point, so add a duplicate
    # if n == 1:
    #   thetas = np.concatenate([thetas, thetas])

  thetas = thetas[:, None]
  coords = np.concatenate([np.cos(thetas), np.sin(thetas)], axis=1)

  targets = []
  for terms in target_terms:
    y = np.zeros_like(thetas)

    for (k, s_or_c) in terms:
      assert k == int(k)
      assert 0 <= k and k <= M / 2
      assert s_or_c in [None, 's', 'c']

      if k > 0:
        assert s_or_c in 'sc'
        if s_or_c == 'c':
          y += np.cos(k * thetas) * 2 ** .5 * terms[(k, s_or_c)]
        else:
          y += np.sin(k * thetas) * 2 ** .5 * terms[(k, s_or_c)]
      else:
        y += np.cos(0 * thetas) * terms[(k, s_or_c)]

    targets += [y]

  return thetas, coords, targets


def unit_circle_eigenvalues(kernel_fn, M, normalized=True, k_type='ntk'):
  """Return the eigenvalues of the given rotationally-invariant kernel on the discretized unit circle.

  kernel_fn -- The neural_tangents kernel function. Assumed to be rotation-invariant
  M -- The number of points the unit circle's discretized into
  normalized -- If True, normalize all eigenvalues to sum to one
  k_type -- Either 'ntk' or 'nngp'
  """
  thetas, coords, _ = get_unit_circle_dataset(M, [{(0, None): 1}])
  Ks = kernel_fn(coords[0:1], coords, k_type)[0]

  if normalized:
    Ks /= Ks[0]
    Ks /= M

  eigenvalues = [(np.cos(k * thetas)[:, 0] * Ks).sum().item() for k in range(M)]

  return np.array(eigenvalues)