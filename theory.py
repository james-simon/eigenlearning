import time

import jax.numpy as np
import numpy as basenp
import scipy as sp

from unit_circle import get_unit_circle_dataset, unit_circle_eigenvalues
from hypercube import get_hypercube_dataset, hypercube_eigenvalues
from hypersphere import get_hypersphere_dataset, hypersphere_eigenvalues
from image_datasets import get_image_dataset

import utils
from utils import kernel_predictions, net_predictions



# helper function used in calculating kappa
def lrn_sum(kappa, lambdas, mults=1):
    return (mults * lambdas / (lambdas + kappa)).sum()

# find kappa for a given eigensystem and n
def find_kappa(n, lambdas, mults=1, ridge=0):
    kappa = sp.optimize.bisect(lambda k: lrn_sum(kappa, lambdas, mults=mults) + ridge / k - n, 1e-10, 1e10)
    return kappa

# compute eigenmode learnabilities
def eigenmode_learnabilities(n, lambdas, mults=1, ridge=0, kappa=None):
    if kappa is None:
        kappa = find_kappa(n, lambdas, mults=mults, ridge=ridge)

    if isinstance(lambdas, list):
        lambdas = np.array(lambdas)

    lrns = lambdas / (lambdas + kappa)
    return lrns

# compute pure-noise MSE \mathcal{E}_0
def noise_fitting_factor(n, lambdas, mults=1, ridge=0, kappa=None):
    lrns = eigenmode_learnabilities(n, lambdas, mults=mults, ridge=ridge, kappa=kappa)

    lrn2_sum = (lrns ** 2 * mults).sum()

    return n / (n - lrn2_sum)

def theoretical_predictions(n, f_terms, kernel_fn=None, domain=None, lambdas=None, mults=1, ridge=0, **kwargs):
    assert (kernel_fn is not None and domain is not None) or (lambdas is not None)

    # if f_terms is not in dictionary form, make it so (helps with compatibility with synthetic domains)
    if type(f_terms) in [list, np.ndarray, basenp.ndarray]:
        f_terms = {i: f_terms[i] for i in range(len(f_terms))}

    # handle the case n=0
    if n == 0:
        return {
            'modewise_lrns': lambdas * 0,
            'e0': 1,
            'mse_train': sum([coeff ** 2 for coeff in f_terms.values()]),
            'mse_test': sum([coeff ** 2 for coeff in f_terms.values()])
        }

    # if eigenvals aren't provided, compute them from the kernel + synthetic domain
    if lambdas is None:
        if domain == 'circle':
            lambdas, mults = unit_circle_eigenvalues(kernel_fn, kwargs['M']), 1
        if domain == 'hypercube':
            lambdas, mults = hypercube_eigenvalues(kernel_fn, kwargs['d'])
        if domain == 'hypersphere':
            lambdas, mults = hypersphere_eigenvalues(kernel_fn, kwargs['d'], k_max=70)
        else:
            assert 'invalid domain' == True

    # compute lrns and e0
    kappa = find_kappa(n, lambdas, mults=mults, ridge=ridge)
    lrns = eigenmode_learnabilities(n, lambdas, mults=mults, ridge=ridge, kappa=kappa)
    e0 = noise_fitting_factor(n, lambdas, mults=mults, ridge=ridge, kappa=kappa)

    # compute mse
    mse_te = 0
    for f_term in f_terms:
        k = f_term if isinstance(f_term, int) else f_term[0]
        mse_te += e0 * (1 - lrns[k]) ** 2 * f_terms[f_term] ** 2

    if 'noise_std' in kwargs:
        mse_te += e0 * kwargs['noise_std'] ** 2

    mse_tr = (ridge / (n * kappa)) ** 2 * mse_te

    return {
        'modewise_lrns': lrns,
        'e0': e0.item(),
        'mse_train': mse_tr.item(),
        'mse_test': mse_te.item()
    }