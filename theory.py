import time

import jax.numpy as jnp
import numpy as np
import scipy as sp

# find kappa for a given eigensystem and n
def find_kappa(n, lambdas, mults=1, ridge=0):
    
    def lrn_sum(kappa):
        return (mults * lambdas / (lambdas + kappa)).sum()
    
    if isinstance(mults, int):
        idx_n = min(n, len(lambdas))
    else:
        idxs = jnp.where(mults.cumsum() >= n)[0]
        idx_n = idxs.min() if len(idxs) > 0 else None

    # try optimizing in logspace...
    kappa_0 = max(ridge / n, (sorted(lambdas, reverse=True)[idx_n] if idx_n is not None else 10 ** -9))
    log_kappa = sp.optimize.fsolve(
        lambda log_kappa: lrn_sum(jnp.exp(log_kappa)) + ridge / jnp.exp(log_kappa) - n, jnp.log(kappa_0))
    kappa = jnp.exp(log_kappa).item()

    # if that failed, try optimizing in linear space...
    # error = lrn_sum(kappa, lambdas, mults=mults) + ridge / kappa - n
    # if np.abs(error) > n / 100:
    #     kappa = sp.optimize.fsolve(
    #         lambda k: lrn_sum(kappa, lambdas, mults=mults) + ridge / k - n,
    #         kappa_0).item()

    # if uncommented: check for failure again and throw an exception if it fails
    # error = lrn_sum(kappa, lambdas, mults=mults) + ridge / kappa - n
    # if np.abs(error) > n / 100:
    #   raise ValueError('kappa optimization failed!')

    return kappa

# compute eigenmode learnabilities
def get_eigenmode_learnabilities(lambdas, kappa):
    if isinstance(lambdas, list):
        lambdas = jnp.array(lambdas)
    return lambdas / (lambdas + kappa)

# compute pure-noise MSE \mathcal{E}_0
def noise_fitting_factor(eigenlearnabilities, n, mults=1):
    lrn_sum = (mults * eigenlearnabilities**2).sum()
    return n / (n - lrn_sum)

def theoretical_predictions(n, f_terms, lambdas, mults=1, ridge=0, **kwargs):
    # if f_terms is not in dictionary form, make it so (helps with compatibility with synthetic domains)
    if type(f_terms) in [list, jnp.ndarray, np.ndarray]:
        f_terms = {i: f_terms[i] for i in range(len(f_terms))}

    # handle the case n=0
    if n == 0:
        return {
            'modewise_lrns': lambdas * 0,
            'e0': 1,
            'mse_train': sum([coeff ** 2 for coeff in f_terms.values()]),
            'mse_test': sum([coeff ** 2 for coeff in f_terms.values()])
        }

    # compute lrns and e0
    kappa = find_kappa(n, lambdas, mults=mults, ridge=ridge)
    lrns = get_eigenmode_learnabilities(lambdas, kappa)
    e0 = noise_fitting_factor(lrns, n, mults)

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