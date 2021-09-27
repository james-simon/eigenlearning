import time

import jax.numpy as np
from jax import random

import neural_tangents as nt

import scipy as sp

from unit_circle import get_unit_circle_dataset, unit_circle_eigenvalues
from hypercube import get_hypercube_dataset, hypercube_eigenvalues
from hypersphere import get_hypersphere_dataset, hypersphere_eigenvalues

from utils import net_predictions

def kernel_measures(net_fns, dataset, g_fns=[], k_type='ntk'):
  _, _, kernel_fn = net_fns
  (train_X, train_y), (test_X, test_y) = dataset

  t0 = time.time()
  if len(train_X) > 0:
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_X, train_y)
    test_y_hat = predict_fn(x_test=test_X, get=k_type, compute_cov=False)
  else:
    test_y_hat = np.zeros(shape=(len(test_X), 1))
  t = time.time() - t0

  lrn = ((test_y * test_y_hat).mean() / (test_y ** 2).mean()).item()
  mse = ((test_y - test_y_hat) ** 2).mean().item()
  g_coeffs = [(g * test_y_hat).mean().item() for g in g_fns]

  return {
    'lrn': lrn,
    'mse': mse,
    'g_coeffs': g_coeffs,
    't': t
  }

def net_measures(net_fns, dataset, g_fns, n_epochs, lr, subkey, stop_mse=0, print_every=None):
  (train_X, train_y), (test_X, test_y) = dataset

  t0 = time.time()
  net_results = net_predictions(net_fns, dataset, n_epochs, lr, subkey, stop_mse=stop_mse, print_every=print_every)
  t = time.time() - t0

  test_y_hat = net_results['test_preds']
  train_y_hat = net_results['train_preds']
  epcs = net_results['epcs']

  lrn = ((test_y*test_y_hat).mean()/(test_y**2).mean()).item()
  mse = ((test_y - test_y_hat)**2).mean().item()
  g_coeffs = [(g*test_y_hat).mean().item() for g in g_fns]
  train_mse = ((train_y - train_y_hat)**2).mean().item()

  return {
    'lrn': lrn,
    'mse': mse,
    'g_coeffs': g_coeffs,
    'train_mse': train_mse,
    'epcs': epcs,
    't': t
  }


def learning_measure_statistics(net_fns, domain, n, f_terms, g_terms=[], pred_type='both', n_trials=1, **kwargs):
  if 'seed' in kwargs:
    key = np.array([0, kwargs['seed']], dtype='uint32') if isinstance(kwargs['seed'], int) else kwargs['seed']
  else:
    key = np.array([0, 17], dtype='uint32')

  assert domain in ['circle', 'hypercube', 'hypersphere']
  if pred_type in ['net', 'both']:
    assert ('n_epochs' in kwargs) and ('lr' in kwargs)

  measures = {
    'kernel': {},
    'net': {}
  }

  for i in range(n_trials):
    key, subkey, subkey2 = random.split(key, 3)

    if domain == 'circle':
      _, X, targets = get_unit_circle_dataset(kwargs['M'], [f_terms] + g_terms)
      f_X = targets[0]
      g_fns = targets[1:]
      _, D, [f_D] = get_unit_circle_dataset(kwargs['M'], [f_terms], full=False, n=n, subkey=subkey)

    if domain == 'hypercube':
      X, targets = get_hypercube_dataset(kwargs['d'], [f_terms] + g_terms)
      f_X = targets[0]
      g_fns = targets[1:]
      D, [f_D] = get_hypercube_dataset(kwargs['d'], [f_terms], full=False, n=n, subkey=subkey)

    if domain == 'hypersphere':
      X, targets = get_hypersphere_dataset(kwargs['d'], [f_terms] + g_terms, kwargs['n_test'], subkey)
      f_X = targets[0]
      g_fns = targets[1:]
      D, [f_D] = get_hypersphere_dataset(kwargs['d'], [f_terms], n, subkey2)

    dataset = ((D, f_D), (X, f_X))

    measures_k = kernel_measures(net_fns,
                                 dataset,
                                 g_fns=g_fns,
                                 k_type='ntk') if pred_type in ['kernel', 'both'] else {}

    measures_n = net_measures(net_fns,
                              dataset,
                              g_fns,
                              kwargs['n_epochs'],
                              kwargs['lr'],
                              subkey,
                              stop_mse=kwargs['stop_mse'] if 'stop_mse' in kwargs else 0,
                              print_every=kwargs['print_every'] if 'print_every' in kwargs else None
                              ) if pred_type in ['net', 'both'] else {}

    for m in measures_k:
      if m not in measures['kernel']:
        measures['kernel'][m] = []
      measures['kernel'][m] += [measures_k[m]]

    for m in measures_n:
      if m not in measures['net']:
        measures['net'][m] = []
      measures['net'][m] += [measures_n[m]]

  for ptype in ['kernel', 'net']:
    for m in measures[ptype]:
      if m is not 'g_coeffs':
        vals = np.array(measures[ptype][m])
        measures[ptype][m] = (vals.mean().item(), vals.std(ddof=1).item())
      else:
        vals = np.array(measures[ptype][m])
        means = vals.mean(axis=0)
        stds = vals.std(axis=0, ddof=1)
        measures[ptype][m] = [(means[i].item(), stds[i].item()) for i in range(len(means))]

  return measures

# helper function used in calculating C
def L_sum(C, lambdas, mults=1):
  return (mults * lambdas / (lambdas + C)).sum()

# find C for a given eigensystem and n
def find_C(n, lambdas, mults=1):
  return sp.optimize.fsolve(lambda C: L_sum(C, lambdas, mults=mults) - n, sorted(lambdas, reverse=True)[min([round(n), len(lambdas) - 1])])


def learning_measure_predictions(kernel_fn, domain, n, f_terms, g_terms=[], **kwargs):
  if n == 0:
    L = 0
    E = sum([coeff ** 2 for coeff in f_terms.values()])
    g_coeff_preds = [(0, 0) for _ in g_terms]

    return {
      'lrn': L,
      'mse': E,
      'g_coeffs': g_coeff_preds
    }

  if ('lambdas' in kwargs) and ('mults' in kwargs):
    lambdas, mults = kwargs['lambdas'], kwargs['mults']
  else:
    # find eigenvalues
    if domain == 'circle':
      lambdas, mults = unit_circle_eigenvalues(kernel_fn, kwargs['M']), 1
    if domain == 'hypercube':
      lambdas, mults = hypercube_eigenvalues(kernel_fn, kwargs['d'])
    if domain == 'hypersphere':
      lambdas, mults = hypersphere_eigenvalues(kernel_fn, kwargs['d'], k_max=70)

  # calculate C and q
  C = find_C(n, lambdas, mults).item()
  q = (mults * lambdas / (lambdas + C) ** 2).sum().item()

  # calculate L
  L_num, L_denom = 0, 0
  for f_term in f_terms:
    k = f_term if isinstance(f_term, int) else f_term[0]
    L_num += f_terms[f_term] ** 2 * lambdas[k] / (lambdas[k] + C)
    L_denom += f_terms[f_term] ** 2
  L = (L_num / L_denom).item()

  # calculate E
  E = 0
  for f_term in f_terms:
    E += (n * C / q) * f_terms[f_term] ** 2 / (lambdas[k] + C) ** 2
  E = E.item()

  # calcualte g_coeffs
  g_coeff_preds = []

  for g_termset in g_terms:
    g_coeff_mean = 0

    for tf_i in f_terms:
      if tf_i in g_termset:
        k = f_term if isinstance(f_term, int) else f_term[0]
        g_coeff_mean += (f_terms[tf_i] * g_termset[tf_i] * lambdas[k] / (lambdas[k] + C)).item()

    f_factor = 0
    g_factor = 0
    for tf_i in f_terms:
      k = tf_i if isinstance(tf_i, int) else tf_i[0]
      f_factor += f_terms[tf_i] ** 2 / (lambdas[k] + C) ** 2
    for tg_i in g_termset:
      k = tg_i if isinstance(tg_i, int) else tg_i[0]
      g_factor += g_termset[tg_i] ** 2 * lambdas[k] ** 2 / (lambdas[k] + C) ** 2

    g_coeff_var = ((C / q) * f_factor * g_factor).item()
    g_coeff_std = g_coeff_var ** .5
    g_coeff_preds += [(g_coeff_mean, g_coeff_std)]

  return {
    'lrn': L,
    'mse': E,
    'g_coeffs': g_coeff_preds
  }