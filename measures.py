import time

import jax.numpy as np
from jax import random

import neural_tangents as nt

import scipy as sp

from unit_circle import get_unit_circle_dataset, unit_circle_eigenvalues
from hypercube import get_hypercube_dataset, hypercube_eigenvalues
from hypersphere import get_hypersphere_dataset, hypersphere_eigenvalues
from image_datasets import get_image_dataset

import utils
from utils import kernel_predictions, net_predictions

def kernel_measures(kernel_fn, dataset, g_fns=[], k_type='ntk', diag_reg=0, compute_acc=False):
  """Return learning measures for a kernel on a particular dataset

  kernel_fn -- a JAX kernel function
  dataset -- a double tuple containing (train_X, train_y), (test_X, test_y)
  g_fns -- alternate target functions of the same shape as test_y to find the coefficients of
  k_type -- 'ntk' or 'nngp
  """

  t0 = time.time()
  test_y_hat = kernel_predictions(kernel_fn, dataset, k_type, diag_reg=diag_reg)
  t = time.time() - t0

  (_, _), (_, test_y) = dataset

  # lrn = ((test_y * test_y_hat).mean() / (test_y ** 2).mean()).item()
  # mse = ((test_y - test_y_hat) ** 2).mean().item()
  # l1_loss = np.abs(test_y - test_y_hat).mean().item()

  lrn = utils.lrn(test_y, test_y_hat)
  mse = utils.mse(test_y, test_y_hat)
  l1_loss = utils.l1_loss(test_y, test_y_hat)
  acc = utils.acc(test_y, test_y_hat)

  g_coeffs = [(g * test_y_hat).mean().item() for g in g_fns]

  # acc = (
  #   (test_y * test_y_hat > 0).mean().item() if test_y.shape[1] == 1 else (np.argmax(test_y, axis=1) == np.argmax(test_y_hat, axis=1)).mean()
  # ) if compute_acc else np.nan

  # compute the bound in Arora et al. (https://arxiv.org/abs/1901.08584)
  (train_X, train_y), (test_X, test_y) = dataset
  train_X_normed = train_X / ((train_X**2).sum(axis=1)**.5)[:,None]
  K_dd = kernel_fn(train_X_normed, train_X_normed, get=k_type)
  arora_et_al_bound = np.sqrt(train_y.T @ np.linalg.inv(K_dd) @ train_y / len(train_y)).diagonal().mean().item()

  return {
    'lrn': lrn,
    'mse': mse,
    'l1_loss': l1_loss,
    'acc': acc,
    'g_coeffs': g_coeffs,
    't': t,
    'arora_et_al_bound': arora_et_al_bound
  }

def net_measures(net_fns, dataset, g_fns, n_epochs, lr, subkey, stop_mse=0, print_every=None, compute_acc=False):
  """Return learning measures for a network architecture on a particular dataset

  net_fns -- a JAX init_fn, apply_fn (uncentered), and kernel_fn (unused here)
  dataset -- a double tuple containing (train_X, train_y), (test_X, test_y)
  g_fns -- alternate target functions of the same shape as test_y to find the coefficients of
  n_epochs -- the number of epochs to train
  lr -- the learning rate
  subkey -- the random key to use for initialization
  stop_mse -- a lower threshold for training MSE; training stops if it's passed
  print_every -- if not None, train and test metrics are printed every print_every epochs
  """
  (train_X, train_y), (test_X, test_y) = dataset

  t0 = time.time()
  net_results = net_predictions(net_fns, dataset, n_epochs, lr, subkey, stop_mse=stop_mse, print_every=print_every, compute_acc=compute_acc)
  t = time.time() - t0

  test_y_hat = net_results['test_preds']
  train_y_hat = net_results['train_preds']
  epcs = net_results['epcs']

  # lrn = ((test_y*test_y_hat).mean()/(test_y**2).mean()).item()
  # mse = ((test_y - test_y_hat)**2).mean().item()
  # l1_loss = np.abs(test_y - test_y_hat).mean().item()
  # acc = (
  #   (test_y * test_y_hat > 0).mean().item() if test_y.shape[1] == 1 else (np.argmax(test_y, axis=1) == np.argmax(test_y_hat, axis=1)).mean()
  # ) if compute_acc else np.nan
  # train_mse = ((train_y - train_y_hat)**2).mean().item()

  lrn = utils.lrn(test_y, test_y_hat)
  mse = utils.mse(test_y, test_y_hat)
  l1_loss = utils.l1_loss(test_y, test_y_hat)
  acc = utils.acc(test_y, test_y_hat)
  train_mse = utils.mse(train_y, train_y_hat)

  g_coeffs = [(g*test_y_hat).mean().item() for g in g_fns]

  return {
    'lrn': lrn,
    'mse': mse,
    'l1_loss': l1_loss,
    'acc': acc,
    'g_coeffs': g_coeffs,
    'train_mse': train_mse,
    'epcs': epcs,
    't': t
  }


def learning_measure_statistics(net_fns, domain, n, f_terms=None, g_terms=[], pred_type='both', n_trials=1, **kwargs):
  """Return experimental learning measures for a network architecture on a particular dataset

  net_fns -- a JAX init_fn, apply_fn (uncentered), and kernel_fn
  domain -- 'circle', 'hypercube', or 'hypersphere'
  n -- the trainset size
  f_terms -- coefficients of the target function f
  g_terms -- list of coefficients of the probe functions g
  pred_type -- 'net', 'kernel', or 'both'
  n_trials -- the number of trials (sampled datasets w/random initializations) to average over
  kwargs -- other parameters to pass to prediction functions
  """
  if 'seed' in kwargs:
    key = np.array([0, kwargs['seed']], dtype='uint32') if isinstance(kwargs['seed'], int) else kwargs['seed']
  else:
    key = np.array([0, 17], dtype='uint32')

  assert domain in ['circle', 'hypercube', 'hypersphere', 'mnist', 'fmnist', 'cifar10', 'cifar100']
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

    elif domain == 'hypercube':
      X, targets = get_hypercube_dataset(kwargs['d'], [f_terms] + g_terms)
      f_X = targets[0]
      g_fns = targets[1:]
      D, [f_D] = get_hypercube_dataset(kwargs['d'], [f_terms], full=False, n=n, subkey=subkey)

    elif domain == 'hypersphere':
      X, targets = get_hypersphere_dataset(kwargs['d'], [f_terms] + g_terms, kwargs['n_test'], subkey)
      f_X = targets[0]
      g_fns = targets[1:]
      D, [f_D] = get_hypersphere_dataset(kwargs['d'], [f_terms], n, subkey2)

    elif domain in ['mnist', 'fmnist', 'cifar10']:
      D, f_D, X, f_X = get_image_dataset(domain, n_train=n, n_test=kwargs['n_test'], subkey=subkey, classes=(kwargs['classes'] if 'classes' in kwargs else None))
      g_fns = []

    elif domain in ['mnist normalized', 'fmnist normalized', 'cifar10 normalized']:
      D, f_D, X, f_X = get_image_dataset(domain.split()[0], n_train=n, n_test=kwargs['n_test'], subkey=subkey, normalized=True, classes=(kwargs['classes'] if 'classes' in kwargs else None))
      g_fns = []

    else:
      print(f'invalid domain {domain}')
      assert False

    # if domain == 'mnist':
    #   D, f_D, X, f_X = get_mnist_dataset(n_train=n, n_test=kwargs['n_test'], subkey=subkey, classes=(kwargs['classes'] if 'classes' in kwargs else None))
    #   g_fns = []
    #
    # if domain == 'fmnist':
    #   D, f_D, X, f_X = get_fashion_mnist_dataset(n_train=n, n_test=kwargs['n_test'], subkey=subkey, classes=(kwargs['classes'] if 'classes' in kwargs else None))
    #   g_fns = []
    #
    # if domain == 'cifar10':
    #   D, f_D, X, f_X = get_cifar10_dataset(n_train=n, n_test=kwargs['n_test'], subkey=subkey, classes=(kwargs['classes'] if 'classes' in kwargs else None))
    #   g_fns = []

    dataset = ((D, f_D), (X, f_X))

    measures_k = kernel_measures(net_fns[2],
                                 dataset,
                                 g_fns=g_fns,
                                 k_type='ntk',
                                 compute_acc=kwargs['compute_acc'] if 'compute_acc' in kwargs else False
                                 ) if pred_type in ['kernel', 'both'] else {}

    measures_n = net_measures(net_fns,
                              dataset,
                              g_fns,
                              kwargs['n_epochs'],
                              kwargs['lr'],
                              subkey,
                              stop_mse=kwargs['stop_mse'] if 'stop_mse' in kwargs else 0,
                              print_every=kwargs['print_every'] if 'print_every' in kwargs else None,
                              compute_acc=kwargs['compute_acc'] if 'compute_acc' in kwargs else False
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
  """Return predicted learning measures for a network architecture on a particular dataset

  kernel_fn -- a JAX kernel function
  domain -- 'circle', 'hypercube', or 'hypersphere'
  n -- the trainset size
  f_terms -- coefficients of the target function f
  g_terms -- list of coefficients of the probe functions g
  pred_type -- 'net', 'kernel', or 'both'
  kwargs -- other optional parameters
  """
  if n == 0:
    L = 0
    E = sum([coeff ** 2 for coeff in f_terms.values()])
    g_coeff_preds = [(0, 0) for _ in g_terms]

    return {
      'lrn': L,
      'mse': E,
      'g_coeffs': g_coeff_preds
    }

  # find eigenvalues if they're not already given
  if ('lambdas' in kwargs) and ('mults' in kwargs):
    lambdas, mults = kwargs['lambdas'], kwargs['mults']
  else:
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