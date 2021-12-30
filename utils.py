import jax
import jax.numpy as np
from jax import jit, grad, vmap
from jax.experimental import optimizers

import neural_tangents as nt
from neural_tangents import stax

def get_net_fns(width, d_out, n_hidden_layers=1, W_std=1.4, b_std=.1, phi='relu', phi_deg=40):
  """Generate JAX functions for a fully-connected network given hyperparameters.

  width -- the width of the hidden layers
  d_out -- the output dimension
  n_hidden_layers -- the number of hidden layers
  W_std -- the initialization standard deviation of the trainable weight parameters
  b_std -- the initialization standard deviation of the trainable bias parameters
  phi -- the activation function; should be either 'relu', 'erf', or a function
  deg -- the approximation degree for computing the kernel if phi's an arbitrary function
  """
  if phi == 'relu':
    layers = [stax.Dense(width, W_std=W_std, b_std=b_std), stax.Relu()] * n_hidden_layers
    layers += [stax.Dense(d_out, W_std=1, b_std=0)]
  elif phi == 'erf':
    layers = [stax.Dense(width, W_std=W_std, b_std=b_std), stax.Erf()] * n_hidden_layers
    layers += [stax.Dense(d_out, W_std=1, b_std=0)]
  else:
    layers = [stax.Dense(width, W_std=W_std, b_std=b_std), stax.ElementwiseNumerical(fn=phi, deg=phi_deg)] * n_hidden_layers
    layers += [stax.Dense(d_out, W_std=1, b_std=0)]

  init_fn, apply_fn_uncentered, kernel_fn = stax.serial(*layers)
  return init_fn, apply_fn_uncentered, kernel_fn


def sample_kernel(kernel_fn, cosines, d, norm=1, k_type='ntk'):
  """Sample different angles of a rotationally-invariant kernel.

  kernel_fn -- the JAX kernel function to sample; assumed to be rotationally-invariant
  cosines -- the cosines of the angles at which to sample the kernel
  d -- the input dimension
  norm -- the norm of the sample input points
  k_type -- either 'ntk' or 'nngp'
  """
  sines = (1 - cosines**2)**.5
  u0 = jax.ops.index_update(np.zeros(d), 0, 1)
  u1 = jax.ops.index_update(np.zeros(d), 1, 1)
  xs = np.outer(cosines, u0) + np.outer(sines, u1)
  xs = norm*xs

  # get the kernel between these points and an endpoint
  Ks = kernel_fn(xs[0:1], xs, k_type)[0][::-1]

  return Ks

def kernel_predictions(kernel_fn, dataset, k_type='ntk', diag_reg=0):
  (train_X, train_y), (test_X, test_y) = dataset

  if len(train_X) > 0:
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_X, train_y, diag_reg=diag_reg, diag_reg_absolute_scale=True)
    test_y_hat = predict_fn(x_test=test_X, get=k_type, compute_cov=False)
  else:
    test_y_hat = np.zeros(shape=(len(test_X), 1))

  return test_y_hat

def net_predictions(net_fns, dataset, n_epochs, lr, subkey, stop_mse=0, snapshot_es=[], print_every=None, compute_acc=False):
  """Train a neural network and return its final predictions.

  net_fns -- a JAX init_fn, apply_fn (uncentered), and kernel_fn (unused here)
  dataset -- a double tuple containing (train_X, train_y), (test_X, test_y)
  n_epochs -- the number of epochs to train
  lr -- the learning rate
  subkey -- the random key to use for initialization
  stop_mse -- a lower threshold for training MSE; training stops if it's passed
  snapshot_es -- epochs at which to capture and return a snapshot of the network's train and test predictions
  print_every -- if not None, train and test metrics are printed every print_every epochs
  """
  (train_X, train_y), (test_X, test_y) = dataset

  if len(train_X) == 0:
    return {
      'train_preds': np.array([]),
      'test_preds': np.zeros_like(test_y),
      'epcs': 0,
      'snapshots': {}
    }

  init_fn, apply_fn_uncentered, _ = net_fns
  _, initial_params = init_fn(subkey, (-1, train_X.shape[1]))
  # this generates a centered apply_fn which is zero at initialization
  apply_fn = jit(lambda params, x: apply_fn_uncentered(params, x) - apply_fn_uncentered(initial_params, x))

  opt_init, opt_apply, get_params = optimizers.sgd(lr)
  state = opt_init(initial_params)

  loss = lambda y, y_hat: np.mean((y - y_hat) ** 2)
  grad_loss = jit(grad(lambda params, x, y: loss(apply_fn(params, x), y)))

  snapshots = {}

  if print_every is not None:
    print('Epoch\t\tTrain Loss\tTest Loss' + ('\t\tTrain Acc\tTest Acc' if compute_acc else ''))
  for i in range(n_epochs):
    params = get_params(state)
    state = opt_apply(i, grad_loss(params, train_X, train_y), state)

    train_loss = loss(apply_fn(params, train_X), train_y)
    if train_loss < stop_mse:
      break

    if print_every is not None and i % print_every == 0:
      train_y_hat, test_y_hat = apply_fn(params, train_X), apply_fn(params, test_X)
      test_loss = loss(test_y_hat, test_y)
      if not compute_acc:
        print('{}\t\t{:.8f}\t{:.8f}'.format(i, train_loss, test_loss))
      else:
        train_acc = (train_y * train_y_hat > 0).mean().item() if test_y.shape[1] == 1 else (
                    np.argmax(train_y, axis=1) == np.argmax(train_y_hat, axis=1)).mean()
        test_acc = (test_y * test_y_hat > 0).mean().item() if test_y.shape[1] == 1 else (
                    np.argmax(test_y, axis=1) == np.argmax(test_y_hat, axis=1)).mean()
        print('{}\t\t{:.8f}\t{:.8f}\t\t{:.8f}\t{:.8f}'.format(i, train_loss, test_loss, train_acc, test_acc))

    if i in snapshot_es:
      train_preds = apply_fn(get_params(state), train_X).tolist()
      test_preds = apply_fn(get_params(state), test_X).tolist()
      snapshots[i] = {'train_preds':train_preds, 'test_preds':test_preds}

  train_preds = apply_fn(get_params(state), train_X)
  test_preds = apply_fn(get_params(state), test_X)

  return {
    'train_preds': train_preds,
    'test_preds': test_preds,
    'epcs': i + 1,
    'snapshots': snapshots
  }