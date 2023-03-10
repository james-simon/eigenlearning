import jax.numpy as jnp
from jax import jit, grad
from jax.example_libraries import optimizers

import neural_tangents as nt
from neural_tangents import stax


def get_net_fns(width, d_out, n_hidden_layers=1, W_std=2**.5, b_std=.1, phi=stax.Relu, phi_deg=40):
    """Generate JAX functions for a fully-connected network given hyperparameters.

    width -- the width of the hidden layers
    d_out -- the output dimension
    n_hidden_layers -- the number of hidden layers
    W_std -- the initialization standard deviation of the trainable weight parameters
    b_std -- the initialization standard deviation of the trainable bias parameters
    phi -- the activation function; should be either 'relu', 'erf', or a function
    deg -- the approximation degree for computing the kernel if phi's an arbitrary function
    """
    if phi in [stax.Relu, stax.Erf]:
        layers = [stax.Dense(width, W_std=W_std, b_std=b_std), phi()] * n_hidden_layers
    else:
        layers = [stax.Dense(width, W_std=W_std, b_std=b_std),
                  stax.ElementwiseNumerical(fn=phi, deg=phi_deg)] * n_hidden_layers
    layers += [stax.Dense(d_out, W_std=1, b_std=0)]

    init_fn, apply_fn_uncentered, kernel_fn = stax.serial(*layers)
    return init_fn, apply_fn_uncentered, kernel_fn


def train_kernel(kernel_fn, dataset, k_type='ntk', ridge=0):
    train_X, train_y, test_X, test_y = dataset

    assert len(train_X) > 0
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_X, train_y,
                                                          diag_reg=ridge, diag_reg_absolute_scale=True)
    train_y_hat = predict_fn(x_test=train_X, get=k_type, compute_cov=False)
    test_y_hat = predict_fn(x_test=test_X, get=k_type, compute_cov=False)

    return train_y_hat, test_y_hat


def train_net(net_fns, dataset, loss, subkey, n_epochs, lr, stop_mse=0, print_every=None):
    """Train a neural network and return its final predictions.

    net_fns -- a JAX init_fn, apply_fn (uncentered), and kernel_fn (unused here)
    dataset -- a double tuple containing (train_X, train_y), (test_X, test_y)
    loss -- 
    n_epochs -- the number of epochs to train
    lr -- the learning rate
    subkey -- the random key to use for initialization
    stop_mse -- a lower threshold for training MSE; training stops if it's passed
    print_every -- if not None, train and test metrics are printed every print_every epochs
    """
    train_X, train_y, test_X, test_y = dataset

    assert len(train_X) > 1

    init_fn, apply_fn_uncentered, _ = net_fns
    _, initial_params = init_fn(subkey, (-1, train_X.shape[1]))
    # this generates a centered apply_fn which is zero at initialization
    apply_fn = jit(lambda params, x: apply_fn_uncentered(params, x) - apply_fn_uncentered(initial_params, x))

    opt_init, opt_apply, get_params = optimizers.sgd(lr)
    state = opt_init(initial_params)

    grad_loss = jit(grad(lambda params, x, y: loss(y, apply_fn(params, x))))

    if print_every is not None:
        print('Epoch\t\tTrain Loss\tTest Loss')
    for i in range(n_epochs):
        params = get_params(state)

        # print current state (using params before step)
        if print_every is not None and i % print_every == 0:
            train_y_hat, test_y_hat = apply_fn(params, train_X), apply_fn(params, test_X)
            train_loss, test_loss = loss(train_y, train_y_hat), loss(test_y, test_y_hat)
            print('{}\t\t{:.8f}\t{:.8f}'.format(i, train_loss, test_loss))

        # perform GD step
        state = opt_apply(i, grad_loss(params, train_X, train_y), state)

        # check whether train loss is sufficiently low every 10 epochs
        if i % 10 == 0:
            train_loss = loss(train_y, apply_fn(params, train_X))
            if train_loss < stop_mse:
                break

    params = get_params(state)
    train_y_hat = apply_fn(params, train_X)
    test_y_hat = apply_fn(params, test_X)

    misc = {
        'params': params,
        'apply_fn': apply_fn,
        'stop_epoch': i + 1,
    }

    return train_y_hat, test_y_hat, misc


def compute_metrics(y, y_hat):
    y, y_hat = y.squeeze(), y_hat.squeeze()
    learnability = ((y * y_hat).sum() / (y ** 2).sum()).item()
    if len(y.shape) == 1:
        mse = ((y - y_hat) ** 2).mean().item()
        l1_loss = jnp.abs(y - y_hat).mean().item()
        acc = (y * y_hat > 0).mean().item()
    else:
        mse = ((y - y_hat) ** 2).sum(axis=1).mean().item()
        l1_loss = jnp.abs(y - y_hat).sum(axis=1).mean().item()
        acc = (jnp.argmax(y, axis=1) == jnp.argmax(y_hat, axis=1)).mean().item()
    return learnability, mse, l1_loss, acc
