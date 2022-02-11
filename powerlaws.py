import numpy as np
import scipy


def nearest_idx(arr, val):
    return np.abs(arr - val).argmin().item()


def powerlaw_fit(xs, ys, a1=None, a2=None, logsample=False):
    xs, ys = np.array(xs), np.array(ys)

    xmin = xs[0] if a1 is None else xs[0] * (xs[-1] / xs[0]) ** a1
    xmax = xs[-1] if a2 is None else xs[0] * (xs[-1] / xs[0]) ** a2

    if not logsample:
        sample_is = np.where((xs >= xmin) & (xs <= xmax))[0]
    else:
        sample_is = [nearest_idx(xs, x) for x in np.logspace(np.log10(xmin), np.log10(xmax))]

    fit = scipy.stats.linregress(np.log(xs[sample_is]), np.log(ys[sample_is]))
    return {'params': (np.e ** fit.intercept, -fit.slope),
            'r': -fit.rvalue,
            'fit_idxs': sample_is}


def shifted_powerlaw_fit(idxs, vals, a1=None, a2=None, logsample=True):
    best_fit = None

    for dl in np.logspace(-10, -2, 100):
        fit = powerlaw_fit(idxs, vals - dl, a1=a1, a2=a2, logsample=logsample)
        fit['params'] += (dl,)
        if best_fit is None or fit['r'] > best_fit['r']:
            best_fit = fit

    return best_fit


def compute_tailsums(xs):
    return np.flip(np.flip(xs).cumsum())


def tailsum_powerlaw_fit(idxs, vals, a1=None, a2=None, logsample=True):
    best_fit = None

    for ds in np.logspace(-5, 0, num=100):
        fit = powerlaw_fit(idxs, vals + (idxs - 1) / idxs[-1] * ds, a1=a1, a2=a2, logsample=logsample)
        fit['params'] += (ds,)
        if best_fit is None or fit['r'] > best_fit['r']:
            best_fit = fit

    return best_fit
