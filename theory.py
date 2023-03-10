import jax.numpy as jnp
import scipy.optimize as scipy_opt


class Spectrum():
    
    def __init__(self, lambdas, multiplicities=None, kk=None):
        # assert len(jnp.unique(lambdas)) == len(lambdas)
        if multiplicities:
            assert len(multiplicities) == len(lambdas)
        else:
            multiplicities = jnp.ones(len(lambdas))
        if kk:
            assert len(kk) == len(lambdas)
        else:
            kk = list(range(len(lambdas)))
        
        lambdas, multiplicities = jnp.array(lambdas), jnp.array(multiplicities)
        
        sort_order = lambdas.argsort()[::-1]
        self.sort_order = sort_order
        self.n_levels = len(lambdas)
        self.lambdas = lambdas[sort_order]
        self.multiplicities = multiplicities[sort_order]
        self.kk = [kk[i] for i in sort_order]
        self.k_ind = {k:i for i,k in enumerate(self.kk)}
        self.len = self.multiplicities.sum()
        
    def get_mode_eigenlevel(self, k):
        return self.k_ind[k]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.lambdas[index]


# compute eigenmode learnabilities
def get_eigenmode_learnabilities(spectrum, kappa):
    lambdas = spectrum.lambdas
    return lambdas / (lambdas + kappa)


# find kappa for a given eigensystem and n
def find_kappa(n, spectrum, ridge):
    mults = spectrum.multiplicities
    
    def lrn_sum(kappa):
        eigenlrns = get_eigenmode_learnabilities(spectrum, kappa)
        return (eigenlrns * mults).sum()
    
    kappa = scipy_opt.bisect(lambda kap: lrn_sum(kap) + ridge / (kap+1e-14) - n, 0, 1e10)
    return kappa


# compute pure-noise MSE \mathcal{E}_0
def get_overfitting_coefficient(eigenlearnabilities, n, mults):
    return n / (n - (eigenlearnabilities**2 * mults).sum())


def theoretical_predictions(n, eigenlevel_coeffs, spectrum, ridge=0, noise_std=0):
    assert n > 0
    assert len(eigenlevel_coeffs) == spectrum.n_levels
    f = eigenlevel_coeffs
    mults = spectrum.multiplicities

    # compute lrns and e0
    kappa = find_kappa(n, spectrum, ridge)
    eigenlearnabilities = get_eigenmode_learnabilities(spectrum, kappa)
    e0 = get_overfitting_coefficient(eigenlearnabilities, n, mults)

    # compute learnability
    # f are the eigenlevel projection coeffs, so mults are already accounted for
    L = (f**2 * eigenlearnabilities).sum() / (f**2).sum()
    
    # compute mse
    test_mse = e0 * (((1-eigenlearnabilities)**2 * f**2).sum() + noise_std**2)cd D

    train_mse = (ridge / (n * kappa))**2 * test_mse
    
    return {
        "kappa": kappa,
        "learnability": L,
        "overfitting_coeff": e0,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "eigenlearnabilities": eigenlearnabilities
    }
