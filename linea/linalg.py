import numpy as np


def linreg(X, flux, error):
    """
    Ordinary least squares linear regression.

    Parameters
    ----------
    X : `~numpy.ndarray`
    flux : `~numpy.ndarray`
    errors : `~numpy.ndarray`

    Returns
    -------
    betas : `~numpy.ndarray`

    cov : `~numpy.ndarray`

    """
    inv_N = np.linalg.inv(np.diag(error)**2)
    XT_invN_X = np.linalg.inv(X.T @ inv_N @ X)
    betas = XT_invN_X @ X.T @ inv_N @ flux
    cov = XT_invN_X
    return betas, cov


class RegressionResult(object):
    """
    Result from a linear regression
    """
    def __init__(self, design_matrix, betas, cov):
        self.X = design_matrix
        self.betas = betas
        self.cov = cov

        self.best_fit = self.X @ betas
