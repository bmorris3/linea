import numpy as np


def linreg(X, flux, error):
    r"""
    Least squares linear regression.

    We solve for the least-squares estimators :math:`\hat{\beta}`,

    .. math::

        \hat{\beta} = ({\bf X}^{\rm T} {\bf N}^{-1} {\bf X})^{-1} {\bf X}^{\rm T} {\bf N}^{-1} f.

    Uncertainties on each of the least-squares estimators are computed with the
    pre-computed matrix inversion,

    .. math::

        \sigma_{\hat{\beta}}^2 = ({\bf X}^{\rm T} {\bf N}^{-1} {\bf X})^{-1}.

    We've defined the uncertainty matrix :math:`\bf N`,

    .. math::

        {\bf N} = {\bf I_M} \sigma_f^2,

    where :math:`\bf I_M` is the identity matrix.

    Parameters
    ----------
    X : `~numpy.ndarray`
        Design matrix (concatenated column vectors)
    flux : `~numpy.ndarray`
        Flux measurements (row vector)
    error : `~numpy.ndarray`
        Uncertainties on each flux measurement (row vector)

    Returns
    -------
    betas : `~numpy.ndarray`
        Least squares estimators :math:`\hat{\beta}`
    cov : `~numpy.ndarray`
        Covariance matrix for the least squares estimators
        :math:`\sigma_{\hat{\beta}}^2`
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
