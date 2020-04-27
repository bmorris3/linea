import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

from astropy.io import fits
from astropy.time import Time
from astropy.stats import SigmaClip, mad_std

from .linalg import linreg, RegressionResult

__all__ = ['CheopsLightCurve', 'JointLightCurve']


def normalize(vector):
    """
    Normalize a vector such that its contents range from [-0.5, 0.5]
    """
    return (vector - vector.min()) / vector.ptp() - 0.5


class CheopsLightCurve(object):
    """
    Data handling class for CHEOPS light curves.
    """

    def __init__(self, record_array, norm=True):
        """
        Parameters
        ----------
        record_array : `~numpy.recarray`
            Record array of column vectors and their labels (names). Often
            this record array comes straight from a FITS file.
        norm : bool
            Normalize the fluxes such that the median flux is unity. Default is
            True.
        """
        self.recs = record_array

        for key in self.recs.columns.names:
            setattr(self, key.lower(), self.recs[key])

        self.time = Time(self.bjd_time, format='jd')
        self.mask = (np.isnan(self.flux) | self.status.astype(bool) |
                     self.event.astype(bool))

        if norm:
            self.fluxerr = self.fluxerr / np.nanmedian(self.flux)
            self.flux = self.flux / np.nanmedian(self.flux)

    @classmethod
    def from_fits(cls, path, norm=True):
        """
        Load a FITS file from DACE or the DRP.

        Parameters
        ----------
        path : str
            Path to the FITS file containing the data to load.
        norm : bool
            Normalize the fluxes such that the median flux is unity. Default is
            True.
        """
        return cls(fits.getdata(path), norm=norm)

    @classmethod
    def from_example(cls, norm=True):
        """
        Load example 55 Cnc e light curve (**NOTE**: this is not real data).

        Parameters
        ----------
        norm : bool
            Normalize the fluxes such that the median flux is unity. Default is
            True.
        """
        path = os.path.join(os.path.dirname(__file__), 'data', 'example.fits')
        return cls.from_fits(path, norm=norm)

    def plot(self, ax=None, **kwargs):
        """
        Plot the light curve.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axis instance on which to build the plot
        kwargs: dict
            Further keyword arguments to pass to `~matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axis instance with the light curve plotted on it.
        """
        if ax is None:
            ax = plt.gca()

        ax.errorbar(self.bjd_time[~self.mask], self.flux[~self.mask],
                    self.fluxerr[~self.mask], **kwargs)

        return ax

    def design_matrix(self, norm=True):
        """
        Generate the design matrix.

        Parameters
        ----------
         norm : bool
            Normalize the column vectors within the design matrix such that they
            have mean=zero and range=unity.

        Returns
        -------
        X : `~numpy.ndarray`
            Design matrix (concatenated column vectors of observables)
        """
        if norm:
            X = np.vstack([
                ((self.bjd_time - self.bjd_time.mean()) /
                 self.bjd_time.ptp()),
                normalize(self.roll_angle),
                normalize(self.roll_angle**2),
                normalize(self.centroid_x - self.centroid_x.mean()),
                normalize(self.centroid_y - self.centroid_y.mean()),
                normalize((self.centroid_x - self.centroid_x.mean())**2),
                normalize((self.centroid_y - self.centroid_y.mean())**2),
                normalize(self.conta_lc),
                normalize(self.dark),
                np.ones(len(self.bjd_time)),
            ]).T

        else:
            X = np.vstack([
                (self.bjd_time - self.bjd_time.mean()),
                self.roll_angle,
                self.roll_angle**2,
                self.centroid_x - self.centroid_x.mean(),
                self.centroid_y - self.centroid_y.mean(),
                (self.centroid_x - self.centroid_x.mean())**2,
                (self.centroid_y - self.centroid_y.mean())**2,
                self.conta_lc,
                self.dark,
                np.ones(len(self.bjd_time)),
            ]).T

        return X[~self.mask]

    def sigma_clip_centroid(self, sigma=3.5, plot=False):
        """
        Sigma-clip the light curve on centroid position (update mask).

        Parameters
        ----------
        sigma : float
            Factor of standard deviations away from the median centroid position
            to clip on.
        plot : bool
            Plot the accepted centroids (in black) and the centroids of the
            rejected fluxes (in red).
        """
        x_mean = np.median(self.centroid_x)
        y_mean = np.median(self.centroid_y)
        x_std = mad_std(self.centroid_x)
        y_std = mad_std(self.centroid_y)

        outliers = (sigma * min([x_std, y_std]) <
                    np.hypot(self.centroid_x - x_mean,
                             self.centroid_y - y_mean))

        if plot:
            plt.scatter(self.centroid_x[~outliers], self.centroid_y[~outliers],
                        marker=',', color='k')
            plt.scatter(self.centroid_x[outliers], self.centroid_y[outliers],
                        marker='.', color='r')
            plt.xlabel('BJD')
            plt.ylabel('Flux')

        self.mask |= outliers

    def sigma_clip_flux(self, sigma_upper=4, sigma_lower=4, maxiters=None,
                        plot=False):
        """
        Sigma-clip the light curve on fluxes (update mask).

        Parameters
        ----------
        sigma_upper : float
            Factor of standard deviations above the median centroid position
            to clip on.
        sigma_lower : float
            Factor of standard deviations below the median centroid position
            to clip on.
        maxiters : float or None
            Number of sigma-clipping iterations. Default is None, which repeats
            until there are no outliers left.
        plot : float
            Plot the accepted fluxes (in black) and the rejected fluxes (in red)
        """
        sc = SigmaClip(sigma_upper=sigma_upper, sigma_lower=sigma_lower,
                       stdfunc=mad_std, maxiters=maxiters)
        self.mask[~self.mask] |= sc(self.flux[~self.mask]).mask

        if plot:
            plt.plot(self.bjd_time[self.mask], self.flux[self.mask], 'r.')
            plt.plot(self.bjd_time[~self.mask], self.flux[~self.mask], 'k.')
            plt.xlabel('BJD')
            plt.ylabel('Flux')

    def regress(self, design_matrix):
        r"""
        Regress the design matrix against the fluxes.

        Parameters
        ----------
        design_matrix : `~numpy.ndarray`
            Design matrix (concatenated column vectors of observables)

        Returns
        -------
        betas : `~numpy.ndarray`
            Least squares estimators :math:`\hat{\beta}`
        cov : `~numpy.ndarray`
            Covariance matrix for the least squares estimators
            :math:`\sigma_{\hat{\beta}}^2`
        """
        b, c = linreg(design_matrix,
                      self.flux[~self.mask],
                      self.fluxerr[~self.mask])

        return RegressionResult(design_matrix, b, c)

    def plot_phase_curve(self, r, params, t_fine, transit_fine, sinusoid_fine,
                         t0_offset=0, n_regressors=2, bins=15):
        """
        Plot the best-fit phase curve.

        Parameters
        ----------
        r : `~linea.RegressionResult`
            Result of the linear regression
        params : `~batman.TransitParams`
            Transiting exoplanet parameters
        t_fine : `~numpy.ndarray`
            Times computed on a grid finer than the original observations
        transit_fine : `~numpy.ndarray`
            Transit model computed at times ``t_fine``
        sinusoid_fine : `~numpy.ndarray`
            Sinusoidal phase curve model computed at times ``t_fine``
        t0_offset : float, optional
            Time offset between the mid-transit time defined by ``params`` and
            the true mid-transit time [days]. Default is zero.
        n_regressors : int, optional
            Number of regressors used to parameterize the phase curve.
            Default is two.
        bins : int, optional
            Number of bins to break the light curve into when plotting (black),
            default is 15.

        Returns
        -------
        fig, ax : `~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`
            Figure and axis objects containing the phase curve plot.
        """
        transit = r.X[:, 0]
        sinusoid = (r.X[:, 1:n_regressors+1] @ r.betas[1:n_regressors+1] /
                    r.best_fit)

        phases = ((self.bjd_time[~self.mask] - params.t0 - t0_offset) %
                  params.per) / params.per
        phases[phases > 0.95] -= 1
        phases_fine = (((t_fine - params.t0 - t0_offset) % params.per) /
                       params.per)
        phases_fine[phases_fine > 0.95] -= 1

        fig, ax = plt.subplots(2, 1, figsize=(4.5, 8), sharex=True)
        ax[0].plot(phases, (transit + 1) * (
                self.flux[~self.mask] / r.best_fit + sinusoid), '.',
                   color='silver')

        bs = binned_statistic(phases,
                              (transit + 1) * (self.flux[~self.mask] /
                                               r.best_fit + sinusoid),
                              bins=bins, statistic='median')
        bincenters = 0.5 * (bs.bin_edges[1:] + bs.bin_edges[:-1])

        ax[0].plot(bincenters, bs.statistic, 's', color='k')

        ax[0].plot(phases_fine[np.argsort(phases_fine)],
                   (transit_fine + sinusoid_fine)[np.argsort(phases_fine)], 'r')

        ax[0].set(ylabel='Phase Curve')

        bs_resid = binned_statistic(phases,
                                    self.flux[~self.mask] / r.best_fit - 1,
                                    bins=bins, statistic='median')

        ax[1].plot(phases, self.flux[~self.mask] / r.best_fit - 1, '.',
                   color='silver')
        ax[1].plot(bincenters, bs_resid.statistic, 's', color='k')

        ax[1].set(ylabel='Residuals', xlabel='Phase')
        ax[1].set_xticks(np.arange(-0, 1, 0.1), minor=True)

        for axis in ax:
            for sp in ['right', 'top']:
                axis.spines[sp].set_visible(False)
        ax[0].ticklabel_format(useOffset=False)

        return fig, ax

    def phase(self, planet_params):
        """
        Orbital phase of planet at times ``lc.bjd_time``.

        Parameters
        ----------
        planet_params : `~linea.Planet`
            Planet parameter object.

        Returns
        -------
        phases : `~numpy.ndarray`
            Orbital phases at times ``lc.bjd_time``
        """
        return (((self.bjd_time - planet_params.t0) % planet_params.per) /
                planet_params.per)


class JointLightCurve(CheopsLightCurve):
    """
    Joint analysis object for multiple CHEOPS light curves.
    """
    def __init__(self, light_curves):

        self.light_curves = light_curves
        self.recs = [lc.recs for lc in light_curves]

        self.attrs = [attr.lower() for attr in
                      light_curves[0].recs.columns.names]

        for attr in self.attrs:
            setattr(self, attr, [getattr(lc, attr) for lc in light_curves])

    def concat(self):
        extra_attrs = ['time', 'mask']
        c = namedtuple('ConcatenatedLightCurve', self.attrs + extra_attrs)
        for attr in self.attrs + extra_attrs:
            setattr(c, attr, np.concatenate([getattr(lc, attr)
                                             for lc in self]))
        return c

    def pad_shapes(self):
        shapes = []
        for lc in self:
            shapes.append(np.count_nonzero(~lc.mask))
        return shapes

    def combined_design_matrix(self, Xs):

        shapes = self.pad_shapes()
        ndim = Xs[0].shape[1]
        Xs_padded = []

        for i in range(len(Xs)):
            before = shapes[:i]
            after = shapes[i+1:]

            prepad = np.zeros((sum(before), ndim)) if len(before) > 0 else None
            postpad = np.zeros((sum(after), ndim)) if len(after) > 0 else None

            segments = []
            for j in [prepad, Xs[i], postpad]:
                if j is not None:
                    segments.append(j)

            Xs_padded.append(np.vstack(segments))

        return np.hstack(Xs_padded)

    def __iter__(self):
        yield from self.light_curves

    def regress(self, design_matrix):
        r"""
        Regress the design matrix against the fluxes.

        Parameters
        ----------
        design_matrix : `~numpy.ndarray`
            Design matrix (concatenated column vectors of observables)

        Returns
        -------
        betas : `~numpy.ndarray`
            Least squares estimators :math:`\hat{\beta}`
        cov : `~numpy.ndarray`
            Covariance matrix for the least squares estimators
            :math:`\sigma_{\hat{\beta}}^2`
        """
        mask = np.concatenate([lc.mask for lc in self])

        b, c = linreg(design_matrix,
                      np.concatenate(self.flux)[~mask],
                      np.concatenate(self.fluxerr)[~mask])

        return RegressionResult(design_matrix, b, c)
