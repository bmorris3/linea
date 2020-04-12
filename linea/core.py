import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

from astropy.io import fits
from astropy.time import Time
from astropy.stats import SigmaClip, mad_std

from .linalg import linreg, RegressionResult

__all__ = ['CheopsLightCurve']


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
        record_array : `~astropy.io.fits.fitsrec.FITS_rec`
        norm : bool
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
        norm : bool
        """
        return cls(fits.getdata(path), norm=norm)

    @classmethod
    def from_example(cls, norm=True):
        """
        Load example 55 Cnc e light curve.

        Parameters
        ----------
        norm : bool
        """
        path = os.path.join(os.path.dirname(__file__), 'data', 'example.fits')
        return cls.from_fits(path, norm=norm)

    def plot(self, ax=None, **kwargs):
        """
        Plot the light curve.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
        kwargs: dict

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
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

        Returns
        -------
        X : `~numpy.ndarray`
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
        Sigma-clip the light curve (update mask).

        Parameters
        ----------
        sigma_upper : float
        sigma_lower : float
        maxiters : float or None
        plot : float
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
        """
        Regress the design matrix against the fluxes.

        Parameters
        ----------
        design_matrix : `~numpy.ndarray`

        Returns
        -------
        betas : `~numpy.ndarray`

        cov : `~numpy.ndarray`

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
        params : `~batman.TransitParams`
        t_fine : `~numpy.ndarray`
        transit_fine : `~numpy.ndarray`
        sinusoid_fine : `~numpy.ndarray`
        t0_offset : float
        n_regressors : int
        bins : int

        Returns
        -------
        fig, ax : `~matplotlib.pyplot.Figure`, `~matplotlib.pyplot.Axes`
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
