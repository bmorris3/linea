import numpy as np
import pytest

import astropy.units as u
from astropy.time import Time
from astropy.io.fits import fitsrec

from batman import TransitModel

from ..core import CheopsLightCurve
from ..planets import Planet


true_basis_vector_weights = [-2, 2, 1.1, 1, 25, 2, 3, 0.5, 0.1, 1e3]


def simulate_roll_angle(phase, time, phase_offset=1.5):
    # Compute rough approximation to a varying parallactic angle
    H = 2 * np.pi * (phase - phase_offset)
    q = np.arctan2(np.sin(H) + 0.3 * (time - time[0]),
                   (np.tan(np.radians(0)) *
                    np.cos(np.radians(20)) -
                    np.sin(np.radians(20))*np.cos(H)))
    return np.degrees(q)


def generate_recarray_55Cnce(seed=42, sinusoid_amp_depth_frac=0.25,
                             n_outliers=50, cheops_orbit_min=99.5,
                             obs_efficiency=0.5):
    p = Planet.from_name("55 Cnc e")

    np.random.seed(seed)

    all_times = np.linspace(p.t0 - 0.5, p.t0 + 1., 2500)
    cheops_phase = (((all_times[0] - all_times) * u.day) %
                    (cheops_orbit_min * u.min) /
                    (cheops_orbit_min * u.min)
                    ).to(u.dimensionless_unscaled).value

    bjd_time = all_times[cheops_phase > (1 - obs_efficiency)]
    utc_time = Time(bjd_time, format='jd').isot
    mjd_time = Time(bjd_time, format='jd').mjd

    n_points = len(bjd_time)

    conta_lc = 10 * np.ones(n_points) + np.random.randn(n_points)
    conta_lc_err = np.ones(n_points) / 100
    status = np.zeros(n_points)
    event = np.zeros(n_points)
    dark = 10 + np.random.randn(n_points)
    background = 100 + np.random.randn(n_points)
    roll_angle = simulate_roll_angle(cheops_phase[cheops_phase > 0.5], bjd_time)
    location_x = 512 * np.ones(n_points)
    location_y = 512 * np.ones(n_points)
    centroid_x = location_x[0] + 0.2 * np.random.randn(n_points)
    centroid_y = location_y[0] + 0.2 * np.random.randn(n_points)

    model = TransitModel(p, bjd_time,
                         supersample_factor=3,
                         exp_time=bjd_time[1] - bjd_time[0]
                         ).light_curve(p)
    planet_phase = ((bjd_time - p.t0) % p.per) / p.per

    sinusoid_amp = sinusoid_amp_depth_frac * p.rp**2

    model += sinusoid_amp * np.sin(2 * np.pi * (planet_phase + 0.5))

    basis_vectors = ((bjd_time - bjd_time.mean()), background, conta_lc, dark,
                     roll_angle/360, (centroid_x - location_x)**2,
                     (centroid_y - location_y)**2, (centroid_x - location_x),
                     (centroid_y - location_y), dark.mean())

    flux = np.zeros_like(bjd_time)

    for c, v in zip(true_basis_vector_weights, basis_vectors):
        flux += c * v

    flux *= model

    fluxerr = np.std(flux) * np.ones(len(bjd_time))

    rand_indices = np.random.randint(0, flux.shape[0], size=n_outliers)
    flux[rand_indices] += 3.5 * flux.std() * np.random.randn(n_outliers)

    formatter = [
        ('UTC_TIME', '|S26', utc_time),
        ('MJD_TIME', '>f8', mjd_time),
        ('BJD_TIME', '>f8', bjd_time),
        ('FLUX', '>f8', flux),
        ('FLUXERR', '>f8', fluxerr),
        ('STATUS', '>i4', status),
        ('EVENT', '>i4', event),
        ('DARK', '>f8', dark),
        ('BACKGROUND', '>f8', background),
        ('CONTA_LC', '>f8', conta_lc),
        ('CONTA_LC_ERR', '>f8', conta_lc_err),
        ('ROLL_ANGLE', '>f8', roll_angle),
        ('LOCATION_X', '>f4', location_x),
        ('LOCATION_Y', '>f4', location_y),
        ('CENTROID_X', '>f4', centroid_x),
        ('CENTROID_Y', '>f4', centroid_y)
    ]

    ra = np.recarray((n_points,),
                     names=[name for name, _, _ in formatter],
                     formats=[fmt for _, fmt, _ in formatter])

    for name, fmt, arr in formatter:
        ra[name] = arr

    return fitsrec.FITS_rec(ra)


@pytest.mark.parametrize('depth_frac,', [0.1, 0.25, 0.5, 1, 1.5, 2])
def test_phase_curve(depth_frac):
    """
    from linea.tests.test_core import test_phase_curve as f; f()
    """
    p = Planet.from_name("55 Cnc e")

    ra = generate_recarray_55Cnce(sinusoid_amp_depth_frac=depth_frac)
    lc = CheopsLightCurve(ra, norm=True)

    # check that normalization brings flux continuum to unity:
    assert abs(np.median(lc.flux) - 1) < 1e-5

    lc.sigma_clip_flux()

    # Check that some points get masked by the flux sigma clipping
    assert np.count_nonzero(lc.mask) > 0

    transit_model = TransitModel(p, lc.bjd_time[~lc.mask],
                                 supersample_factor=3,
                                 exp_time=lc.bjd_time[1] - lc.bjd_time[0],
                                 ).light_curve(p) - 1

    t = lc.bjd_time[~lc.mask, None] - lc.bjd_time.mean()

    # Build a design matrix
    X = np.hstack([
        # Transit model:
        transit_model[:, None],

        # Sinusoidal phase curve trend:
        np.sin(2 * np.pi * t / p.per),
        np.cos(2 * np.pi * t / p.per),

        # Default design matrix:
        lc.design_matrix(),
    ])

    r = lc.regress(X)

    sinusoid = X[:, 1:3] @ r.betas[1:3]

    phase_curve_amp_ppm = 1e6 * sinusoid.ptp()
    phase_curve_amp_error_ppm = 1e6 * np.max(np.sqrt(np.diag(r.cov))[1:3])

    true_amp_ppm = 2e6 * p.rp**2 * depth_frac

    # Check that phase curve amplitude is within 2 sigma of the true answer
    agreement_sigma = (abs(phase_curve_amp_ppm - true_amp_ppm) /
                       phase_curve_amp_error_ppm)
    assert agreement_sigma < 2
