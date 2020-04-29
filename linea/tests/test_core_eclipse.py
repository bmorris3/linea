import numpy as np
import pytest

import astropy.units as u
from astropy.time import Time
from astropy.io.fits import fitsrec

from batman import TransitModel

from ..core import JointLightCurve, CheopsLightCurve
from ..planets import Planet
from .test_core_phase_curve import simulate_roll_angle

true_basis_vector_weights = [-2, 2, 1, 1, 20, 1, 1.5, 0.1, 0.1, 2e3]


def generate_recarrays_WASP189(depth_ppm=80, seed=42, n_outliers=50,
                               cheops_orbit_min=99.5, obs_efficiency=0.55,
                               n_visits=4):
    p = Planet.from_name("WASP-189 b")

    np.random.seed(seed)

    record_arrays = []

    for i in range(n_visits):

        all_times = np.linspace(p.t0 + p.per * (0.5 + i) - 0.1 * p.per,
                                p.t0 + p.per * (0.5 + i) + 0.1 * p.per, 2500)
        cheops_phase = (((all_times[0] - all_times) * u.day) %
                        (cheops_orbit_min * u.min) /
                        (cheops_orbit_min * u.min)
                        ).to(u.dimensionless_unscaled).value

        observable = cheops_phase > (1 - obs_efficiency)
        bjd_time = all_times[observable]
        utc_time = Time(bjd_time, format='jd').isot
        mjd_time = Time(bjd_time, format='jd').mjd

        n_points = len(bjd_time)

        conta_lc = 10 * np.ones(n_points) + np.random.randn(n_points)
        conta_lc_err = np.ones(n_points) / 100
        status = np.zeros(n_points)
        event = np.zeros(n_points)
        dark = 10 + np.random.randn(n_points)
        background = 100 + np.random.randn(n_points)
        roll_angle = simulate_roll_angle(cheops_phase[observable],
                                         bjd_time, phase_offset=0.1)
        location_x = 512 * np.ones(n_points)
        location_y = 512 * np.ones(n_points)
        centroid_x = location_x[0] + 0.2 * np.random.randn(n_points)
        centroid_y = location_y[0] + 0.2 * np.random.randn(n_points)

        p.t_secondary = 0.5
        p.fp = depth_ppm * 1e-6
        model = TransitModel(p, bjd_time,
                             supersample_factor=3,
                             transittype='secondary',
                             exp_time=bjd_time[1] - bjd_time[0]
                             ).light_curve(p)

        basis_vectors = ((bjd_time - bjd_time.mean()), background, conta_lc,
                         dark, roll_angle/360, (centroid_x - location_x)**2,
                         (centroid_y - location_y)**2,
                         (centroid_x - location_x), (centroid_y - location_y),
                         dark.mean())

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
        record_arrays.append(ra)

    return [fitsrec.FITS_rec(ra) for ra in record_arrays]


@pytest.mark.parametrize('eclipse_depth,', [20, 40, 60, 80, 100])
def test_eclipse(eclipse_depth):
    """
    from linea.tests.test_core import test_eclipse as f; f()
    """
    p = Planet.from_name("WASP-189 b")

    ras = generate_recarrays_WASP189(depth_ppm=eclipse_depth)
    lcs = JointLightCurve([CheopsLightCurve(ra) for ra in ras])

    # check that normalization brings flux continuum to unity:
    assert all([abs(np.median(lc.flux) - 1) < 1e-5 for lc in lcs])

    for lc in lcs:
        lc.sigma_clip_flux()

    # Check that some points get masked by the flux sigma clipping
    assert all([np.count_nonzero(lc.mask) > 0 for lc in lcs])

    all_lcs = lcs.concatenate()
    eclipse_model = TransitModel(p, all_lcs.bjd_time[~all_lcs.mask],
                                 supersample_factor=3,
                                 transittype='secondary',
                                 exp_time=lcs[0].bjd_time[1]-lcs[0].bjd_time[0],
                                 ).light_curve(p) - 1

    X_combined = lcs.combined_design_matrix()

    # Build a design matrix
    X = np.hstack([
        # Default combined design matrix:
        X_combined,

        # Eclipse model
        eclipse_model[:, None]
    ])

    r = lcs.regress(X)

    obs_eclipse_depth = r.betas[-1]
    eclipse_error_ppm = np.sqrt(np.diag(r.cov))[-1]

    # Check that eclipse depth is within 2 sigma of the true answer
    agreement_sigma = (abs(obs_eclipse_depth - eclipse_depth) /
                       eclipse_error_ppm)

    assert agreement_sigma < 2
