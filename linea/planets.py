import numpy as np

__all__ = ['Planet', 'params_55Cnce', 'params_WASP189']


class Planet(object):
    """
    Transiting planet parameters.

    This is meant to be a duck-type drop-in for the ``batman`` package's
    transiting exoplanet parameters ``TransitParams`` object.
    """
    def __init__(self, per=None, t0=None, inc=None, rp=None, ecc=None, w=None,
                 a=None, u=None, fp=None, t_secondary=None,
                 limb_dark='quadratic'):
        self.per = per
        self.t0 = t0
        self.inc = inc
        self.rp = rp
        self.ecc = ecc
        self.w = w
        self.a = a
        self.u = u
        self.limb_dark = limb_dark
        self.fp = fp
        self.t_secondary = t_secondary


def params_55Cnce():
    """
    Planet parameters for 55 Cnc e
    """
    params_e = Planet()
    params_e.per = 0.736539
    params_e.t0 = 2455733.013
    params_e.inc = 83.3
    params_e.rp = 0.0187

    # a/rs = b/cosi
    b = 0.41

    eccentricity = 0  # np.sqrt(ecosw**2 + esinw**2)
    omega = 90  # np.degrees(np.arctan2(esinw, ecosw))

    ecc_factor = (np.sqrt(1 - eccentricity**2) /
                  (1 + eccentricity * np.sin(np.radians(omega))))

    params_e.a = b / np.cos(np.radians(params_e.inc)) / ecc_factor
    params_e.ecc = eccentricity
    params_e.w = omega
    params_e.u = [0.4, 0.2]
    params_e.limb_dark = 'quadratic'
    return params_e


def params_WASP189():
    """
    Planet parameters for WASP-189 b
    """
    params_w189 = Planet()
    params_w189.per = 2.7240338
    params_w189.t0 = 2456706.4558
    params_w189.inc = 83.4
    params_w189.rp = 0.00372**0.5

    # a/rs = b/cosi
    # b = 0.51

    eccentricity = 0
    omega = 90

    params_w189.t_secondary = 0.5
    params_w189.fp = 1e-6

    params_w189.a = 4.4
    params_w189.ecc = eccentricity
    params_w189.w = omega
    params_w189.u = [0, 0]
    params_w189.limb_dark = 'quadratic'
    return params_w189
