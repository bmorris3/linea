import os
from json import load

__all__ = ['Planet']

planets_path = os.path.join(os.path.dirname(__file__), 'data', 'planets.json')


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

    @classmethod
    def from_name(cls, name):
        """
        Initialize a Planet instance from the target name.

        There's a small (but growing?) database of planets pre-defined in the
        ``linea/data/planets.json`` file. If your favorite planet is missing,
        pull requests are welcome!

        Parameters
        ----------
        name : str (i.e.: "55 Cnc e" or "WASP-189 b")
             Name of the planet
        """
        with open(planets_path, 'r') as f:
            planets = load(f)

        return cls(**planets[name])

