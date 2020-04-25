Background
==========

The foundation of ``linea`` is linear detrending via least squares, so some
background reading on least squares from
`wikipedia <https://en.wikipedia.org/wiki/Least_squares>`_ might be helpful
before you browse this tutorial. We'll use some of the same symbols as the
wikipedia entry for consistency.

Linear detrending
-----------------

The default raw aperture photometry from CHEOPS, :math:`f_i`, contains several
systematic trends as a function of time :math:`t`, roll angle :math:`\theta`,
stellar centroid position (:math:`x, y`), contamination :math:`c`$`
(primarily from 53 Cnc), and varying background flux :math:`d`. To first order,
the flux that we observe is a linear combination of the astrophysical signals we
wish to detect and some unknown function of these observational
`basis vectors <https://en.wikipedia.org/wiki/Basis_(linear_algebra)>`_.
For some CHEOPS programs, the astrophysical signal has unconstrained quantities,
like transit times or depths. For this particular analysis, the orbit of 55 Cnc
e is known with great precision, allowing us to fix the planet's parameters and
fit only for the phase and amplitude of the sinusoidal phase curve signal.
In principle, there is some
`design matrix <https://en.wikipedia.org/wiki/Design_matrix>`_
:math:`\bf X` for which we can solve for the least-squares estimators
:math:`\hat{\beta}` such that

.. math::

    {\bf X} \hat{\beta} = f.

In practice, we do not know the precise functional form of the linear
combination of basis vectors which produces the flux variations we observe, so
we assemble a design matrix :math:`\bf X` from the observational vectors
mentioned earlier, of the form

.. math::

    \begin{split}
    {\bf X_{\rm sys}} &= \\
      & \begin{bmatrix}
        t_1 & x_1 & y_1 & x_1y_1 & x_1^2 & y_1^2 & \theta_1 & c_1 & d_1 & 1 \\
        t_2 & x_2 & y_2 & x_2y_2 & x_2^2 & y_2^2 & \theta_2 & c_2 & d_2 & 1\\
        \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
        t_N & x_N & y_N & x_Ny_N & x_N^2 & y_N^2 & \theta_N & c_N & d_N & 1
      \end{bmatrix}
    \end{split}

for all :math:`N` observations, after :math:`3\sigma`-clipping the raw fluxes.
We call the matrix :math:`\bf X_{\rm sys}` the *systematics matrix*, which we
assemble from the FITS record array associated with each CHEOPS observation.

The systematics matrix does not include any astrophysical signal associated with
this observation, so in order to properly model the flux variations due to the
transit event and the phase curve, we introduce three more column vectors to
concatenate with :math:`\bf X_{\rm sys}`, which we call
:math:`\bf X_{\rm model}`,

.. math::

    \bf X_{\rm model} =
      \begin{bmatrix}
        T_1 & \sin(2\pi t_1 / P) & \cos(2\pi t_1 / P) \\
        T_2 &\sin(2\pi t_2 / P) & \cos(2\pi t_2 / P)\\
        \vdots & \vdots & \vdots\\
        T_N & \sin(2\pi t_N / P) & \cos(2\pi t_N / P)
      \end{bmatrix}

where :math:`T_i` is the Mandel & Agol (2002) transit model minus one. We solve
for the least-squares estimators for the two sinusoidal column vectors to
account for the unknown phase of the sinusoidal phase curve signal.
The complete design matrix is

.. math::

    \bf X =
      \begin{bmatrix}
        {\bf X}_{\rm sys} & {\bf X}_{\rm model}
      \end{bmatrix}

We also define the uncertainty matrix :math:`\bf N`,

.. math::

    {\bf N} = {\bf I_N} \sigma_f^2,

where :math:`\bf I_N` is the
`identity matrix <https://en.wikipedia.org/wiki/Identity_matrix>`_.

Now we solve for the least-squares estimators :math:`\hat{\beta}`,

.. math::

    \hat{\beta} = ({\bf X}^{\rm T} {\bf N}^{-1} {\bf X})^{-1} {\bf X}^{\rm T} {\bf N}^{-1} f.

Uncertainties on each of the least-squares estimators are computed with the
pre-computed matrix inversion,

.. math::
    \sigma_{\hat{\beta}}^2 = ({\bf X}^{\rm T} {\bf N}^{-1} {\bf X})^{-1}.
