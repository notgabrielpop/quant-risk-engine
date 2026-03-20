"""Bivariate copula family utilities.

Wraps pyvinecopulib for copula fitting and tail dependence computation.
Provides helper functions for working with bivariate copula families.
"""

import numpy as np
import pyvinecopulib as pv

# Map family enum to human-readable names
FAMILY_NAMES = {
    pv.BicopFamily.gaussian: "Gaussian",
    pv.BicopFamily.student: "Student-t",
    pv.BicopFamily.clayton: "Clayton",
    pv.BicopFamily.gumbel: "Gumbel",
    pv.BicopFamily.frank: "Frank",
    pv.BicopFamily.joe: "Joe",
    pv.BicopFamily.bb1: "BB1",
    pv.BicopFamily.bb6: "BB6",
    pv.BicopFamily.bb7: "BB7",
    pv.BicopFamily.bb8: "BB8",
    pv.BicopFamily.indep: "Independence",
}

# Families we allow during vine fitting
ALLOWED_FAMILIES = [
    pv.BicopFamily.student,
    pv.BicopFamily.clayton,
    pv.BicopFamily.gumbel,
    pv.BicopFamily.frank,
    pv.BicopFamily.gaussian,
]


def family_name(fam) -> str:
    return FAMILY_NAMES.get(fam, str(fam))


def tail_dependence_bicop(bicop: pv.Bicop) -> tuple:
    """Compute lower and upper tail dependence for a fitted bivariate copula.

    Returns (lambda_L, lambda_U).
    """
    fam = bicop.family
    par = bicop.parameters
    rot = bicop.rotation

    lam_L, lam_U = 0.0, 0.0

    if fam == pv.BicopFamily.student:
        rho = par[0, 0]
        df = par[1, 0] if par.shape[0] > 1 else par[0, 1]
        from scipy import stats as sp_stats
        arg = np.sqrt((df + 1) * (1 - rho) / (1 + rho))
        lam = 2 * sp_stats.t.cdf(-arg, df + 1)
        lam_L = lam_U = lam

    elif fam == pv.BicopFamily.clayton:
        theta = par[0, 0]
        if theta > 0:
            base_L = 2 ** (-1.0 / theta)
        else:
            base_L = 0.0
        base_U = 0.0
        lam_L, lam_U = _apply_rotation(base_L, base_U, rot)

    elif fam == pv.BicopFamily.gumbel:
        theta = par[0, 0]
        base_L = 0.0
        base_U = 2 - 2 ** (1.0 / theta) if theta >= 1 else 0.0
        lam_L, lam_U = _apply_rotation(base_L, base_U, rot)

    elif fam == pv.BicopFamily.joe:
        theta = par[0, 0]
        base_L = 0.0
        base_U = 2 - 2 ** (1.0 / theta) if theta >= 1 else 0.0
        lam_L, lam_U = _apply_rotation(base_L, base_U, rot)

    # Gaussian and Frank have no tail dependence
    return lam_L, lam_U


def _apply_rotation(lam_L: float, lam_U: float, rot: int) -> tuple:
    """Adjust tail dependence for copula rotation."""
    if rot == 0:
        return lam_L, lam_U
    elif rot == 180:
        return lam_U, lam_L
    elif rot == 90:
        return 0.0, 0.0
    elif rot == 270:
        return 0.0, 0.0
    return lam_L, lam_U


def fit_bivariate(u1: np.ndarray, u2: np.ndarray,
                  families=None) -> pv.Bicop:
    """Fit a bivariate copula selecting best family by AIC."""
    if families is None:
        families = ALLOWED_FAMILIES
    data = np.column_stack([u1, u2])
    controls = pv.FitControlsBicop(
        family_set=families,
        selection_criterion="aic"
    )
    cop = pv.Bicop.from_data(data, controls=controls)
    return cop
