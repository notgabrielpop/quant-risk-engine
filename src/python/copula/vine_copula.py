"""Vine copula estimation and simulation using pyvinecopulib.

Fits a vine copula to uniform-transformed returns and provides
simulation and inspection utilities.
"""

import numpy as np
import pandas as pd
import pyvinecopulib as pv
from .copula_families import ALLOWED_FAMILIES, family_name, tail_dependence_bicop


def fit_vine_copula(u_data: np.ndarray, trunc_lvl: int = 5) -> pv.Vinecop:
    """Fit a vine copula to uniform data (n_obs x n_dims)."""
    controls = pv.FitControlsVinecop(
        family_set=ALLOWED_FAMILIES,
        selection_criterion="aic",
        trunc_lvl=trunc_lvl,
        tree_criterion="tau",
    )
    cop = pv.Vinecop.from_data(u_data, controls=controls)
    return cop


def simulate_vine(cop: pv.Vinecop, n: int, seed: int = 42) -> np.ndarray:
    """Simulate n samples from fitted vine copula. Returns (n, d) uniform."""
    return cop.simulate(n, seeds=[seed])


def vine_structure_table(cop: pv.Vinecop,
                         asset_names: list) -> pd.DataFrame:
    """Extract edge-level info from fitted vine: family, params, tail dep."""
    d = len(asset_names)
    struct = cop.structure
    rows = []

    for tree in range(min(d - 1, cop.trunc_lvl)):
        for edge in range(d - 1 - tree):
            pair_cop = cop.get_pair_copula(tree, edge)
            fam = pair_cop.family
            par = pair_cop.parameters
            rot = pair_cop.rotation
            lam_L, lam_U = tail_dependence_bicop(pair_cop)

            # Get variable indices from structure matrix
            struct_matrix = np.array(struct.matrix)
            var1 = int(struct_matrix[0, edge]) if tree == 0 else int(struct_matrix[tree, edge])
            var2 = int(struct_matrix[tree + 1, edge]) if tree + 1 < d else -1

            # Map to asset names (1-indexed in pyvinecopulib)
            name1 = asset_names[var1 - 1] if 0 < var1 <= d else f"V{var1}"
            name2 = asset_names[var2 - 1] if 0 < var2 <= d else f"V{var2}"

            par_str = ", ".join(f"{p:.4f}" for p in par.flatten())
            rows.append({
                'Tree': tree + 1,
                'Edge': edge + 1,
                'Var1': name1,
                'Var2': name2,
                'Family': family_name(fam),
                'Parameters': par_str,
                'Rotation': rot,
                'Lambda_L': lam_L,
                'Lambda_U': lam_U,
            })

    return pd.DataFrame(rows)


def get_pair_copula_matrix(cop: pv.Vinecop, d: int) -> dict:
    """Get a summary of all pair copulas in the vine.

    Returns dict with keys (tree, edge) -> {family, params, tail_dep}.
    """
    info = {}
    for tree in range(min(d - 1, cop.trunc_lvl)):
        for edge in range(d - 1 - tree):
            pc = cop.get_pair_copula(tree, edge)
            lam_L, lam_U = tail_dependence_bicop(pc)
            info[(tree, edge)] = {
                'family': family_name(pc.family),
                'params': pc.parameters.flatten().tolist(),
                'rotation': pc.rotation,
                'lambda_L': lam_L,
                'lambda_U': lam_U,
            }
    return info
