"""
persistence.py - the six extended-persistence diagrams of a field
==================================================================
On a fixed complex (alpha.py), a field f at the vertices defines a lower-star filtration. Extended
persistence follows the ascending sweep of the sublevel sets by a descending sweep of the superlevel
sets relative to the whole complex, so that every class has a finite birth and death:

    Ord_0, Ord_1  classes of dimension 0 and 1 born and killed during the ascending sweep
    Rel_1, Rel_2  classes of dimension 1 and 2 born and killed during the descending sweep
    Ext_0, Ext_1  classes of dimension 0 and 1 born in the first sweep and killed in the second

Every pair returned by GUDHI lands in one of the six diagrams; none is dropped.

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations
from typing import List
import numpy as np


SUB_DIAGRAM_NAMES: tuple = ("Ord0", "Ord1", "Rel1", "Rel2", "Ext0", "Ext1")
N_SUB_DIAGRAMS: int = len(SUB_DIAGRAM_NAMES)


def _safe_diagram(dg) -> np.ndarray:
    """Coerce diagram to ``ndarray (n, 2)``; empty -> ``(0, 2)``."""
    arr = np.asarray(dg, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    return np.zeros((0, 2), dtype=np.float64)


def _extract_sub_diagrams(raw) -> List[np.ndarray]:
    """
    Convert the raw output of ``SimplexTree.extended_persistence()``
    into a list of 6 numpy arrays.

    The raw output has 4 group lists ``[ord, rel, ext+, ext-]``; each
    entry is ``(dim, (birth, death))``. For the "rel" group we swap
    the (birth, death) pair so that birth <= death by convention.
    """
    def extract(group_idx: int, dim: int, swap: bool = False) -> np.ndarray:
        if group_idx >= len(raw):
            return np.zeros((0, 2), dtype=np.float64)
        pts = np.array(
            [p[1] for p in raw[group_idx] if p[0] == dim],
            dtype=np.float64,
        )
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float64)
        return pts[:, ::-1] if swap else pts

    mapped = {(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (3, 0), (2, 1), (3, 1)}
    for g, group in enumerate(raw):
        for dim, _ in group:
            if (g, dim) not in mapped:
                raise ValueError(f"persistence pair of group {g} and dimension {dim} has no diagram")

    def stack(*parts):
        parts = [p for p in parts if len(p)]
        return np.vstack(parts) if parts else np.zeros((0, 2), dtype=np.float64)

    # every pair returned by GUDHI is kept: on a planar complex the ordinary pairs have dimension
    # 0 or 1, the relative pairs dimension 1 or 2, and the extended pairs of either sign go to Ext_0
    # or Ext_1 according to their dimension (written with birth <= death)
    return [
        extract(0, 0),                                   # Ord_0
        extract(0, 1),                                   # Ord_1
        extract(1, 1, swap=True),                        # Rel_1
        extract(1, 2, swap=True),                        # Rel_2
        stack(extract(2, 0), extract(3, 0, swap=True)),  # Ext_0
        stack(extract(3, 1, swap=True), extract(2, 1)),  # Ext_1
    ]


# ==================================================================
class PersistenceComputer:
    """The six extended-persistence diagrams of one or several fields on a fixed complex
    (an AlphaSimplexTree): the complex is built once, only the filtration values change.

        pc = PersistenceComputer(AlphaSimplexTree(seeds, alpha_sq_max=(3 * sigma) ** 2))
        dgms = pc.compute_all(fields)          # dgms["psi"][5] is Ext_1 of psi
    """

    def __init__(self, structure):
        self.structure = structure

    # ----------------------------------------------------------------
    def compute_field(self, field_array: np.ndarray) -> List[np.ndarray]:
        """
        Compute the 6 sub-diagrams for one field array (ny, nx) or (G,).

        Returns
        -------
        list of 6 ndarrays of shape (n_points_i, 2).
        """
        if self.structure.is_empty:
            return [np.zeros((0, 2), dtype=np.float64)
                    for _ in range(N_SUB_DIAGRAMS)]

        flat = np.asarray(field_array, dtype=np.float64).ravel()
        raw = self.structure.compute_raw_persistence(flat)
        return [_safe_diagram(d) for d in _extract_sub_diagrams(raw)]

    # ----------------------------------------------------------------
    def compute_all(self, field_dict: dict) -> dict:
        """
        Compute the 6 sub-diagrams for every field in ``field_dict``.

        The structure is built once; only field values change between
        fields, so the C++ template clone is fast.
        """
        return {k: self.compute_field(v) for k, v in field_dict.items()}


