"""
alpha.py - the domain of a recording: the alpha complex of its seeds
=====================================================================
The seeds (seeds_gridfree.py) are triangulated by GUDHI's alpha complex, truncated at squared radius
(3 sigma)^2; the filtration values are then those of a field at the vertices (lower-star), and
extended persistence is computed on that fixed complex.

Author: Olivier Vertu Ndinga Oba
"""

from __future__ import annotations
import numpy as np

try:
    import gudhi
except ImportError:
    gudhi = None


class AlphaSimplexTree:
    """
    SimplexTree built from an adaptive Alpha complex for EPH.
    
    Unlike StructuredSimplexTree which builds from a regular grid
    triangulation, this builds from an Alpha complex over arbitrary
    seed points. The filtration is set by field values, not alpha radii.
    
    Parameters
    ----------
    points : ndarray of shape (V, 2)
        Seed point coordinates.
    alpha_sq_max : float
        Maximum squared alpha radius. Simplices with alpha > sqrt(alpha_sq_max)
        are excluded. Set to (c_alpha * sigma)**2.
    """
    
    def __init__(self, points: np.ndarray, alpha_sq_max: float = float('inf')):
        if gudhi is None:
            raise ImportError("GUDHI is required for AlphaSimplexTree.")
        
        self.points = np.asarray(points, dtype=np.float64)
        self.n_vertices = len(self.points)
        self.is_empty = self.n_vertices < 3
        
        if self.is_empty:
            self.active_verts = []
            self._template = None
            self._active_verts_np = np.array([], dtype=np.int64)
            return
        
        # Build Alpha complex
        alpha_complex = gudhi.AlphaComplex(points=self.points)
        st = alpha_complex.create_simplex_tree(max_alpha_square=alpha_sq_max)
        
        # every simplex, listed once before any filtration value is changed
        simplices = [s for s, _ in st.get_simplices()]
        self._active_verts_np = np.array(sorted({v for s in simplices for v in s}), dtype=np.int64)
        self.active_verts = self._active_verts_np.tolist()
        # every simplex starts at -1e20; the field values of the vertices then give each simplex the
        # largest value of its vertices (make_filtration_non_decreasing): a pure lower-star filtration
        for s in simplices:
            st.assign_filtration(s, -1e20)
        
        self._template = st
    
    def compute_raw_persistence(self, field_values: np.ndarray) -> list:
        """
        Compute extended persistence for one scalar field.
        
        Parameters
        ----------
        field_values : ndarray of shape (V,)
            Field values at each seed point (vertex).
        
        Returns
        -------
        Raw output from SimplexTree.extended_persistence().
        """
        if self.is_empty:
            return [[], [], [], []]
        
        active_vals = field_values[self._active_verts_np]
        if not np.all(np.isfinite(active_vals)):
            n_bad = int((~np.isfinite(active_vals)).sum())
            raise ValueError(
                f"{n_bad} active vertices have non-finite values."
            )
        
        st = gudhi.SimplexTree(self._template)
        for v in self.active_verts:
            st.assign_filtration([v], float(field_values[v]))
        st.make_filtration_non_decreasing()
        st.extend_filtration()
        return st.extended_persistence()
    
    def __repr__(self) -> str:
        if self.is_empty:
            return "AlphaSimplexTree(empty)"
        n_simplices = sum(1 for _ in self._template.get_filtration())
        return (f"AlphaSimplexTree(n_vertices={self.n_vertices}, "
                f"n_active={len(self.active_verts)}, "
                f"n_simplices={n_simplices})")
