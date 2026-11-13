"""
gaussian_splatting - from tracked agents to the topology of their motion
========================================================================
The method of the paper, up to the distances between recordings:

    fields_cpu, fields_gpu   the five fields (density, kinetic intensity, the two potentials of the
                             Helmholtz-Hodge decomposition and the Jacobi coupling) in closed form at
                             any points, from Gaussian footprints of the velocity moments
    scale_ladder             the scales sigma_1..sigma_M, from the data
    seeds_gridfree           the seeds of a recording at a scale (coreset, flat triangles, J = 0)
    alpha                    the alpha complex of the seeds
    persistence              the six extended-persistence diagrams of a field
    distance, swd_gpu        the sliced-Wasserstein distance (CPU, and GPU with the same numbers)

The stages that run these over all recordings are in pipeline/ (run_pipeline.py).

Author: Olivier Vertu Ndinga Oba
"""

from .fields_cpu import AVAILABLE_FIELDS, FieldComputer
from .persistence import N_SUB_DIAGRAMS, SUB_DIAGRAM_NAMES, PersistenceComputer
from .alpha import AlphaSimplexTree
from .distance import sample_directions, swd
from .scale_ladder import compute_scale_ladder

__all__ = ["AVAILABLE_FIELDS", "FieldComputer", "N_SUB_DIAGRAMS", "SUB_DIAGRAM_NAMES", "PersistenceComputer",
           "AlphaSimplexTree", "sample_directions", "swd", "compute_scale_ladder"]
