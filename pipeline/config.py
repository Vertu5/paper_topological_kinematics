"""
config.py - the settings of the pipeline
========================================
The data layout, the constants of the method (listed in Appendix B of the paper; each is checked by
analysis/parameter_study.py), the missions and the ranking of their controllers, and execution
settings that do not change any number.

Author: Olivier Vertu Ndinga Oba
"""
from pathlib import Path

ARGOS_ARENA = 2 * 1.231
ARGOS_DT = 0.1
DATA_DIR = Path('data/raw')
ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / 'results' / 'epuck'   # every result of the e-puck collection
OUTPUT_ROOT_DEFAULT = RESULTS_DIR

# ---- settings of the chain (the paper lists them in Appendix B; each one is checked in
#      analysis/parameter_study.py)
M_SCALES = 4               # number of scales of the ladder
EPS_SIGMA = 0.01           # floor of sigma_min (m), the tracking precision
SIGMA_MAX_FACTOR = 10.0    # sigma_max = (D_s / 2.35) / SIGMA_MAX_FACTOR: the cap with the largest plateau under the method (parameter study, plan final; analysis/cap_all.py)
LADDER_INSTANTS = None     # d_min and D_s over every instant of every run (None), or that many instants per run
WINDOW = 5.0               # evaluation window of the footprint, in sigma
N_CORESET = 300            # coreset size (farthest-point sampling of the agent positions), when COVER is None
COVER = 0.5                # at scale sigma, the coreset leaves no agent position farther than COVER * sigma (None: N_CORESET points)
N_CORESET_MAX = 20000      # largest coreset when COVER is set
FRAC_LOW_GRAD = 0.05       # fraction of flattest Delaunay triangles whose barycentres become seeds
N_JACOBI = 150             # largest number of seeds on the coupling skeleton J = 0
N_BIS = 12                 # bisection steps on J = 0
DEDUP_TOL = 1e-3           # seeds closer than DEDUP_TOL * sigma are merged
C_ALPHA = 3.0              # alpha complex truncated at squared radius (C_ALPHA * sigma)^2
N_DIRS = 25                # directions of the sliced Wasserstein distance

BEHAVIORS = ['directional_gate', 'foraging', 'homing',
             'shelter', 'xor_aggregation']

# Storage order of the runs only (see pipeline/io.py); Top-n is defined by pipeline.ranking.
TOP_CONTROLLERS = {
    'directional_gate': [7, 9, 0, 4, 3, 2, 5, 8, 6, 1],
    'foraging':         [9, 4, 1, 8, 5, 3, 0, 7, 2, 6],
    'homing':           [7, 3, 2, 9, 8, 6, 1, 0, 5, 4],
    'shelter':          [5, 0, 9, 8, 3, 2, 6, 1, 7, 4],
    'xor_aggregation':  [7, 1, 6, 4, 2, 8, 0, 9, 3, 5],
}

# ---- execution only (they do not change any number)
BATCH_SEEDS = 15           # runs per GPU call in stage 1
N_JOBS_SWD_CPU = 20        # CPU workers of stage 3 when no GPU is available
N_THREADS_SEEDS = 16       # threads computing the coresets of a batch
