"""
classification - from the distances between recordings to the taxonomy
=======================================================================
    affinity   rank normalisation of a distance matrix and the affinities built from it
    snf        similarity network fusion of the scale views (CPU and GPU, same numbers)
    ap         affinity propagation (scikit-learn on the CPU; the same arithmetic on the GPU)
    fusion     the method of the paper: balanced rank fusion, one SNF across the scales, affinity
               propagation at every setting, and the plateau consensus
    kernels_gpu  the Triton kernels used by snf and ap

Author: Olivier Vertu Ndinga Oba
"""
