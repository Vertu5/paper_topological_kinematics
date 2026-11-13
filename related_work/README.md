# related_work/

Reference papers kept locally (PDFs are git-ignored and not redistributed).

- Legarda Herranz, Garone, Birattari, *Persistent homology as a universal metric for
  collective motion*, Research Square preprint, 2026, doi:10.21203/rs.3.rs-8672197/v1.
  Same e-puck data (Top-1: best controller per mission, 250 runs, dodecagonal arenas of
  4.91 m^2, controllers from Hasselmann et al. 2021). Persistent homology of the
  Helmholtz-Hodge potentials of a velocity field; spectral clustering with the number
  of clusters fixed to five; per-mission accuracies 100 % (foraging), 96 % (directional
  gate), 98 % (shelter), 87 % (XOR aggregation), 80 % (homing).

Differences of this work: five rotation-invariant fields from the first three velocity
moments (density, kinetic intensity, both potentials, Jacobi coupling), grid-free
extended persistence at four data-driven scales, balanced rank fusion with cross-scale
SNF, and Affinity Propagation so that the number of behaviours and the exemplars are
outputs, evaluated from Top-1 to Top-10.
- Carrière, Cuturi, Oudot, *Sliced Wasserstein kernel for persistence diagrams*, Proceedings of
  the 34th International Conference on Machine Learning (ICML), PMLR 70, 664-673, 2017
  (`carriere2017_sliced_wasserstein_kernel.pdf`, from proceedings.mlr.press). The sliced
  Wasserstein distance between two diagrams is of order one: Eq. (2) and Definition 3.1 use the
  1-Wasserstein distance |x - y| between the projections, and Algorithm 1 averages the l1 norm of
  the sorted projections over M directions of the half-circle. Theorem 3.3:
  d1/(2M) <= SW <= 2 sqrt(2) d1, with d1 the first diagram distance and M = 1 + 2N(2N - 1).

Also kept locally (checked in the text before citing):
- Lindeberg, *Scale-space theory: A basic tool for analysing structures at different scales*, Journal of Applied
  Statistics 21(2):225-270, 1994 (lindeberg1994_scale_space_theory_review.pdf): a signal embedded in a one-parameter
  family obtained by convolution with Gaussian kernels of increasing width. Witkin (1983, IJCAI, pp. 1019-1022) was
  checked from its abstract (convolution with Gaussian masks over a continuum of sizes, all scales of observation).
- Cohen-Steiner, Edelsbrunner, Harer, *Extending persistence using Poincare and Lefschetz duality*, FoCM 9:79-103,
  2009, with its erratum (cohen2009_extending_persistence*.pdf): Section 6 proves that, on any simplicial complex,
  the bottleneck distance between the extended diagrams of two vertex functions is at most their sup-norm difference.
- Weinhart, Thornton, Luding, Bokhove, *From discrete particles to continuum fields near a boundary*, Granular Matter
  14(2):289-294, 2012 (weinhart2012_discrete_particles_continuum_fields.pdf): density and momentum density as sums of
  m_i W(r - r_i) and m_i v_i W(r - r_i) over the particles; a Gaussian W is taken for its smooth fields and exact
  integrals (Section 3.1). Our M_0 and M_1 are these fields for unit masses, up to the normalization of the kernel.

- Rosenberg, Hirschberg, *V-Measure: A conditional entropy-based external cluster evaluation measure*, EMNLP-CoNLL 2007,
  pp. 410-420 (rosenberg2007_vmeasure.pdf): homogeneity h = 1 - H(C|K)/H(C) and completeness c = 1 - H(K|C)/H(K), both 1 in
  the degenerate cases (Eqs. 1 and 2), as scikit-learn computes them. Hubert and Arabie (1985) is not freely available: the
  ARI properties stated in the paper (1 for identical partitions, 0 on average for random partitions with the same group
  sizes) were checked numerically on the Top-1 and Top-10 partitions (2000 random draws, mean -0.00000 +/- 0.00013).
