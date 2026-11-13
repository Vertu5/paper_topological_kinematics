# Lean 4 proofs of the propositions of the paper

`TopologicalKinematics/Basic.lean` proves, with Mathlib and without any unproved step:

* Proposition 1 (closed-form Helmholtz-Hodge potentials): for one agent, at every offset `d ≠ 0`,
  `∇(u κσ(d)) - ∇⊥(w κσ(d)) = v e^{-|d|²/2σ²}` (`hhd_single_agent`); for a recording, with any constant
  factor, `∇φ + ∇⊥ψ = c Σᵢ Σₜ vᵢ(t) e^{-|r - xᵢ(t)|²/2σ²}` at every point that no agent occupies
  (`hhd_recording`). At an occupied point the identity follows by continuity, both sides being smooth.
* Proposition 2 (invariance): the elementary terms `|d|²`, `u`, `w`, the kernel `κσ`, the two potentials and
  the Jacobi determinant under rotations; under a mirror image `φ` is unchanged, `ψ` changes sign and the
  Jacobi determinant is unchanged.
* Proposition 3 (tail of the footprint): `∫_ρ^∞ 2π s e^{-s²/2σ²} ds = 2πσ² e^{-ρ²/2σ²}` (`gaussian_tail`).

Build and check (the first build downloads the compiled Mathlib, about 5 GB):

```bash
cd manuscript/lean
lake exe cache get
lake build
lake env lean CheckAxioms.lean   # each theorem: [propext, Classical.choice, Quot.sound]
```

Author: Olivier Vertu Ndinga Oba
