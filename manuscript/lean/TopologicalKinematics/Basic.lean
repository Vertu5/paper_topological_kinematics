import Mathlib

/-!
# Recovering a Taxonomy of Collective Behaviors from Motion Alone: the mathematical core in Lean 4

Paper: O. V. Ndinga Oba, "Recovering a Taxonomy of Collective Behaviors from Motion Alone", 2026.

Formalised and checked by Lean with Mathlib, without `sorry`:
* Proposition 1 (closed-form Helmholtz-Hodge potentials): for one agent, at every `d ≠ 0`,
  `∇(u κσ(d)) - ∇⊥(w κσ(d)) = v e^{-|d|²/2σ²}`; summed over agents and instants with any constant
  factor `c`, the potentials satisfy `∇φ + ∇⊥ψ = c Σ v_i e^{-|r - x_i|²/2σ²}` at every point that no
  agent occupies (at an occupied point the identity follows by continuity).
* Proposition 2 (invariance): the elementary terms `|d|²`, `u = v·d`, `w = det[v, d]`, the kernel
  κσ, the two potentials and the Jacobi determinant are unchanged by a rotation; under a mirror
  image `u` and `φ` are unchanged, `w` and `ψ` change sign, and the Jacobi determinant is unchanged.
* Proposition 3 (tail of the footprint): `∫_ρ^∞ 2π s e^{-s²/2σ²} ds = 2πσ² e^{-ρ²/2σ²}`, and the
  exponent at `ρ = 5σ` is `-25/2`.

Not formalised (algorithmic or empirical): extended persistence, the sliced-Wasserstein distance,
similarity network fusion, affinity propagation, and the experiments.

Author: Olivier Vertu Ndinga Oba
-/

noncomputable section

open Real MeasureTheory Filter Set
open scoped Topology

namespace TopologicalKinematics

/-! ## The plane -/

/-- A point (or a vector) of the plane. -/
abbrev R2 := ℝ × ℝ

/-- Squared Euclidean norm. -/
def normSq (p : R2) : ℝ := p.1 ^ 2 + p.2 ^ 2

/-- Dot product `u = v · d`. -/
def dot (p q : R2) : ℝ := p.1 * q.1 + p.2 * q.2

/-- Determinant `det[p, q] = p₁ q₂ - p₂ q₁` (oriented area); `w = det[v, d]`. -/
def det2 (p q : R2) : ℝ := p.1 * q.2 - p.2 * q.1

/-- Rotation of the plane by the angle `θ`. -/
def rot (θ : ℝ) (p : R2) : R2 := (cos θ * p.1 - sin θ * p.2, sin θ * p.1 + cos θ * p.2)

/-- Mirror image in the first axis. -/
def mirror (p : R2) : R2 := (p.1, -p.2)

/-! ## The footprint, the kernel κσ and the potentials of one agent -/

/-- The Gaussian footprint of width `σ` at the offset `d = r - x`. -/
def lens (σ : ℝ) (d : R2) : ℝ := exp (-normSq d / (2 * σ ^ 2))

/-- The kernel `κσ(d) = σ²/|d|² (1 - e^{-|d|²/2σ²})`, extended by its limit `1/2` at `d = 0`. -/
def kappa (σ : ℝ) (d : R2) : ℝ :=
  if normSq d = 0 then 1 / 2 else σ ^ 2 / normSq d * (1 - exp (-normSq d / (2 * σ ^ 2)))

/-- The divergence potential of one agent of velocity `v` at the offset `d`: `u κσ(d)`. -/
def Phi (σ : ℝ) (v d : R2) : ℝ := dot v d * kappa σ d

/-- The vorticity potential of one agent of velocity `v` at the offset `d`: `-w κσ(d)`. -/
def Psi (σ : ℝ) (v d : R2) : ℝ := -det2 v d * kappa σ d

/-! ## Proposition 2: rotations and mirror images -/

theorem normSq_rot (θ : ℝ) (p : R2) : normSq (rot θ p) = normSq p := by
  unfold normSq rot
  linear_combination (p.1 ^ 2 + p.2 ^ 2) * sin_sq_add_cos_sq θ

theorem dot_rot (θ : ℝ) (p q : R2) : dot (rot θ p) (rot θ q) = dot p q := by
  unfold dot rot
  linear_combination (p.1 * q.1 + p.2 * q.2) * sin_sq_add_cos_sq θ

theorem det2_rot (θ : ℝ) (p q : R2) : det2 (rot θ p) (rot θ q) = det2 p q := by
  unfold det2 rot
  linear_combination (p.1 * q.2 - p.2 * q.1) * sin_sq_add_cos_sq θ

theorem lens_rot (σ θ : ℝ) (d : R2) : lens σ (rot θ d) = lens σ d := by
  simp only [lens, normSq_rot]

theorem kappa_rot (σ θ : ℝ) (d : R2) : kappa σ (rot θ d) = kappa σ d := by
  simp only [kappa, normSq_rot]

/-- The divergence potential is unchanged when the velocity and the offset turn together. -/
theorem Phi_rot (σ θ : ℝ) (v d : R2) : Phi σ (rot θ v) (rot θ d) = Phi σ v d := by
  simp only [Phi, dot_rot, kappa_rot]

/-- The vorticity potential is unchanged when the velocity and the offset turn together. -/
theorem Psi_rot (σ θ : ℝ) (v d : R2) : Psi σ (rot θ v) (rot θ d) = Psi σ v d := by
  simp only [Psi, det2_rot, kappa_rot]

/-- The Jacobi determinant of two gradients is unchanged when both gradients turn by the rotation
(the gradients of rotation-invariant fields turn with the plane). -/
theorem jacobi_rot (θ : ℝ) (g h : R2) : det2 (rot θ g) (rot θ h) = det2 g h := det2_rot θ g h

theorem normSq_mirror (p : R2) : normSq (mirror p) = normSq p := by
  simp only [normSq, mirror]; ring

theorem dot_mirror (p q : R2) : dot (mirror p) (mirror q) = dot p q := by
  simp only [dot, mirror]; ring

theorem det2_mirror (p q : R2) : det2 (mirror p) (mirror q) = -det2 p q := by
  simp only [det2, mirror]; ring

theorem kappa_mirror (σ : ℝ) (d : R2) : kappa σ (mirror d) = kappa σ d := by
  simp only [kappa, normSq_mirror]

/-- Under a mirror image the divergence potential is unchanged... -/
theorem Phi_mirror (σ : ℝ) (v d : R2) : Phi σ (mirror v) (mirror d) = Phi σ v d := by
  simp only [Phi, dot_mirror, kappa_mirror]

/-- ...and the vorticity potential changes sign: clockwise and anticlockwise rotations differ. -/
theorem Psi_mirror (σ : ℝ) (v d : R2) : Psi σ (mirror v) (mirror d) = -Psi σ v d := by
  simp only [Psi, det2_mirror, kappa_mirror]; ring

/-- The Jacobi determinant is unchanged by a mirror image: the gradient of `φ` is mirrored, the
gradient of `ψ` is mirrored and changes sign, and the two sign changes cancel. -/
theorem jacobi_mirror (g h : R2) : det2 (mirror g) (-mirror h) = det2 g h := by
  simp only [det2, mirror, Prod.fst_neg, Prod.snd_neg]; ring

/-! ## Proposition 3: the tail of the footprint -/

/-- The radial antiderivative: `d/ds (-2πσ² e^{-s²/2σ²}) = 2π s e^{-s²/2σ²}`. -/
theorem hasDerivAt_tail (σ : ℝ) (hσ : σ ≠ 0) (s : ℝ) :
    HasDerivAt (fun u : ℝ => -(2 * π * σ ^ 2) * exp (-u ^ 2 / (2 * σ ^ 2)))
      (2 * π * s * exp (-s ^ 2 / (2 * σ ^ 2))) s := by
  have h1 : HasDerivAt (fun u : ℝ => -u ^ 2 / (2 * σ ^ 2)) (-(2 * s) / (2 * σ ^ 2)) s := by
    have := (hasDerivAt_pow 2 s).neg.div_const (2 * σ ^ 2)
    convert this using 1
    norm_num
  convert (h1.exp).const_mul (-(2 * π * σ ^ 2)) using 1
  field_simp

theorem tendsto_tail (σ : ℝ) (hσ : σ ≠ 0) :
    Tendsto (fun u : ℝ => -(2 * π * σ ^ 2) * exp (-u ^ 2 / (2 * σ ^ 2))) atTop (𝓝 0) := by
  have hpos : 0 < 2 * σ ^ 2 := by positivity
  have h : Tendsto (fun u : ℝ => -u ^ 2 / (2 * σ ^ 2)) atTop atBot := by
    have : Tendsto (fun u : ℝ => u ^ 2) atTop atTop := tendsto_pow_atTop two_ne_zero
    simpa [neg_div] using this.atTop_div_const hpos
  simpa using (tendsto_exp_atBot.comp h).const_mul (-(2 * π * σ ^ 2))

/-- **Proposition 3.** The mass of the footprint outside the disk of radius `ρ`, in polar
coordinates: `∫_ρ^∞ 2π s e^{-s²/2σ²} ds = 2πσ² e^{-ρ²/2σ²}`; the whole plane (`ρ = 0`) holds
`2πσ²`, so the fraction left outside is `e^{-ρ²/2σ²}`. -/
theorem gaussian_tail (σ : ℝ) (hσ : σ ≠ 0) (ρ : ℝ) (hρ : 0 ≤ ρ) :
    ∫ s in Ioi ρ, 2 * π * s * exp (-s ^ 2 / (2 * σ ^ 2)) =
      2 * π * σ ^ 2 * exp (-ρ ^ 2 / (2 * σ ^ 2)) := by
  have hderiv : ∀ x ∈ Ioi ρ, HasDerivAt (fun u : ℝ => -(2 * π * σ ^ 2) * exp (-u ^ 2 / (2 * σ ^ 2)))
      (2 * π * x * exp (-x ^ 2 / (2 * σ ^ 2))) x := fun x _ => hasDerivAt_tail σ hσ x
  have hcont : ContinuousWithinAt (fun u : ℝ => -(2 * π * σ ^ 2) * exp (-u ^ 2 / (2 * σ ^ 2)))
      (Ici ρ) ρ := (hasDerivAt_tail σ hσ ρ).continuousAt.continuousWithinAt
  have hnonneg : ∀ x ∈ Ioi ρ, 0 ≤ 2 * π * x * exp (-x ^ 2 / (2 * σ ^ 2)) := by
    intro x hx
    have : 0 ≤ x := le_of_lt (lt_of_le_of_lt hρ hx)
    positivity
  have hint := integrableOn_Ioi_deriv_of_nonneg hcont hderiv hnonneg (tendsto_tail σ hσ)
  rw [integral_Ioi_of_hasDerivAt_of_tendsto hcont hderiv hint (tendsto_tail σ hσ)]
  ring

/-- At `ρ = 5σ` the fraction of the mass left outside is `e^{-25/2} ≈ 3.7 × 10⁻⁶`. -/
theorem tail_exponent_at_5sigma (σ : ℝ) (hσ : σ ≠ 0) :
    -(5 * σ) ^ 2 / (2 * σ ^ 2) = -(25 / 2 : ℝ) := by
  field_simp
  ring

/-! ## Proposition 1: the closed-form Helmholtz-Hodge potentials -/

theorem normSq_eq_zero_iff (d : R2) : normSq d = 0 ↔ d = 0 := by
  constructor
  · intro h
    have h1 : d.1 = 0 := by unfold normSq at h; nlinarith [sq_nonneg d.1, sq_nonneg d.2]
    have h2 : d.2 = 0 := by unfold normSq at h; nlinarith [sq_nonneg d.1, sq_nonneg d.2]
    exact Prod.ext h1 h2
  · rintro rfl; simp [normSq]

/-- `κσ` away from the origin, as a quotient of smooth functions. -/
def kappaS (σ : ℝ) (d : R2) : ℝ := σ ^ 2 * (1 - exp (-normSq d / (2 * σ ^ 2))) / normSq d

theorem kappa_eq_kappaS (σ : ℝ) {d : R2} (hd : normSq d ≠ 0) : kappa σ d = kappaS σ d := by
  simp only [kappa, kappaS, hd, ↓reduceIte]; ring

theorem continuous_normSq : Continuous normSq := by
  unfold normSq; fun_prop

/-- Near a point other than the origin, `κσ` is the smooth quotient `kappaS`. -/
theorem kappa_eventuallyEq (σ : ℝ) {d : R2} (hd : normSq d ≠ 0) : kappa σ =ᶠ[𝓝 d] kappaS σ := by
  filter_upwards [continuous_normSq.continuousAt.eventually_ne hd] with p hp
  exact kappa_eq_kappaS σ hp

/-- A potential of the form `(α p₁ + β p₂) κσ(p)`: `φ` is the case `(α, β) = (v₁, v₂)` and `ψ`
the case `(α, β) = (v₂, -v₁)`. -/
def linK (σ α β : ℝ) (p : R2) : ℝ := (α * p.1 + β * p.2) * kappaS σ p

theorem differentiableAt_linK (σ α β : ℝ) {d : R2} (hd : normSq d ≠ 0) :
    DifferentiableAt ℝ (linK σ α β) d := by
  unfold linK kappaS
  have hd' : d.1 ^ 2 + d.2 ^ 2 ≠ 0 := by simpa [normSq] using hd
  unfold normSq
  fun_prop (disch := exact hd')

/-- The derivative of `F` at `d` in the direction `e`, read on the line `t ↦ d + t e`. -/
theorem fderiv_apply_eq_of_line {F : R2 → ℝ} {d e : R2} {L : ℝ} (hF : DifferentiableAt ℝ F d)
    (hL : HasDerivAt (fun t : ℝ => F (d + t • e)) L 0) : fderiv ℝ F d e = L := by
  have hline : HasDerivAt (fun t : ℝ => d + t • e) e 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const e).const_add d
  have h := hF.hasFDerivAt.comp_hasDerivAt_of_eq (0 : ℝ) hline (by simp)
  exact h.unique hL

/-- The derivative of `(α p₁ + β p₂) κσ(p)` along the line `d + t e`, with
`∂κσ/∂p = p (e^{-|p|²/2σ²} - 2 κσ(p)) / |p|²`. -/
theorem hasDerivAt_linK_line (σ α β : ℝ) (hσ : σ ≠ 0) {d : R2} (hd : normSq d ≠ 0) (e : R2) :
    HasDerivAt (fun t : ℝ => linK σ α β (d + t • e))
      ((α * e.1 + β * e.2) * kappaS σ d + (α * d.1 + β * d.2) * (d.1 * e.1 + d.2 * e.2) *
        (exp (-normSq d / (2 * σ ^ 2)) - 2 * kappaS σ d) / normSq d) 0 := by
  have hn0 : d.1 ^ 2 + d.2 ^ 2 ≠ 0 := by simpa [normSq] using hd
  have hx : HasDerivAt (fun t : ℝ => d.1 + t * e.1) e.1 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const e.1).const_add d.1
  have hy : HasDerivAt (fun t : ℝ => d.2 + t * e.2) e.2 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const e.2).const_add d.2
  -- the squared norm along the line, the footprint, the kernel and the linear factor
  have hn : HasDerivAt (fun t : ℝ => (d.1 + t * e.1) ^ 2 + (d.2 + t * e.2) ^ 2)
      (2 * (d.1 * e.1 + d.2 * e.2)) 0 :=
    ((hx.pow 2).add (hy.pow 2)).congr_deriv (by simp; ring)
  have hE : HasDerivAt
      (fun t : ℝ => exp (-((d.1 + t * e.1) ^ 2 + (d.2 + t * e.2) ^ 2) / (2 * σ ^ 2)))
      (exp (-(d.1 ^ 2 + d.2 ^ 2) / (2 * σ ^ 2)) *
        (-(2 * (d.1 * e.1 + d.2 * e.2)) / (2 * σ ^ 2))) 0 :=
    ((hn.neg.div_const (2 * σ ^ 2)).exp).congr_deriv (by simp)
  have hK : HasDerivAt
      (fun t : ℝ => σ ^ 2 * (1 - exp (-((d.1 + t * e.1) ^ 2 + (d.2 + t * e.2) ^ 2) /
        (2 * σ ^ 2))) / ((d.1 + t * e.1) ^ 2 + (d.2 + t * e.2) ^ 2))
      ((σ ^ 2 * -(exp (-(d.1 ^ 2 + d.2 ^ 2) / (2 * σ ^ 2)) * (-(2 * (d.1 * e.1 + d.2 * e.2)) /
          (2 * σ ^ 2))) * (d.1 ^ 2 + d.2 ^ 2) -
          σ ^ 2 * (1 - exp (-(d.1 ^ 2 + d.2 ^ 2) / (2 * σ ^ 2))) *
          (2 * (d.1 * e.1 + d.2 * e.2))) / (d.1 ^ 2 + d.2 ^ 2) ^ 2) 0 :=
    (((hE.const_sub 1).const_mul (σ ^ 2)).div hn (by simpa using hn0)).congr_deriv (by simp)
  have hlin : HasDerivAt (fun t : ℝ => α * (d.1 + t * e.1) + β * (d.2 + t * e.2))
      (α * e.1 + β * e.2) 0 :=
    (hx.const_mul α).add (hy.const_mul β)
  have h := hlin.mul hK
  have hfun : (fun t : ℝ => linK σ α β (d + t • e)) = fun t : ℝ =>
      (α * (d.1 + t * e.1) + β * (d.2 + t * e.2)) *
        (σ ^ 2 * (1 - exp (-((d.1 + t * e.1) ^ 2 + (d.2 + t * e.2) ^ 2) / (2 * σ ^ 2))) /
          ((d.1 + t * e.1) ^ 2 + (d.2 + t * e.2) ^ 2)) := by
    funext t; simp [linK, kappaS, normSq]
  rw [hfun]
  refine h.congr_deriv ?_
  simp only [kappaS, normSq, zero_mul, add_zero]
  generalize exp (-(d.1 ^ 2 + d.2 ^ 2) / (2 * σ ^ 2)) = E
  field_simp

theorem fderiv_linK (σ α β : ℝ) (hσ : σ ≠ 0) {d : R2} (hd : normSq d ≠ 0) (e : R2) :
    fderiv ℝ (linK σ α β) d e =
      (α * e.1 + β * e.2) * kappaS σ d + (α * d.1 + β * d.2) * (d.1 * e.1 + d.2 * e.2) *
        (exp (-normSq d / (2 * σ ^ 2)) - 2 * kappaS σ d) / normSq d :=
  fderiv_apply_eq_of_line (differentiableAt_linK σ α β hd) (hasDerivAt_linK_line σ α β hσ hd e)

theorem Phi_eventuallyEq (σ : ℝ) (v : R2) {d : R2} (hd : normSq d ≠ 0) :
    Phi σ v =ᶠ[𝓝 d] linK σ v.1 v.2 := by
  filter_upwards [kappa_eventuallyEq σ hd] with p hp
  simp only [Phi, linK, dot, hp]

theorem Psi_eventuallyEq (σ : ℝ) (v : R2) {d : R2} (hd : normSq d ≠ 0) :
    Psi σ v =ᶠ[𝓝 d] linK σ v.2 (-v.1) := by
  filter_upwards [kappa_eventuallyEq σ hd] with p hp
  simp only [Psi, linK, det2, hp]; ring

theorem differentiableAt_Phi (σ : ℝ) (v : R2) {d : R2} (hd : normSq d ≠ 0) :
    DifferentiableAt ℝ (Phi σ v) d :=
  (differentiableAt_linK σ v.1 v.2 hd).congr_of_eventuallyEq (Phi_eventuallyEq σ v hd)

theorem differentiableAt_Psi (σ : ℝ) (v : R2) {d : R2} (hd : normSq d ≠ 0) :
    DifferentiableAt ℝ (Psi σ v) d :=
  (differentiableAt_linK σ v.2 (-v.1) hd).congr_of_eventuallyEq (Psi_eventuallyEq σ v hd)

/-- **Proposition 1, one agent.** At every offset `d ≠ 0`,
`∇(u κσ(d)) - ∇⊥(w κσ(d)) = v e^{-|d|²/2σ²}`, with `∇⊥ = (-∂_y, ∂_x)`; written with the
potentials `Φ = u κσ` and `Ψ = -w κσ`, this is `∇Φ + ∇⊥Ψ = v e^{-|d|²/2σ²}`. -/
theorem hhd_single_agent (σ : ℝ) (hσ : σ ≠ 0) (v d : R2) (hd : d ≠ 0) :
    fderiv ℝ (Phi σ v) d (1, 0) - fderiv ℝ (Psi σ v) d (0, 1) = v.1 * lens σ d ∧
    fderiv ℝ (Phi σ v) d (0, 1) + fderiv ℝ (Psi σ v) d (1, 0) = v.2 * lens σ d := by
  have hd' : normSq d ≠ 0 := fun h => hd ((normSq_eq_zero_iff d).1 h)
  rw [(Phi_eventuallyEq σ v hd').fderiv_eq, (Psi_eventuallyEq σ v hd').fderiv_eq,
    fderiv_linK σ _ _ hσ hd', fderiv_linK σ _ _ hσ hd', fderiv_linK σ _ _ hσ hd',
    fderiv_linK σ _ _ hσ hd']
  have hn0 : d.1 ^ 2 + d.2 ^ 2 ≠ 0 := by simpa [normSq] using hd'
  simp only [kappaS, lens, normSq]
  generalize exp (-(d.1 ^ 2 + d.2 ^ 2) / (2 * σ ^ 2)) = E
  constructor <;> field_simp <;> ring

/-- **Proposition 1.** The potentials of a recording, `φ(r) = c Σᵢ Σₜ uᵢ(t) κσ(r - xᵢ(t))` and
`ψ(r) = -c Σᵢ Σₜ wᵢ(t) κσ(r - xᵢ(t))` (the paper takes `c = 1/(N T σ v̄)`), satisfy
`∇φ + ∇⊥ψ = c Σᵢ Σₜ vᵢ(t) e^{-|r - xᵢ(t)|²/2σ²}` at every point `r` that no agent occupies. -/
theorem hhd_recording {N T : ℕ} (σ c : ℝ) (hσ : σ ≠ 0) (x v : Fin N → Fin T → R2) (r : R2)
    (hr : ∀ i t, r ≠ x i t) :
    let φ := fun p : R2 => c * ∑ i, ∑ t, Phi σ (v i t) (p - x i t)
    let ψ := fun p : R2 => c * ∑ i, ∑ t, Psi σ (v i t) (p - x i t)
    fderiv ℝ φ r (1, 0) - fderiv ℝ ψ r (0, 1) = c * ∑ i, ∑ t, (v i t).1 * lens σ (r - x i t) ∧
    fderiv ℝ φ r (0, 1) + fderiv ℝ ψ r (1, 0) = c * ∑ i, ∑ t, (v i t).2 * lens σ (r - x i t) := by
  intro φ ψ
  have hne : ∀ i t, normSq (r - x i t) ≠ 0 := fun i t h =>
    hr i t (sub_eq_zero.1 ((normSq_eq_zero_iff _).1 h))
  have dPhi : ∀ i t, DifferentiableAt ℝ (fun p : R2 => Phi σ (v i t) (p - x i t)) r := fun i t =>
    (differentiableAt_comp_sub (x i t)).2 (differentiableAt_Phi σ (v i t) (hne i t))
  have dPsi : ∀ i t, DifferentiableAt ℝ (fun p : R2 => Psi σ (v i t) (p - x i t)) r := fun i t =>
    (differentiableAt_comp_sub (x i t)).2 (differentiableAt_Psi σ (v i t) (hne i t))
  -- the derivative of `c Σᵢ Σₜ F(p - xᵢ(t))` is `c Σᵢ Σₜ (∇F)(r - xᵢ(t))`
  have lin : ∀ (F : R2 → R2 → ℝ)
      (hF : ∀ i t, DifferentiableAt ℝ (fun p : R2 => F (v i t) (p - x i t)) r) (e : R2),
      fderiv ℝ (fun p : R2 => c * ∑ i, ∑ t, F (v i t) (p - x i t)) r e =
        c * ∑ i, ∑ t, fderiv ℝ (F (v i t)) (r - x i t) e := by
    intro F hF e
    have hin : ∀ i, DifferentiableAt ℝ (fun p : R2 => ∑ t, F (v i t) (p - x i t)) r := fun i =>
      (HasFDerivAt.fun_sum fun t _ => (hF i t).hasFDerivAt).differentiableAt
    have hall : DifferentiableAt ℝ (fun p : R2 => ∑ i, ∑ t, F (v i t) (p - x i t)) r :=
      (HasFDerivAt.fun_sum fun i _ => (hin i).hasFDerivAt).differentiableAt
    rw [fderiv_const_mul hall c, fderiv_fun_sum fun i _ => hin i, smul_apply,
      smul_eq_mul, sum_apply]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [fderiv_fun_sum fun t _ => hF i t, sum_apply]
    exact Finset.sum_congr rfl fun t _ => by rw [fderiv_comp_sub]
  have gPhi : ∀ e : R2, fderiv ℝ φ r e = c * ∑ i, ∑ t, fderiv ℝ (Phi σ (v i t)) (r - x i t) e :=
    lin (Phi σ) dPhi
  have gPsi : ∀ e : R2, fderiv ℝ ψ r e = c * ∑ i, ∑ t, fderiv ℝ (Psi σ (v i t)) (r - x i t) e :=
    lin (Psi σ) dPsi
  have hs := fun i t => hhd_single_agent σ hσ (v i t) (r - x i t)
    (fun h => hr i t (sub_eq_zero.1 h))
  rw [gPhi, gPhi, gPsi, gPsi]
  constructor
  · rw [← mul_sub, ← Finset.sum_sub_distrib]
    congr 1; refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun t _ => (hs i t).1
  · rw [← mul_add, ← Finset.sum_add_distrib]
    congr 1; refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun t _ => (hs i t).2

end TopologicalKinematics
