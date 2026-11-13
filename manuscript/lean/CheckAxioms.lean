import TopologicalKinematics.Basic

/-! Run `lake env lean CheckAxioms.lean`: every theorem below must depend only on the standard
axioms `propext`, `Classical.choice` and `Quot.sound` (no `sorryAx`). -/

open TopologicalKinematics

#print axioms hhd_single_agent
#print axioms hhd_recording
#print axioms gaussian_tail
#print axioms tail_exponent_at_5sigma
#print axioms Phi_rot
#print axioms Psi_rot
#print axioms jacobi_rot
#print axioms Phi_mirror
#print axioms Psi_mirror
#print axioms jacobi_mirror
