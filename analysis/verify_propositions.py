#!/usr/bin/env python3
"""
verify_propositions.py - symbolic and numerical checks of the propositions
===========================================================================
Paper: "Recovering a Taxonomy of Collective Behaviors from Trajectories Alone" (Ndinga Oba, 2026). Exact symbolic algebra and
calculus with SymPy, and high-precision quadrature:
  * Proposition 1: Helmholtz-Hodge decomposition of the Gaussian velocity field;
  * Proposition 2: invariance of the five kinematic fields under rotations;
  * Proposition 3: tail bound and truncation error at 5 sigma;
  * the closed-form gradients of the potentials and their evaluation at the agents (appendix A);
  * the rotation invariance of the magnitude of the traceless part of M2;
  * numerically, the envelope property of critical values (appendix G) and what the average at fixed
    positions does to a vortex that reverses or drifts (limitations).

Author: Olivier Vertu Ndinga Oba
"""

import sympy as sp
import numpy as np
from scipy import integrate

def verify_proposition_1():
    print("=" * 70)
    print("VERIFYING PROPOSITION 1: Helmholtz-Hodge Decomposition")
    print("=" * 70)
    
    # Define symbolic variables
    x, y, x0, y0 = sp.symbols('x y x0 y0', real=True)
    vx, vy = sp.symbols('vx vy', real=True)
    sigma = sp.symbols('sigma', positive=True)
    
    dx = x - x0
    dy = y - y0
    r_sq = dx**2 + dy**2
    
    # Kernel kappa(r) = (sigma^2 / r^2) * (1 - exp(-r^2 / (2 * sigma^2)))
    lens = sp.exp(- r_sq / (2 * sigma**2))
    kappa = (sigma**2 / r_sq) * (1 - lens)
    
    # Divergence potential Phi and Vorticity potential Psi
    # Phi = (v . d) * kappa
    # Psi = - det(v, d) * kappa = - (vx * dy - vy * dx) * kappa
    u = vx * dx + vy * dy
    w = vx * dy - vy * dx
    
    Phi = u * kappa
    Psi = - w * kappa
    
    print("Potentials defined:")
    print("  Phi(x, y) = (v · d) * kappa(d)")
    print("  Psi(x, y) = - det[v, d] * kappa(d)")
    
    # Compute gradients
    # grad Phi = (dPhi/dx, dPhi/dy)
    grad_Phi_x = sp.diff(Phi, x)
    grad_Phi_y = sp.diff(Phi, y)
    
    # grad_perp Psi = (-dPsi/dy, dPsi/dx)
    grad_perp_Psi_x = - sp.diff(Psi, y)
    grad_perp_Psi_y = sp.diff(Psi, x)
    
    # Reconstructed field: F = grad Phi + grad_perp Psi
    F_x = grad_Phi_x + grad_perp_Psi_x
    F_y = grad_Phi_y + grad_perp_Psi_y
    
    # Target Gaussian velocity field: V = (vx * lens, vy * lens)
    target_x = vx * lens
    target_y = vy * lens
    
    print("\nComputing symbolic difference F - v * lens...")
    diff_x = sp.simplify(sp.ratsimp(F_x - target_x))
    diff_y = sp.simplify(sp.ratsimp(F_y - target_y))
    
    print(f"  F_x - target_x = {diff_x}")
    print(f"  F_y - target_y = {diff_y}")
    
    assert diff_x == 0, "Proposition 1 failed for x-component!"
    assert diff_y == 0, "Proposition 1 failed for y-component!"
    print(">>> SUCCESS: F = ∇Phi + ∇^perp Psi ≡ v * exp(-r^2 / (2 sigma^2)) EXACTLY everywhere for r != 0.")
    
    # Check orthogonality of divergence and curl
    div_F = sp.diff(F_x, x) + sp.diff(F_y, y)
    laplace_Phi = sp.diff(Phi, x, 2) + sp.diff(Phi, y, 2)
    diff_div = sp.simplify(sp.ratsimp(div_F - laplace_Phi))
    print(f"\nChecking ∇·F == ΔPhi (div(grad_perp Psi) == 0): difference = {diff_div}")
    assert diff_div == 0
    
    curl_F = sp.diff(F_y, x) - sp.diff(F_x, y)
    laplace_Psi = sp.diff(Psi, x, 2) + sp.diff(Psi, y, 2)
    diff_curl = sp.simplify(sp.ratsimp(curl_F - laplace_Psi))
    print(f"Checking ∇×F == ΔPsi (curl(grad Phi) == 0): difference = {diff_curl}")
    assert diff_curl == 0
    print(">>> SUCCESS: Helmholtz-Hodge orthogonality verified symbolically.\n")

    # The Poisson equations of the paper, written against the Gaussian flux itself and not against F:
    #   Delta Phi = div(flux),  Delta Psi = rot(flux),  with rot W = d_x W_y - d_y W_x.
    div_target = sp.diff(target_x, x) + sp.diff(target_y, y)
    rot_target = sp.diff(target_y, x) - sp.diff(target_x, y)
    diff_poisson_phi = sp.simplify(sp.ratsimp(laplace_Phi - div_target))
    diff_poisson_psi = sp.simplify(sp.ratsimp(laplace_Psi - rot_target))
    print(f"Poisson: DeltaPhi - div(flux) = {diff_poisson_phi},  DeltaPsi - rot(flux) = {diff_poisson_psi}")
    assert diff_poisson_phi == 0 and diff_poisson_psi == 0
    # Same equation in the notation of Bhatia et al. (2013), Eq. 16: Delta R = -div(J w), J(a, b) = (-b, a)
    minus_div_J_target = -(sp.diff(-target_y, x) + sp.diff(target_x, y))
    assert sp.simplify(minus_div_J_target - rot_target) == 0
    print(">>> SUCCESS: Delta Phi = div(flux) and Delta Psi = -div(J flux) (Bhatia et al., Eq. 16) hold exactly.")

    # The potentials vanish at infinity: |Phi|, |Psi| <= |v| sigma^2 / r, since |v.d| <= |v| r and 0 <= 1 - e^{-s} <= 1.
    f_Phi = sp.lambdify((x, y, x0, y0, vx, vy, sigma), Phi, "numpy")
    f_Psi = sp.lambdify((x, y, x0, y0, vx, vy, sigma), Psi, "numpy")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        r = 10 ** rng.uniform(0.5, 4); th = rng.uniform(0, 2 * np.pi)
        v = rng.normal(size=2); s = rng.uniform(0.3, 3)
        bound = np.hypot(v[0], v[1]) * s ** 2 / r
        for f in (f_Phi, f_Psi):
            worst = max(worst, abs(f(r * np.cos(th), r * np.sin(th), 0.0, 0.0, v[0], v[1], s)) / bound)
    print(f"Decay: largest |potential| / (|v| sigma^2 / r) over 600 random far points = {worst:.4f} (must be <= 1)")
    assert worst <= 1 + 1e-12
    print(">>> SUCCESS: both potentials vanish at infinity like 1/r, so the decomposition has no harmonic remainder.\n")


def verify_proposition_2():
    print("=" * 70)
    print("VERIFYING PROPOSITION 2: SO(2) Rotation Invariance")
    print("=" * 70)
    
    theta = sp.symbols('theta', real=True)
    sigma = sp.symbols('sigma', positive=True)
    vx, vy, dx, dy = sp.symbols('vx vy dx dy', real=True)
    
    # Original quantities
    normSq = dx**2 + dy**2
    lens = sp.exp(- normSq / (2 * sigma**2))
    kappa = (sigma**2 / normSq) * (1 - lens)
    u = vx * dx + vy * dy
    w = vx * dy - vy * dx
    Phi = u * kappa
    Psi = - w * kappa
    
    # Rotated quantities under R(theta)
    # R(theta) = [[cos theta, -sin theta], [sin theta, cos theta]]
    cos_t = sp.cos(theta)
    sin_t = sp.sin(theta)
    
    vx_rot = cos_t * vx - sin_t * vy
    vy_rot = sin_t * vx + cos_t * vy
    
    dx_rot = cos_t * dx - sin_t * dy
    dy_rot = sin_t * dx + cos_t * dy
    
    normSq_rot = dx_rot**2 + dy_rot**2
    lens_rot = sp.exp(- normSq_rot / (2 * sigma**2))
    kappa_rot = (sigma**2 / normSq_rot) * (1 - lens_rot)
    
    u_rot = vx_rot * dx_rot + vy_rot * dy_rot
    w_rot = vx_rot * dy_rot - vy_rot * dx_rot
    Phi_rot = u_rot * kappa_rot
    Psi_rot = - w_rot * kappa_rot
    
    print("Checking invariance of elementary building blocks:")
    diff_norm = sp.simplify(normSq_rot - normSq)
    print(f"  ||R d||^2 - ||d||^2 = {diff_norm}")
    assert diff_norm == 0
    
    diff_u = sp.simplify(u_rot - u)
    print(f"  (R v · R d) - (v · d) = {diff_u}")
    assert diff_u == 0
    
    diff_w = sp.simplify(w_rot - w)
    print(f"  det(R v, R d) - det(v, d) = {diff_w}")
    assert diff_w == 0
    
    diff_Phi = sp.simplify(Phi_rot - Phi)
    print(f"  Phi(R v, R d) - Phi(v, d) = {diff_Phi}")
    assert diff_Phi == 0
    
    diff_Psi = sp.simplify(Psi_rot - Psi)
    print(f"  Psi(R v, R d) - Psi(v, d) = {diff_Psi}")
    assert diff_Psi == 0
    
    # Jacobi determinant J(grad Phi, grad Psi) = det(grad Phi, grad Psi)
    # Under rotation, if g1 and g2 are rotated by R(theta), det(R g1, R g2) = det(R) * det(g1, g2) = det(g1, g2)
    g1x, g1y, g2x, g2y = sp.symbols('g1x g1y g2x g2y', real=True)
    g1x_rot = cos_t * g1x - sin_t * g1y
    g1y_rot = sin_t * g1x + cos_t * g1y
    g2x_rot = cos_t * g2x - sin_t * g2y
    g2y_rot = sin_t * g2x + cos_t * g2y
    
    det_rot = g1x_rot * g2y_rot - g1y_rot * g2x_rot
    det_orig = g1x * g2y - g1y * g2x
    diff_det = sp.simplify(det_rot - det_orig)
    print(f"  Jacobi det(R g1, R g2) - det(g1, g2) = {diff_det}")
    assert diff_det == 0
    
    print(">>> SUCCESS: Proposition 2 (SO(2) rotation invariance) verified symbolically.\n")


def tail_fraction_quadrature(window=5.0):
    """The fraction of the mass of a footprint outside the window, by numerical quadrature (SciPy quad),
    and its difference with the closed form exp(-window^2 / 2)."""
    def f_num(r):
        return 2.0 * np.pi * r * np.exp(- r**2 / 2.0)
    num_integral, err = integrate.quad(f_num, window, np.inf)
    num_total, _ = integrate.quad(f_num, 0, np.inf)
    fraction = num_integral / num_total
    exact = float(sp.exp(-sp.Rational(window**2) / 2).evalf(50))
    return fraction, abs(fraction - exact), num_integral, err


def verify_proposition_3():
    print("=" * 70)
    print("VERIFYING PROPOSITION 3: Gaussian Tail Bound & 5-Sigma Cutoff")
    print("=" * 70)
    
    s, rho = sp.symbols('s rho', positive=True)
    sigma = sp.symbols('sigma', positive=True)
    
    # Integrand: 2 * pi * s * exp(-s^2 / (2 * sigma^2))
    integrand = 2 * sp.pi * s * sp.exp(- s**2 / (2 * sigma**2))
    
    # Indefinite integral
    antideriv = sp.integrate(integrand, s)
    print("Symbolic antiderivative of 2*pi*s*exp(-s^2/(2*sigma^2)) ds:")
    print(f"  F(s) = {antideriv}")
    
    # Tail integral from rho to infinity
    tail_integral = sp.integrate(integrand, (s, rho, sp.oo))
    expected_tail = 2 * sp.pi * sigma**2 * sp.exp(- rho**2 / (2 * sigma**2))
    print("Integral from rho to infinity:")
    print(f"  ∫_{{rho}}^{{oo}} 2*pi*s*exp(-s^2/(2*sigma^2)) ds = {tail_integral}")
    print(f"  Expected = {expected_tail}")
    diff_tail = sp.simplify(tail_integral - expected_tail)
    assert diff_tail == 0
    print(">>> SUCCESS: Closed-form tail bound formula matches exactly!")
    
    # Total mass over the entire plane (rho = 0)
    total_mass = sp.integrate(integrand, (s, 0, sp.oo))
    expected_total = 2 * sp.pi * sigma**2
    assert total_mass == expected_total
    print(f"Total mass on R^2: {total_mass}")
    
    # Fraction remaining outside rho = 5 * sigma
    fraction = sp.exp(- (5 * sigma)**2 / (2 * sigma**2))
    print(f"\nTail fraction outside 5*sigma: exp(-(5*sigma)^2 / (2*sigma^2)) = exp({sp.simplify(-(5*sigma)**2 / (2*sigma**2))})")
    assert sp.simplify(fraction - sp.exp(-sp.Rational(25, 2))) == 0
    print("  Exact value: e^(-25/2) = e^(-12.5)")
    
    val_num = float(sp.exp(-sp.Rational(25, 2)).evalf(50))
    print(f"  High-precision value: {val_num:.10e}")
    
    # Numerical integration check via SciPy quad (sigma = 1: the fraction does not depend on sigma)
    num_fraction, diff, num_integral, err = tail_fraction_quadrature(5.0)
    
    print("\nNumerical quadrature verification (SciPy quad):")
    print(f"  Numerical tail integral: {num_integral:.10e} (est. err: {err:.2e})")
    print(f"  Numerical fraction:      {num_fraction:.10e}")
    print(f"  Difference with theory:  {diff:.2e}")
    assert diff < 1e-14
    
    print(">>> SUCCESS: Proposition 3 verified symbolically and numerically.\n")


def verify_gradients():
    print("=" * 70)
    print("VERIFYING THE CLOSED-FORM GRADIENTS OF THE POTENTIALS (appendix A)")
    print("=" * 70)
    x, y, x0, y0, vx, vy = sp.symbols('x y x0 y0 vx vy', real=True)
    sigma, s = sp.symbols('sigma s', positive=True)
    dx, dy = x - x0, y - y0
    r_sq = dx**2 + dy**2
    lens = sp.exp(-r_sq / (2 * sigma**2))
    kappa = (sigma**2 / r_sq) * (1 - lens)
    u, w = vx * dx + vy * dy, vx * dy - vy * dx
    lam = (lens - 2 * kappa) / r_sq                      # the factor lambda of appendix A
    claimed_phi = (kappa * vx + lam * u * dx, kappa * vy + lam * u * dy)
    claimed_psi = (kappa * (-vy) + lam * w * dx, kappa * vx + lam * w * dy)     # v_perp = (-vy, vx)
    for name, f, claimed in (("u kappa", u * kappa, claimed_phi), ("w kappa", w * kappa, claimed_psi)):
        diff = [sp.simplify(sp.diff(f, var) - c) for var, c in zip((x, y), claimed)]
        print(f"  grad({name}) - closed form = {diff}")
        assert diff == [0, 0]
    # the limit of lambda at d = 0, with s = |d|^2
    lam_s = (sp.exp(-s / (2 * sigma**2)) - 2 * sigma**2 * (1 - sp.exp(-s / (2 * sigma**2))) / s) / s
    lim = sp.limit(lam_s, s, 0)
    print(f"  lim lambda as |d| -> 0 = {lim}")
    assert sp.simplify(lim + 1 / (4 * sigma**2)) == 0
    # the code: expm1 and a floor of 1e-12 m^2 on |d|^2; at an agent the gradient of u kappa is v kappa(0) = v / 2
    sig, v = 0.05, np.array([0.03, -0.02])
    for dist in (0.0, 1e-7, 1e-5, 1e-3):
        d = np.array([dist, 0.0]); d2 = max(d @ d, 1e-12)
        g = sig**2 * (-np.expm1(-d2 / (2 * sig**2))) / d2
        lam_code = (np.exp(-d2 / (2 * sig**2)) - 2 * g) / d2
        grad_code = v * g + (v @ d) * d * lam_code
        dd = sp.Float(dist, 50); ss = dd**2
        g_ex = sp.Rational(1, 2) if dist == 0 else sig**2 * (1 - sp.exp(-ss / (2 * sig**2))) / ss
        lam_ex = -1 / (4 * sp.Float(sig, 50)**2) if dist == 0 else (sp.exp(-ss / (2 * sig**2)) - 2 * g_ex) / ss
        grad_ex = np.array([float(v[0] * g_ex + v[0] * dd * dd * lam_ex), float(v[1] * g_ex)])
        err = np.abs(grad_code - grad_ex).max() / np.linalg.norm(v)
        print(f"  |d| = {dist:.0e} m: relative error of the gradient as the code computes it {err:.1e}")
        assert err < 1e-8          # below single precision, in which the fields are stored
    print(">>> SUCCESS: gradients in closed form, finite and accurate at the agents.\n")


def verify_traceless_invariant():
    print("=" * 70)
    print("VERIFYING THE INVARIANCE OF THE MAGNITUDE OF THE TRACELESS PART OF M2")
    print("=" * 70)
    a, b, c, th = sp.symbols('a b c theta', real=True)
    M = sp.Matrix([[a, b], [b, c]])
    Q = M - (M.trace() / 2) * sp.eye(2)
    R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    Qr = R * Q * R.T
    diff = sp.simplify(sum(e**2 for e in Qr) - sum(e**2 for e in Q))
    print(f"  |R Q R^T|^2 - |Q|^2 = {diff}")
    assert diff == 0
    print(">>> SUCCESS: the magnitude of the traceless part is rotation invariant.\n")


def _synthetic_psi(X, points, sigma):
    """psi of the paper (closed form, float64) at the given points, from tracks X (T, N, 2), dt = 1"""
    V = np.diff(X, axis=0); P = X[:-1]
    vbar = np.linalg.norm(V, axis=2).mean()
    d = points[:, None, None, :] - P[None]
    d2 = np.maximum((d**2).sum(-1), 1e-24)
    kap = sigma**2 * (-np.expm1(-d2 / (2 * sigma**2))) / d2
    w = V[None, ..., 0] * d[..., 1] - V[None, ..., 1] * d[..., 0]
    return -(w * kap).sum((1, 2)) / (X.shape[1] * len(P) * sigma * vbar)


def verify_envelope_and_average():
    from scipy.optimize import minimize
    print("=" * 70)
    print("VERIFYING THE ENVELOPE PROPERTY AND THE AVERAGE AT FIXED POSITIONS (synthetic milling group)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    sigma, N, T = 0.07, 20, 300
    ang0, rad = rng.uniform(0, 2 * np.pi, N), rng.uniform(0.15, 0.30, N)
    tt = np.linspace(0, 6 * np.pi, T)
    X = np.stack([np.stack([0.1 + rad * np.cos(ang0 + t), -0.05 + rad * np.sin(ang0 + t)], 1) for t in tt])
    X += 0.01 * rng.standard_normal(X.shape)
    f = lambda r, Y: float(_synthetic_psi(Y, np.asarray(r)[None], sigma)[0])
    argmin = lambda Y, r0: minimize(lambda r: f(r, Y), r0, method="Nelder-Mead",
                                    options=dict(xatol=1e-12, fatol=1e-16, maxiter=4000))
    m0 = argmin(X, np.array([0.1, -0.05])); r_star = m0.x
    h, eps = 0.02 * rng.standard_normal(X.shape), 1e-4
    total = (argmin(X + eps * h, r_star).fun - m0.fun) / eps          # the minimum moves with the tracks
    fixed = (f(r_star, X + eps * h) - m0.fun) / eps                   # the seed stays where it was
    gap_crit = abs(total - fixed) / abs(total)
    r_o = X[100, 0].copy()                                             # a seed that moves with one agent
    total_o = (f((X + eps * h)[100, 0], X + eps * h) - f(r_o, X)) / eps
    fixed_o = (f(r_o, X + eps * h) - f(r_o, X)) / eps
    gap_ord = abs(total_o - fixed_o) / abs(total_o)
    print(f"  minimum of psi: total {total:+.6f}, fixed seed {fixed:+.6f}, relative gap {gap_crit:.1e}")
    print(f"  ordinary point: total {total_o:+.6f}, fixed seed {fixed_o:+.6f}, relative gap {gap_ord:.1e}")
    assert gap_crit < 1e-5 and gap_ord > 0.5
    # a vortex that reverses halfway, and one that drifts, averaged at fixed positions
    T2 = 400; t2 = np.linspace(0, 8 * np.pi, T2)
    def milling(direction, drift):
        ang = ang0[None, :] + direction(t2)[:, None]
        cx = drift * np.linspace(-0.5, 0.5, T2)[:, None]
        return np.stack([cx + rad * np.cos(ang), rad * np.sin(ang)], 2)
    grid = np.stack(np.meshgrid(np.linspace(-1, 1, 81), np.linspace(-0.5, 0.5, 41)), -1).reshape(-1, 2)
    ref = np.abs(_synthetic_psi(milling(lambda t: t, 0.0), grid, sigma)).max()
    rev = np.abs(_synthetic_psi(milling(lambda t: np.where(t < t[-1] / 2, t, t[-1] - t), 0.0), grid, sigma)).max()
    dri = np.abs(_synthetic_psi(milling(lambda t: t, 1.0), grid, sigma)).max()
    print(f"  |psi| max: reversing halfway {rev / ref:.1e}, drifting by 1 m {dri / ref:.2f} of the fixed vortex")
    assert rev / ref < 0.01 and dri / ref < 0.7        # the reversed half cancels the first up to one step of the
                                                       # forward differences
    print(">>> SUCCESS: critical values are stationary under the motion of their point; averaging at fixed "
          "positions cancels a reversing vortex and smears a drifting one.\n")


if __name__ == "__main__":
    print("STARTING FULL MATHEMATICAL CORE VERIFICATION\n")
    verify_proposition_1()
    verify_proposition_2()
    verify_proposition_3()
    verify_gradients()
    verify_traceless_invariant()
    verify_envelope_and_average()
    print("=" * 70)
    print("ALL PROPOSITIONS (1, 2, 3) AND THE STATEMENTS OF APPENDICES A AND G VERIFIED!")
    print("=" * 70)
