"""Question 3.2 - Case B: insufficient fuel, two-stage minimum-landing-error guidance (Blackmore et al., 2010).
Run from the starter_kit folder:   python case_b.py
"""
import numpy as np
import matplotlib.pyplot as plt

from lander.params import P
from lander.lcvx import LanderSOCP, solve_min_fuel, solve_min_landing_error, is_lossless
from lander.pointmass import simulate

x0 = np.array([-6.0, 9.0, -6.0, 0.0])      # Case B: x, y, vx, vy
m0 = P.m_dry + 0.17                         # severely depleted tank: 0.17 kg of propellant
N = 40

# 1) The standard problem (land at the pad center) has no solution
sol_fuel, evals_fuel = solve_min_fuel(LanderSOCP(P, N=N), x0, m0)
n_feas = sum(np.isfinite(f) for _, f in evals_fuel)
print(f"Minimum fuel (x_N = 0): {'no solution' if sol_fuel is None else 'solution found'} "
      f"({n_feas} feasible t_f out of {len(evals_fuel)} tested)")

# 2) Two-stage guidance
sol, d_star, evals = solve_min_landing_error(P, x0, m0, N=N)
sim = simulate(sol, P, mode="accel")
print(f"Stage 1: minimum landing error d* = {d_star:.3f} m, flight time t_f* = {sol.tgo:.3f} s")
print(f"Stage 2: fuel = {1000 * sol.fuel:.1f} g (out of {1000 * (m0 - P.m_dry):.0f} g available), "
      f"|x_N| = {sol.landing_error:.3f} m, lossless relaxation: {is_lossless(sol)}")
print(f"Simulation: final x = {sim['x_final'][0]:.3f} m, final y = {sim['x_final'][1]:.1e} m, "
      f"final speed = {sim['final_vel_err']:.1e} m/s, thrust violation = {sim['thrust_violation']:.3f} N")

# 3) Figures for the report
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

# (a) trajectory and glide-slope cone anchored at the touchdown point x_N
traj = sim["traj"]
xN = sol.p[-1, 0]
ax[0].plot(traj[:, 0], traj[:, 1], label="simulated trajectory")
ax[0].plot(sol.p[:, 0], sol.p[:, 1], "o", ms=3, label="SOCP nodes")
xs = np.array([xN - 12, xN, xN + 12])
ax[0].plot(xs, np.abs(xs - xN) * np.tan(np.radians(45)), "k--", lw=1, label="glide slope (apex at $x_N$)")
ax[0].plot([-0.5, 0.5], [0, 0], color="green", lw=5, label="pad center")
ax[0].set(xlabel="x [m]", ylabel="y [m]", title="Case B: trajectory", xlim=(-13, 3), ylim=(-0.5, 12))
ax[0].legend(fontsize=7)

# (b) thrust magnitude ||T_k|| with the bounds rho1, rho2
ax[1].step(sol.t[:-1], np.linalg.norm(sol.thrust, axis=1), where="post", label=r"$\|T_k\|$")
ax[1].axhline(P.rho1, color="k", ls=":", label=r"$\rho_1$")
ax[1].axhline(P.rho2, color="k", ls="--", label=r"$\rho_2$")
ax[1].set(xlabel="t [s]", ylabel="thrust [N]", title="Thrust profile")
ax[1].legend(fontsize=7)

# (c) stage 1: landing error as a function of t_f (certified solutions only)
t_ev = np.array([t for t, e in evals if np.isfinite(e)])
e_ev = np.array([e for t, e in evals if np.isfinite(e)])
ax[2].plot(t_ev, e_ev, "o-", ms=3)
ax[2].axvline(sol.tgo, color="r", ls="--", lw=1, label=f"$t_f^*$ = {sol.tgo:.2f} s")
ax[2].set(xlabel="$t_f$ [s]", ylabel="$|x_N|$ [m]", title="Stage 1: landing error vs $t_f$")
ax[2].legend(fontsize=7)

fig.tight_layout()
fig.savefig("case_b.pdf")
print("Figure saved: case_b.pdf")
plt.show()
