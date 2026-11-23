"""Question 5: Robust, hazard-aware SCP + Cascade Controller (Windy Hazard Landing Challenge).

Architecture:
  - Online wind estimation: GustEstimator (OU exponential filter) + known mean wind & altitude shear.
  - Sequential Convex Programming (SCP): linearizes convexified obstacle exteriors (inflated by the
    lander's bounding box R_X, R_Y) into separating half-spaces H_k^T r_k >= b_k with trust region.
  - Analytical OU chance-constraint tightening: tightens obstacle half-spaces using the propagated
    OU gust innovation variance and attitude-tracking tube margin.
"""
import numpy as np

from lander.controllers import CascadeController
from lander.geometry import closest_point_convex, obstacle_polygons
from lander.lcvx import LanderSOCP, golden_section, is_lossless
from lander.params import LEG_DOWN, P as PARAMS, R_X, R_Y
from lander.wind import GustEstimator, known_wind


def _seg_intersects_poly(a, b, poly, pad_x=0.0, pad_y=0.0, n_samples=24):
    """Check if segment [a, b] comes within (pad_x, pad_y) of a convex polygon."""
    for t in np.linspace(0.0, 1.0, n_samples):
        p = (1.0 - t) * a + t * b
        _, d, n = closest_point_convex(poly, p)
        if d < pad_x * abs(n[0]) + pad_y * abs(n[1]):
            return True
    return False


def _initial_path(r0, rgate, polys, kinds, N, margin=0.35):
    """Build a collision-free initial guess pbar (N+1, 2) by routing around blocking obstacles."""
    pts = [r0.copy(), rgate.copy()]
    for _ in range(3):
        changed = False
        new_pts = [pts[0]]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            blocker = None
            for poly, kind in zip(polys, kinds):
                if kind == "ground" and np.max(poly[:, 1]) <= 0.05:
                    continue
                if _seg_intersects_poly(p0, p1, poly, R_X + margin, R_Y + margin):
                    blocker = poly
                    break
            if blocker is not None:
                cands = [
                    np.array([0.0, float(np.max(blocker[:, 1])) + R_Y + margin + 1.2]),
                    blocker[int(np.argmax(blocker[:, 1]))] + np.array([0.0, R_Y + margin + 0.8]),
                    blocker[int(np.argmin(blocker[:, 0]))] + np.array([-(R_X + margin + 0.8), 0.4]),
                    blocker[int(np.argmax(blocker[:, 0]))] + np.array([R_X + margin + 0.8, 0.4]),
                ]
                best_wp, best_len = None, np.inf
                for wp in cands:
                    if wp[1] < 0.8 or wp[1] > 13.5:
                        continue
                    L = np.linalg.norm(wp - p0) + np.linalg.norm(p1 - wp)
                    if L < best_len:
                        best_len, best_wp = L, wp
                if best_wp is not None and np.linalg.norm(best_wp - p0) > 0.3 and np.linalg.norm(p1 - best_wp) > 0.3:
                    new_pts.append(best_wp)
                    changed = True
            new_pts.append(p1)
        pts = new_pts
        if not changed:
            break

    segs = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
    tot = sum(segs)
    if tot < 1e-6:
        return np.linspace(r0, rgate, N + 1)
    cum = np.concatenate([[0.0], np.cumsum(segs)]) / tot
    pbar = np.zeros((N + 1, 2))
    for i, t in enumerate(np.linspace(0.0, 1.0, N + 1)):
        idx = min(max(int(np.searchsorted(cum, t, side="right")) - 1, 0), len(segs) - 1)
        alpha = (t - cum[idx]) / max(cum[idx + 1] - cum[idx], 1e-9)
        pbar[i] = (1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]
    return pbar


class HazardCascadePolicy(CascadeController):
    name = "Robust SCP Cascade"

    def __init__(self, spec, params=PARAMS, N=20, replan_every=25, n_scp=2, z_alpha=1.645,
                 trk_margin=0.20, use_hazards=True, use_robust=True, use_gust_est=True):
        self.spec = spec
        self.polys, self.kinds = obstacle_polygons(spec) if use_hazards else ([], [])
        self.pad_hw = 0.5 * (spec["pad"][1] - spec["pad"][0])
        self.n_scp = n_scp
        self.z_alpha = z_alpha
        self.trk_margin = trk_margin
        self.use_hazards = use_hazards
        self.use_robust = use_robust
        self.gust = GustEstimator(spec, params=params, enabled=use_gust_est)
        self.wind_spec = spec["wind"]

        super().__init__(
            gravity=spec["gravity"],
            params=params,
            N=N,
            replan_every=replan_every,
            derate=0.9,
            spec=spec,
            theta_plan_deg=25.0,
            theta_trk_deg=35.0,
            gamma_deg=10.0,
            x_tol=0.15,
            kp_trk=(3.0, 2.0),
            kd_trk=(3.8, 3.0),
            tgo_margin=1.1,
            tgo_grid=np.linspace(1.5, 9.5, 9),
        )
        c = 0.5 * (spec["pad"][0] + spec["pad"][1])
        lim = ((0.8 - c, 20.0 - 0.8 - c), 400 / 30.0 + 1.0 - (spec["pad"][2] + 0.6))
        self.G = LanderSOCP(
            self.p, N, spec["gravity"], mode="fuel", terminal="gate",
            theta_max_deg=25.0, gamma_deg=10.0, glide_from=1,
            thrust_frac=self.derate, y_gate=self.y_gate, v_gate=self.v_gate,
            x_tol=self.x_tol, shear=self.wind_spec["shear"],
            n_obs=len(self.polys), cone_deg=45.0, trust_region=True,
            w_slack=1e3, map_limits=lim,
        )
        # Detect narrow canyon flanking the pad -> add a staging gate above the canyon
        left_walls = [np.max(p[:, 1]) for p, k in zip(self.polys, self.kinds)
                      if k == "ground" and np.max(p[:, 1]) > 2.0 and np.max(p[:, 0]) < -0.5 and np.max(p[:, 0]) > -2.5]
        right_walls = [np.max(p[:, 1]) for p, k in zip(self.polys, self.kinds)
                       if k == "ground" and np.max(p[:, 1]) > 2.0 and np.min(p[:, 0]) > 0.5 and np.min(p[:, 0]) < 2.5]
        self.y_stage = None
        if left_walls and right_walls:
            self.y_stage = max(max(left_walls), max(right_walls)) + 1.4
            self.replan_every = 15
            self.trk_margin = 0.10
            self.kp_trk = np.array([5.0, 2.0])
            self.kd_trk = np.array([5.5, 3.0])
            self.G_stage = LanderSOCP(
                self.p, N, spec["gravity"], mode="fuel", terminal="gate",
                theta_max_deg=25.0, gamma_deg=0.0, glide_from=1,
                thrust_frac=self.derate, y_gate=self.y_stage, v_gate=0.5,
                x_tol=0.10, shear=self.wind_spec["shear"],
                n_obs=len(self.polys), cone_deg=None, trust_region=True,
                w_slack=1e3, map_limits=lim,
            )

    def reset(self):
        super().reset()
        self.gust.reset()
        self.a_prev = np.zeros(2)

    def _robust_margin(self, t_k, n):
        """Analytical OU chance-constraint + attitude-tracking margin along unit normal n."""
        if not self.use_robust:
            return 0.0
        gust_std = self.wind_spec["gust_std"]
        gust_tau = max(self.wind_spec["gust_tau"], 1e-3)
        torque_std = self.wind_spec.get("torque_std", 0.0)
        dt_eff = np.minimum(t_k, 0.55)
        sigma_xi = gust_std * np.sqrt(1.0 - np.exp(-2.0 * np.maximum(t_k, 0.0) / gust_tau))
        sigma_pos_x = 0.5 * sigma_xi * (dt_eff ** 2)
        torque_margin = 0.08 * torque_std
        return self.trk_margin + torque_margin + self.z_alpha * abs(n[0]) * sigma_pos_x

    def _halfspaces(self, pbar, tgo):
        """Linearize all inflated obstacle polygons around pbar (N+1, 2)."""
        N = self.N
        t_nodes = np.linspace(0.0, tgo, N + 1)
        hs = []
        for poly, kind in zip(self.polys, self.kinds):
            Hv = np.zeros((N + 1, 2))
            bv = np.zeros(N + 1)
            for k in range(1, N + 1):
                q, _, n = closest_point_convex(poly, pbar[k])
                ry_eff = R_Y
                if kind == "ground" and q[1] <= 0.05 and abs(q[0]) <= self.pad_hw + 0.35:
                    ry_eff = max(0.0, min(R_Y, pbar[k, 1] - 0.15))
                box_inf = R_X * abs(n[0]) + ry_eff * abs(n[1])
                rm = self._robust_margin(t_nodes[k], n) if k < N else 0.05
                Hv[k] = n
                bv[k] = float(n @ q + box_inf + rm)
            hs.append((Hv, bv))
        return hs

    def _wind_profile(self, tgo):
        """Expected lateral wind at the N interval starts (shear is inside LanderSOCP)."""
        t_nodes = np.linspace(0.0, tgo, self.N + 1)[:-1]
        base = self.wind_spec["mean"] + self.wind_spec["shear"] * LEG_DOWN / 10.0
        return base + self.gust.predict(t_nodes)

    def _use_staging(self, s):
        return (self.y_stage is not None and self.k < 250 and s[1] > self.y_stage - 0.5
                and (abs(s[0]) > 0.40 or abs(s[2]) > 0.55))

    def _solve(self, s, tgo, certify=True, pbar=None, n_scp=None):
        """SCP solve for a fixed tgo (targeting staging gate above canyon if needed)."""
        staging = self._use_staging(s)
        socp = self.G_stage if staging else self.G
        y_tgt = self.y_stage if staging else self.y_gate
        if pbar is None:
            if self.plan is not None and self.plan.get("staging", False) == staging:
                pbar = self.plan["p"]
            else:
                pbar = _initial_path(s[:2], np.array([0.0, y_tgt]), self.polys, self.kinds, self.N)

        wind = self._wind_profile(tgo)
        delta = 6.0 if self.use_hazards else 100.0
        best_sol = None
        iters = (self.n_scp if n_scp is None else n_scp) if self.use_hazards else 1
        for it in range(iters):
            self.log["n_solves"] += 1
            hs = self._halfspaces(pbar, tgo) if self.use_hazards else None
            sol = socp.solve(
                s[:4], tgo, s[6], wind=wind, halfspaces=hs,
                pbar=pbar, delta=delta if it > 0 else 10.0,
                t_cone=1.5, allow_off=not self.hover_possible,
            )
            if sol is None:
                break
            best_sol = sol
            if np.max(np.abs(sol.p - pbar)) < 0.05:
                break
            pbar = sol.p.copy()
            delta = 1.5

        if best_sol is None:
            return None
        if certify and not is_lossless(best_sol, 1e-3):
            return None
        return dict(kind="lcvx", staging=staging, tgo=tgo, dt=tgo / self.N, p=best_sol.p, v=best_sol.v,
                    a=best_sol.u, m=np.exp(best_sol.z), sol=best_sol)

    def _search(self, s, refine=5):
        """Fast 1-iter SCP on grid, then full SCP with margin."""
        for cert in (True, False):
            best = None

            def eval_tgo(t):
                nonlocal best
                plan = self._solve(s, float(t), certify=cert, n_scp=1)
                if plan is None:
                    return np.inf
                if best is None or plan["sol"].fuel < best["sol"].fuel:
                    best = plan
                return plan["sol"].fuel

            grid = self.tgo_grid
            vals = [eval_tgo(t) for t in grid]
            idx = int(np.argmin(vals))
            if not np.isfinite(vals[idx]):
                continue
            a = float(grid[max(0, idx - 1)])
            b = float(grid[min(len(grid) - 1, idx + 1)])
            if b > a:
                golden_section(eval_tgo, a, b, tol=0.0, max_iter=refine)
            if best is not None:
                t_final = best["tgo"] * self.tgo_margin
                refined = self._solve(s, t_final, certify=cert, pbar=best["p"], n_scp=self.n_scp)
                if refined is not None:
                    return refined
                refined = self._solve(s, best["tgo"], certify=cert, pbar=best["p"], n_scp=self.n_scp)
                return refined if refined is not None else best
        return None

    def _go_terminal(self, s):
        """Only enter terminal descent when laterally centered over the pad (or very low)."""
        centered = abs(s[0]) < 0.55 * self.pad_hw
        if s[1] < 0.35:
            return True
        short = self.plan is not None and self.plan["tgo"] - self.k_since / 50 < 0.6 and s[1] < 2.5 and centered
        return (s[1] < self.y_gate + 0.1 and centered) or short

    def _terminal(self, s):
        """Terminal vertical descent with lateral wind feed-forward and centering guard."""
        vy_ref = -self.v_gate if abs(s[0]) < 0.65 * self.pad_hw else 0.0
        v_ref = np.array([np.clip(-1.5 * s[0], -1.0, 1.0), vy_ref])
        w_est = float(known_wind(self.wind_spec, s[1]) + self.gust.xi_hat)
        a_cmd = np.array([-w_est, -self.g]) + np.array([3.0, 3.0]) * (v_ref - s[2:4])
        engine = self.a_min(s[6]) < -self.g or s[3] < vy_ref
        return self._to_action(s, a_cmd, engine=engine)

    def act(self, obs, info=None):
        if info is not None:
            self.gust.update(info["state"], self.a_prev)
        a = super().act(obs, info)
        self.a_prev = np.asarray(a, float).copy()
        return a


def make_policy(spec):
    return HazardCascadePolicy(spec)
