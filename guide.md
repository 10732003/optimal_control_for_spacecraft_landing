# Guide rapide : Exécuter le SOCP (Phase 1)

Ce guide explique pas à pas comment résoudre le problème de guidage par convexification sans perte (*Lossless Convexification*) implémenté dans `lander/lcvx.py`.

---

## 1. Environnement Python

Depuis la racine du projet `optimal_control_for_spacecraft_landing/`, utilise l'interpréteur Python de ton environnement `uv` (qui contient `cvxpy`, `clarabel`, `numpy`, `scipy`, `matplotlib`) :

```bash
PYTHON=./mgt-convex-project/bin/python
```

---

## 2. Les 4 étapes pour résoudre et simuler une trajectoire

### Étape 1 — Importer et construire le problème SOCP (compilé une seule fois)

```python
import numpy as np
from lander.params import P
from lander.lcvx import LanderSOCP, solve_min_fuel, is_lossless
from lander.pointmass import simulate

# Construit le SOCP une seule fois (respecte les règles DPP)
prob = LanderSOCP(P, N=40, g=-10.0, mode="fuel", terminal="ground")
```

### Étape 2 — Définir l'état initial `x0`

`x0` est un vecteur 1D de taille 4 : `(x, y, vx, vy)` (en mètres et m/s).
Pour le **Cas A** :

```python
x0 = np.array([-6.0, 10.0, -2.0, -1.0])
```

### Étape 3 — Résoudre

Tu as deux options selon ton besoin :

* **Option A : temps de vol $t_f$ fixé** (ex. `tgo = 3.0` s) :
  ```python
  sol = prob.solve(x0, tgo=3.0)
  ```
* **Option B : recherche automatique du temps de vol optimal $t_f^\star$** (grille grossière + *golden section*, recommandée) :
  ```python
  # certify=True rejette les solutions où la relaxation n'est pas exacte
  sol, evals = solve_min_fuel(prob, x0, n_grid=25, tol=0.02, certify=True)
  ```

### Étape 4 — Lire et vérifier la solution (`LanderSolution`)

```python
if sol is not None:
    print(f"Statut         : {sol.status}")
    print(f"Temps optimal  : {sol.tgo:.4f} s")
    print(f"Carburant      : {sol.fuel * 1000:.2f} g")
    print(f"Certifié exact : {is_lossless(sol)}")
    print(f"Écart max gap  : {np.max(sol.relaxation_gap):.2e}")
```

Principaux attributs utiles de `sol` :
* `sol.t` : grille temporelle `(N+1,)`
* `sol.p` : positions `(N+1, 2)` $(x_k, y_k)$
* `sol.v` : vitesses `(N+1, 2)` $(v_{x,k}, v_{y,k})$
* `sol.thrust` : vecteurs de poussée `(N, 2)` $T_k$ en Newtons
* `sol.m` : masse `(N+1,)` en kg
* `sol.relaxation_gap` : $\sigma_k - \|u_k\|_2$ à chaque pas $k$

### Étape 5 — Simuler en boucle ouverte (`simulate`)

```python
sim_accel = simulate(sol, p=P, g=-10.0, mode="accel")
sim_thrust = simulate(sol, p=P, g=-10.0, mode="thrust")

print("Erreur pos (accel) :", sim_accel["final_pos_err"], "m")
print("Erreur pos (thrust):", sim_thrust["final_pos_err"], "m")
```

---

## 3. Commande prête à copier-coller (Cas A)

```bash
cd /Users/adriendubois/Desktop/4A/4A_scolaire/MA1_cours/MGT418-convex-optim/projects/project3--partage/optimal_control_for_spacecraft_landing

./mgt-convex-project/bin/python -c "
import numpy as np
from lander.params import P
from lander.lcvx import LanderSOCP, solve_min_fuel, is_lossless
from lander.pointmass import simulate

prob = LanderSOCP(P, N=40, g=-10.0, mode='fuel', terminal='ground')
x0 = np.array([-6.0, 10.0, -2.0, -1.0])
sol, _ = solve_min_fuel(prob, x0)
sim = simulate(sol, p=P, g=-10.0, mode='accel')

print(f't_f* = {sol.tgo:.4f} s | fuel = {sol.fuel*1000:.2f} g | lossless = {is_lossless(sol)}')
print(f'sim pos_err = {sim[\"final_pos_err\"]:.2e} m | vel_err = {sim[\"final_vel_err\"]:.2e} m/s')
"
```
