# Guide rapide : Pipeline de la Phase 2 (Benchmark B2)

Ce guide explique l'architecture de la Phase 2 (corps rigide avec rotation) et comment exécuter, évaluer et visualiser ton contrôleur.

---

## 1. Architecture de la pipeline

La Phase 2 relie ton planificateur SOCP (`lander/lcvx.py`) au simulateur corps rigide Box2D (`lander/env.py`) via le contrôleur en cascade de `lander/controllers.py` :

1. **`lander/lcvx.py` (`LanderSOCP`)** : planifie une trajectoire jusqu'à la *porte d'approche* (`terminal="gate"`), située juste au-dessus du pad (`y_gate`, `v_gate`).
2. **`lander/controllers.py` (`CascadeController`)** :
   - **Couche haute (toutes les 0.5 s / `replan_every=25` pas)** : replanifie depuis l'état mesuré `s` (`_search` au début, puis `_replan` en horizon glissant).
   - **Couche intermédiaire (tous les pas, 50 Hz)** : loi de suivi PD (`_track`) qui corrige les écarts par rapport au plan.
   - **Couche basse (fournie, 50 Hz)** : boucle d'attitude (`_attitude`) + modulateur sigma-delta (`_throttle`) pour traduire `a_cmd` en commandes moteur `(a0, a1)`.
3. **`lander/evaluate.py`** : exécute une politique (`make_policy(spec)`) sur les 5 conditions de **Benchmark B2** (`nominal`, `heavy`, `light`, `gusty`, `storm`), agrège les scores et génère des GIFs.

---

## 2. Méthodes à compléter (Question 4)

Dans `lander/controllers.py`, trois méthodes de `CascadeController` sont marquées `# TODO (Question 4)` :
* `_search(self, s, refine=6)` : recherche grossière sur `self.tgo_grid`, puis `refine` étapes de *golden section* avec `self._solve(s, t, certify=True)`, puis ré-optimisation avec `self.tgo_margin * best_tgo`.
* `_replan(self, s)` : si un plan existe avec un temps restant $> 0.6\text{ s}$, tente `self._solve(s, f * tgo, certify=False)` pour `f = 1.0, 1.15, 1.3`, sinon relance `self._search(s)`.
* `_track(self, s)` : calcule `a_cmd = a + self.kp_trk * (p_ref - s[:2]) + self.kd_trk * (v_ref - s[2:4])`.

---

## 3. Commandes pour lancer et évaluer

Depuis la racine `optimal_control_for_spacecraft_landing/`, avec l'environnement `mgt-convex-project` :

### Tester rapidement sur 10 seeds (avant de lancer les 100)
```bash
./mgt-convex-project/bin/python -m lander.evaluate \
    --policy lander.controllers:make_cascade \
    --benchmark b2 --n 10 --processes 4
```

### Évaluation complète Benchmark B2 (100 seeds × 5 conditions = 500 épisodes)
```bash
./mgt-convex-project/bin/python -m lander.evaluate \
    --policy lander.controllers:make_cascade \
    --benchmark b2 --n 100 --processes 4 --out results_b2_cascade.json
```

### Comparer avec les deux baselines fournies
```bash
# Baseline heuristique
./mgt-convex-project/bin/python -m lander.evaluate \
    --policy submissions/baseline_heuristic.py \
    --benchmark b2 --n 100 --processes 4 --out results_b2_heuristic.json

# Baseline PPO (RL)
./mgt-convex-project/bin/python -m lander.evaluate \
    --policy submissions/baseline_ppo.py \
    --benchmark b2 --n 100 --processes 4 --out results_b2_ppo.json
```

### Cibler une seule condition (ex. `light` ou `storm`)
```bash
./mgt-convex-project/bin/python -m lander.evaluate \
    --policy lander.controllers:make_cascade \
    --benchmark b2 --scenarios light,storm --n 20
```

---

## 4. Visualiser un vol en GIF

L'option `--gif` enregistre un épisode par condition à vitesse réelle dans `gifs/` (sans évaluer le benchmark) :

```bash
./mgt-convex-project/bin/python -m lander.evaluate \
    --policy lander.controllers:make_cascade \
    --benchmark b2 --scenarios nominal,light,storm --seed 0 --gif
```

---

## 5. Lire la sortie de `lander.evaluate`

Pour chaque condition, le script affiche :
* `success` : taux d'atterrissages réussis (sur les 2 jambes, dans le pad, vitesse $\le 2\text{ m/s}$).
* `crash` / `off_pad` / `out` / `timeout` : taux d'échecs par catégorie.
* `fuel` (`fuel_success`) : carburant moyen consommé (en kg) sur les essais réussis.
* `ms` (`policy_ms`) : temps de calcul moyen par pas de simulation (en ms).
