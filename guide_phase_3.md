# Guide Phase 3 : Architecture robuste et évitement d'obstacles (Question 5)

Ce document explique la logique mathématique et algorithmique implémentée dans `submissions/submission.py` (`HazardCascadePolicy`).

---

## 1. Quel est le problème ?

En Phase 2 (`CascadeController`), le lander évoluait sur un terrain plat sans obstacles. En **Phase 3**, il doit atterrir sur **10 scénarios** (`S01`–`S10`) avec trois nouvelles difficultés majeures :

1. **Obstacles polygonaux (non convexes) :** rochers solides et zones interdites (*no-fly zones*) que la boîte englobante du lander ($R_X = 1.0\text{ m}$, $R_Y = 1.15\text{ m}$) ne doit jamais toucher (**pénalité de $-50$ points**). L'extérieur d'une union de polygones est un ensemble **non convexe**.
2. **Vent complexe et incertain :**
   - **Connu :** vent moyen $\bar{w}$ + cisaillement linéaire en altitude $\frac{s}{10}(y + \ell)$.
   - **Inconnu :** rafales stochastiques $\xi_k$ (processus d'Ornstein–Uhlenbeck de variance $\sigma_\xi^2$ et temps de corrélation $\tau_g$) et couple de turbulence aléatoire.
3. **Contraintes opérationnelles strictes :** optimisation convexe / modèle uniquement (RL interdit) et temps de calcul moyen **$< 50\text{ ms/pas}$**.

---

## 2. Comment on procède (`submissions/submission.py`)

Nous étendons le `CascadeController` de la Phase 2 avec cinq briques complémentaires :

### A. Modélisation et estimation du vent en ligne
* Le vent connu $\bar{w}$ et le cisaillement $\frac{s}{10}y$ (affine en $y$) sont intégrés directement dans la dynamique du `LanderSOCP`.
* À chaque pas (50 Hz), `GustEstimator` (`lander/wind.py`) estime la rafale courante $\hat{\xi}_0$ par filtrage de l'accélération latérale résiduelle, et prédit sa décroissance attendue $\mathbb{E}[\xi_k] = \hat{\xi}_0 e^{-t_k / \tau_g}$ sur l'horizon du SOCP.

### B. Évitement d'obstacles par *Sequential Convex Programming* (SCP)
1. **Initialisation hors obstacle (`_initial_path`) :** si la ligne droite entre l'état courant et la cible traverse un polygone gonflé, on construit une ligne brisée contournant l'obstacle (au-dessus ou sur les côtés) pour démarrer dans la **bonne classe d'homotopie**.
2. **Linéarisation par demi-espaces (`_halfspaces`) :** autour de la trajectoire courante $\bar{r}_k$, on calcule le point le plus proche $q_{j,k}$ et la normale sortante $n_{j,k}$ de chaque polygone $\mathcal{P}_j$, puis on impose le demi-espace séparateur :
   $$n_{j,k}^\top r_k \ge n_{j,k}^\top q_{j,k} + \underbrace{R_X|n_{j,k,x}| + R_Y|n_{j,k,y}|}_{\text{boîte du lander}} + m_{j,k} - \nu_{j,k}$$
   avec une *trust region* $|r_k - \bar{r}_k| \le \Delta_{\text{tr}}$ et des variables de *slack* $\nu \ge 0$ fortement pénalisées ($w_{\text{slack}} = 10^3$) pour éviter toute infaisabilité numérique.

### C. Robustification analytique (*Chance Constraints* OU)
L'erreur de prédiction de la rafale OU à l'horizon $t_k$ a une variance $\sigma_\xi^2(1 - e^{-2t_k/\tau_g})$, ce qui induit un écart-type sur la position latérale $\sigma_{\text{pos},x}(t_k)$. On ajoute donc une marge de sécurité :
$$m_{j,k} = m_{\text{trk}} + c_\tau \sigma_{\text{torque}} + z_\alpha |n_{j,k,x}|\,\sigma_{\text{pos},x}(t_k), \qquad z_\alpha = 1.645\;(95\%).$$

### D. Porte de *staging* pour les canyons étroits (`S05`)
Lorsqu'un canyon étroit flanque le pad des deux côtés, le contrôleur cible d'abord une **porte intermédiaire** au-dessus de l'entrée du canyon (`y_stage`, $x \approx 0$, $v_x \approx 0$) avant de descendre verticalement dans le couloir.

### E. Garde de centrage et compensation du vent en phase terminale
* `_go_terminal` bloque le passage en descente terminale tant que le lander n'est pas centré au-dessus du pad ($|x| < 0.55\,\text{pad\_hw}$).
* `_terminal` injecte le terme de *feed-forward* $-\big(\bar{w} + \frac{s}{10}(y+\ell) + \hat{\xi}_0\big)$ pour éviter toute dérive latérale au ras du sol.

---

## 3. Pourquoi c'est censé marcher ?

1. **Chaque itération est un SOCP pur et DPP :** les contraintes restent linéaires ou cônes du second ordre ; le problème est compilé une seule fois à l'initialisation, ce qui maintient le temps de calcul autour de **$5\text{ ms/pas}$** ($\ll 50\text{ ms}$).
2. **Pas de blocage dans un minimum local :** l'initialisation par contournement + la *trust region* assurent que le SCP converge vers une trajectoire physiquement faisable autour des grands obstacles (flèches rocheuses `S04`, panaches `S06`).
3. **Sûreté en boucle fermée :**
   - La marge $m_{j,k}$ empêche la trajectoire nominale de raser les parois sous l'effet des rafales imprévues et des retards d'attitude.
   - La replanification périodique (0.3–0.5 s) + le suivi PD à 50 Hz corrigent en continu les écarts entre le modèle point-masse et le corps rigide.
   - Le *feed-forward* de vent en phase terminale garantit un posé centré sur le pad même par fort vent de travers (`S02`, `S03`).
