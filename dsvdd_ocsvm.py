"""One-Class SVM (numpy von Grund auf, nach Schölkopf, Platt, Shawe-Taylor, Smola und Williamson) mit RBF-Kernel.

Duales Problem in der Skalierung von libsvm: min 1/2 alpha^T K alpha  unter 0 <= alpha_i <= 1 und sum alpha_i = nu * n, mit K_ij = exp(-gamma * ||x_i - x_j||^2).
(Die Skalierung der Arbeit ist alpha' = alpha / (nu * n): 0 <= alpha'_i <= 1 / (nu n), sum = 1.)
Entscheidungsfunktion f(x) = sum_i alpha_i K(x_i, x) - rho: Punkte mit f < 0 liegen außerhalb der gelernten Grenze und gelten als Anomalien; der Score ist -f (größer = auffälliger).

Löser: SMO mit Auswahl des Paares nach zweiter Ordnung (wie libsvm): je Schritt ein Index i mit alpha_i < 1 (kann wachsen) und ein Index j mit alpha_j > 0 (kann fallen), der Gradient G = K alpha wird nachgeführt.
Anfangswert wie libsvm: floor(nu n) Koeffizienten gleich 1, einer mit dem Rest, alle anderen 0. rho aus den freien Stützvektoren (0 < alpha < 1), sonst aus den Grenzfällen.

Aussagen der Theorie, die die Tests prüfen: nu ist eine obere Schranke für den Anteil der Trainingspunkte mit f < 0 und eine untere Schranke für den Anteil der Stützvektoren (alpha > 0)."""

from dataclasses import dataclass

import numpy as np

import dsvdd_constants as C

TAU = 1e-12


@dataclass(frozen=True)
class Fit:
    X: np.ndarray            # Trainingsdaten [n, p]
    alpha: np.ndarray        # Koeffizienten (Skalierung von libsvm: 0 <= alpha <= 1, Summe nu * n), [n]
    rho: float
    gamma: float
    nu: float
    iterations: int
    converged: bool
    gap: float               # Restverletzung der KKT-Bedingungen (max G_low - min G_up), <= tol bei Konvergenz
    kernel: np.ndarray       # Kernmatrix [n, n] (gespeichert für die Darstellung und die Kosten: n^2 Zahlen)

    @property
    def n_support(self):
        return int((self.alpha > 1e-12).sum())

    @property
    def n_bound(self):
        """Stützvektoren an der oberen Schranke (alpha = 1): sie liegen außerhalb oder auf der Grenze."""
        return int((self.alpha >= 1.0 - 1e-12).sum())


def gamma_scale(X):
    """gamma_0 = 1 / (p * Var(X)) (die Voreinstellung 'scale' von scikit-learn); bei standardisierten Daten 1 / p."""
    X = np.asarray(X, dtype=float)
    return 1.0 / (X.shape[1] * max(float(X.var()), 1e-300))


def rbf_kernel(A, B, gamma):
    """K_ij = exp(-gamma * ||a_i - b_j||^2), [len(A), len(B)]."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    d2 = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def fit_ocsvm(X, nu, gamma, tol=C.SMO_TOL, max_iter=C.SMO_MAX_ITER):
    """Trainiert die One-Class SVM; `gamma` ist der absolute Wert (nicht der Faktor)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if not 0 < nu <= 1:
        raise ValueError("nu muss in (0, 1] liegen")
    K = rbf_kernel(X, X, gamma)
    total = nu * n
    alpha = np.zeros(n)
    n_full = int(total)
    alpha[:n_full] = 1.0
    if n_full < n:
        alpha[n_full] = total - n_full
    G = K @ alpha
    QD = np.diag(K).copy()
    it = 0
    converged = False
    gap = np.inf
    while it < max_iter:
        up = alpha < 1.0 - 1e-15
        low = alpha > 1e-15
        if not up.any() or not low.any():
            gap, converged = 0.0, True
            break
        g_up = np.where(up, G, np.inf)
        i = int(np.argmin(g_up))
        gap = float(np.max(np.where(low, G, -np.inf)) - g_up[i])
        if gap < tol:
            converged = True
            break
        # zweites Glied nach zweiter Ordnung: unter den fallbaren j mit G_j > G_i der größte erwartete Zuwachs
        diff = G - G[i]
        quad = np.maximum(QD[i] + QD - 2.0 * K[i], TAU)
        gain = np.where(low & (diff > 0), diff ** 2 / quad, -np.inf)
        j = int(np.argmax(gain))
        d = min(diff[j] / quad[j], 1.0 - alpha[i], alpha[j])
        alpha[i] += d
        alpha[j] -= d
        G += d * (K[:, i] - K[:, j])
        it += 1
    free = (alpha > 1e-12) & (alpha < 1.0 - 1e-12)
    if free.any():
        rho = float(G[free].mean())
    else:
        ub = G[alpha <= 1e-12].min() if (alpha <= 1e-12).any() else np.inf                # untere Schranke von alpha: G bildet die obere Grenze von rho
        lb = G[alpha >= 1.0 - 1e-12].max() if (alpha >= 1.0 - 1e-12).any() else -np.inf
        rho = float((ub + lb) / 2.0) if np.isfinite(ub) and np.isfinite(lb) else float(ub if np.isfinite(ub) else lb)
    return Fit(X, alpha, rho, float(gamma), float(nu), it, converged, float(gap), K)


def decision_function(fit, Z):
    """f(z) = sum_i alpha_i K(x_i, z) - rho für neue Punkte Z [m, p]; f < 0 = außerhalb der Grenze."""
    Z = np.asarray(Z, dtype=float)
    sv = fit.alpha > 1e-12
    return rbf_kernel(Z, fit.X[sv], fit.gamma) @ fit.alpha[sv] - fit.rho


def train_decision(fit):
    """f an den Trainingspunkten (aus der gespeicherten Kernmatrix, ohne Neuberechnung)."""
    return fit.kernel @ fit.alpha - fit.rho


def score(fit, Z=None):
    """Anomalie-Wert = -f (größer = auffälliger)."""
    return -(train_decision(fit) if Z is None else decision_function(fit, Z))
