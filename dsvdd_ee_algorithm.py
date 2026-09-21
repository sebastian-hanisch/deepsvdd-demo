"""Elliptic Envelope (numpy von Grund auf): klassische Schätzung von Mittelwert und Kovarianz, robuste Schätzung per Minimum Covariance Determinant (FastMCD nach Rousseeuw und Van Driessen:
C-Schritte aus vielen Zufallsstarts, Konsistenz-Korrektur, optional Neugewichtung), quadrierter Mahalanobis-Abstand als Anomalie-Wert, chi-Quadrat-Quantil als Schwelle."""

import math
from dataclasses import dataclass

import numpy as np

import dsvdd_constants as C

EIG_TOL = 1e-12


# --- chi-Quadrat-Verteilung (ohne scipy) --------------------------------------------------------------------------------------


def _gammainc(a, x):
    """Regularisierte untere unvollständige Gammafunktion P(a, x): Reihe für x < a + 1, sonst Kettenbruch (Lentz)."""
    if x <= 0.0:
        return 0.0
    lg = math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - lg)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - lg) * h


def chi2_cdf(x, dof):
    return _gammainc(dof / 2.0, x / 2.0)


def chi2_ppf(q, dof):
    """Quantil der chi-Quadrat-Verteilung mit `dof` Freiheitsgraden (Bisektion auf der Verteilungsfunktion)."""
    lo, hi = 0.0, max(10.0, 4.0 * dof)
    while chi2_cdf(hi, dof) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, dof) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# --- Schätzer -------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    method: str                    # "classical", "mcd", "mcd-reweighted"
    location: np.ndarray
    covariance: np.ndarray
    d2: np.ndarray                 # quadrierter Mahalanobis-Abstand aller Punkte
    dof: int                       # Rang der Kovarianz (= p, wenn n >= p + 1 und die Merkmale nicht ausgeartet sind)
    h: int = 0                     # Größe der MCD-Teilmenge
    support: np.ndarray = None     # Indizes der MCD-Teilmenge des besten Starts
    logdet_history: tuple = ()     # log det der Kovarianz je C-Schritt des besten Starts
    subsets: tuple = ()            # h-Teilmenge je C-Schritt des besten Starts (Schritt-Ansicht)
    raw_location: np.ndarray = None
    raw_covariance: np.ndarray = None
    n_converged: int = 0           # Starts, die konvergiert sind


def _decompose(cov):
    """Eigenzerlegung der Korrelationsmatrix (dadurch unabhängig von den Einheiten der Merkmale) mit Abschneiden der numerisch verschwindenden Eigenwerte:
    (Streuungen, Eigenwerte, Eigenvektoren, Rang)."""
    s = np.sqrt(np.maximum(np.diag(cov), 0.0))
    s = np.where(s > 0, s, 1.0)
    w, V = np.linalg.eigh(cov / np.outer(s, s))
    keep = w > EIG_TOL * max(float(w.max()), 1e-300)
    return s, w[keep], V[:, keep], int(keep.sum())


def mahalanobis_sq(X, location, covariance):
    s, w, V, _ = _decompose(covariance)
    proj = ((X - location) / s) @ V
    return (proj ** 2 / w).sum(axis=1)


def _logdet(cov, ridge=C.RIDGE):
    """log det der Kovarianz (Korrelationsmatrix mit kleiner Regularisierung plus die Streuungen): unabhängig von den Einheiten."""
    s = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(s, s)
    return float(np.linalg.slogdet(corr + ridge * np.eye(len(cov)))[1] + 2.0 * np.log(s).sum())


def _mean_cov(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    return mu, Xc.T @ Xc / max(len(X) - 1, 1)


def fit_classical(X):
    mu, cov = _mean_cov(X)
    return Fit("classical", mu, cov, mahalanobis_sq(X, mu, cov), _decompose(cov)[3])


def subset_size(n, p, support_fraction):
    """h: mindestens (n + p + 1) / 2 (größter Bruchpunkt), höchstens n."""
    return int(min(n, max(math.ceil(support_fraction * n), (n + p + 1) // 2)))


def _c_step(X, mu, cov, h):
    d2 = mahalanobis_sq(X, mu, cov)
    sel = np.sort(np.argpartition(d2, h - 1)[:h])
    mu, cov = _mean_cov(X[sel])
    return mu, cov, sel, _logdet(cov)


def fit_mcd(X, support_fraction=C.DEFAULT_SUPPORT, seed=0, n_starts=C.MCD_STARTS, reweight=True, reweight_quantile=0.975):
    """Minimum Covariance Determinant: die h Punkte, deren Kovarianzmatrix die kleinste Determinante hat. FastMCD (Rousseeuw und Van Driessen): viele zufällige Starts aus p + 1 Punkten mit je zwei C-Schritten
    (Abstände mit der aktuellen Schätzung, die h kleinsten wählen, neu schätzen), die besten `MCD_KEEP` laufen bis die Determinante nicht mehr sinkt; die beste Lösung zählt. Anschließend Konsistenz-Korrektur
    (Median der Abstände gegen den chi-Quadrat-Median) und optional eine Neugewichtung (Punkte innerhalb des Schwellenquantils, mit Korrektur der abgeschnittenen Verteilung)."""
    n, p = X.shape
    h = subset_size(n, p, support_fraction)
    rng = np.random.default_rng([seed, 777])
    starts = []
    for _ in range(n_starts):
        idx = rng.choice(n, size=min(p + 1, n), replace=False)
        mu, cov = _mean_cov(X[idx])
        history, subsets = [], []
        for _step in range(C.MCD_INITIAL_STEPS):
            mu, cov, sel, ld = _c_step(X, mu, cov, h)
            history.append(ld)
            subsets.append(sel)
        starts.append((ld, mu, cov, history, subsets))
    starts.sort(key=lambda t: t[0])
    best = None
    converged = 0
    for _ld, mu, cov, history, subsets in starts[: C.MCD_KEEP]:
        done = False
        for _step in range(C.MCD_MAX_STEPS):
            prev = history[-1]
            mu, cov, sel, ld = _c_step(X, mu, cov, h)
            history.append(ld)
            subsets.append(sel)
            if ld >= prev - 1e-10 * max(abs(prev), 1.0):
                done = True
                break
        converged += done
        if best is None or history[-1] < best[0]:
            best = (history[-1], mu, cov, subsets[-1], tuple(history), tuple(subsets))
    _, mu, cov, sel, history, subsets = best
    d2 = mahalanobis_sq(X, mu, cov)
    dof = _decompose(cov)[3]
    cov = cov * float(np.median(d2)) / chi2_ppf(0.5, dof)                           # Konsistenz-Korrektur (Gauß'sche Daten)
    raw_mu, raw_cov = mu, cov
    d2 = mahalanobis_sq(X, mu, cov)
    method = "mcd"
    if reweight:
        q = chi2_ppf(reweight_quantile, dof)
        keep = d2 < q
        if keep.sum() > p:
            mu, cov = _mean_cov(X[keep])
            cov = cov * chi2_cdf(q, dof) / chi2_cdf(q, dof + 2)                       # Abschneiden der Ränder korrigieren
            d2 = mahalanobis_sq(X, mu, cov)
            method = "mcd-reweighted"
    return Fit(method, mu, cov, d2, dof, h, sel, history, subsets, raw_mu, raw_cov, converged)


def threshold(fit, quantile):
    """Schwelle für d^2: das chi-Quadrat-Quantil (Gauß'sche Normaldaten haben d^2 ~ chi^2 mit p bzw. Rang-vielen Freiheitsgraden)."""
    return chi2_ppf(quantile, fit.dof)


# --- Darstellung ------------------------------------------------------------------------------------------------------------------


def projection_axes(covariance):
    """Zwei Achsen (p x 2) der größten Streuung der Kovarianz - die Ebene, in der die Ellipse gezeichnet wird (bei p = 2 die Koordinatenachsen der Eigenvektoren)."""
    w, V = np.linalg.eigh(covariance)
    order = np.argsort(w)[::-1][:2]
    return V[:, order]


def ellipse_points(location, covariance, level, axes, n_points=100):
    """Rand der auf die Ebene `axes` projizierten Ellipse {x: (x - mu)' Sigma^-1 (x - mu) <= level} (Projektion eines Ellipsoids = Ellipse mit der projizierten Kovarianz)."""
    Sp = axes.T @ covariance @ axes
    w, V = np.linalg.eigh(Sp)
    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(t), np.sin(t)])
    pts = V @ (np.sqrt(np.maximum(w, 0.0) * level)[:, None] * circle)
    center = axes.T @ location
    return (pts + center[:, None]).T
