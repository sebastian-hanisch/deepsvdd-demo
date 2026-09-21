"""Local Outlier Factor (numpy von Grund auf, nach Breunig, Kriegel, Ng und Sander): eine Tour ist auffällig, wenn ihre lokale Dichte deutlich unter der ihrer k Nachbarn liegt.

k-Distanz kdist(a) = Abstand zum k-ten Nachbarn; Erreichbarkeitsdistanz reach(a, b) = max(kdist(b), d(a, b)); lokale Erreichbarkeitsdichte lrd(a) = 1 / Mittel der reach(a, b) über die k Nachbarn b;
LOF(a) = Mittel der lrd(b) der Nachbarn / lrd(a). Normale Touren liegen bei ~1, eine Tour in dünnerer Umgebung als ihre Nachbarn darüber.
Wie sklearn (und die Definition ohne Bindungserweiterung): genau k Nachbarn, bei Abstandsgleichheit entscheidet die Reihenfolge der Indizes; ein kleines epsilon im Nenner fängt Duplikate ab."""

from dataclasses import dataclass

import numpy as np

EPS = 1e-10


@dataclass(frozen=True)
class Fit:
    k: int
    lof: np.ndarray          # [n] Local Outlier Factor
    lrd: np.ndarray          # [n] lokale Erreichbarkeitsdichte
    kdist: np.ndarray        # [n] k-Distanz
    neighbors: np.ndarray    # [n, k] Indizes der k nächsten Nachbarn (aufsteigend nach Abstand)


def pairwise_distances(X):
    """Euklidische Abstände [n, n] (über die Gram-Matrix; auf 0 begrenzt, Diagonale exakt 0)."""
    sq = (X ** 2).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2)


def clamp_k(k, n):
    """Mindestens 1 und höchstens n - 1 Nachbarn."""
    return int(min(max(int(k), 1), n - 1))


def fit_lof(X, k, D=None):
    """LOF aller Zeilen von X (transduktiv: jede Tour wird gegen alle anderen bewertet). `D`: optional vorberechnete Abstandsmatrix."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = clamp_k(k, n)
    D = pairwise_distances(X) if D is None else D
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    order = np.argsort(Dm, axis=1, kind="stable")[:, :k]
    nd = np.take_along_axis(Dm, order, axis=1)                          # [n, k] Abstände zu den Nachbarn
    kdist = nd[:, -1]
    reach = np.maximum(kdist[order], nd)                                # reach(a, b) = max(kdist(b), d(a, b))
    lrd = 1.0 / (reach.mean(axis=1) + EPS)
    lof = lrd[order].mean(axis=1) / lrd
    return Fit(k, lof, lrd, kdist, order)



def score_new(X, fit, Q):
    """LOF für neue Punkte Q gegen die Trainingsmenge X (wie sklearn mit novelty=True): k nächste Nachbarn in X, reach(q, b) = max(kdist(b), d(q, b))."""
    X, Q = np.asarray(X, dtype=float), np.asarray(Q, dtype=float)
    sq_x, sq_q = (X ** 2).sum(axis=1), (Q ** 2).sum(axis=1)
    D2 = sq_q[:, None] + sq_x[None, :] - 2.0 * (Q @ X.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    order = np.argsort(D, axis=1, kind="stable")[:, : fit.k]
    nd = np.take_along_axis(D, order, axis=1)
    reach = np.maximum(fit.kdist[order], nd)
    lrd_q = 1.0 / (reach.mean(axis=1) + EPS)
    return fit.lrd[order].mean(axis=1) / lrd_q
