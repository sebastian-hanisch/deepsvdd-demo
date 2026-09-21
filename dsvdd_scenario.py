"""Erzeugt Lieferrouten-Kennzahlen mit bekannten Anomalien (Sonderfahrten).

Wie in der PCA-Demo: zwei versteckte Faktoren z, jedes der 12 Merkmale lädt auf den Faktor seiner Gruppe (Gruppe g -> Faktor g mod 2) plus kleine Querladungen, dazu Rauschen.
Neu gegenüber der PCA-Demo:
- **Betriebsarten** (1-3): die normalen Touren stammen aus einer, zwei oder drei Gruppen im Faktorraum (Stadt, Land, Fernverkehr) - bei einer Betriebsart ist der Normalbereich ellipsenförmig.
- **Krümmung**: die normale Fläche wird gebogen (die nichtlinearen Terme der PCA-Demo), der Normalbereich ist dann nicht mehr konvex.
- **Anomalien** (Anteil bis 45 %, Abstand im Faktorraum wählbar) in drei Arten: verstreut (jede in eine andere Richtung), als dichte Gruppe abseits (maskiert die klassische Schätzung)
  oder in der Lücke zwischen den Betriebsarten (dort, wo eine einzelne Ellipse alles für normal hält).
- **Rauschmerkmale** (0-40 unabhängige Spalten ohne Zusammenhang mit den Faktoren, angehängt; die bisherigen Spalten bleiben gleich) und **Merkmalszahl p** bis 30 (Merkmale 13 ... p sind feste Mischungen der Faktoren plus eigenes Rauschen), damit auch der Fall n < p vorkommt.
Die Ziehungsreihenfolge (z, eps, ...) und die Matrizen sind die der PCA-Demo: die normalen Touren sind bei einer Betriebsart und ohne Krümmung bei p = 12 dieselben Zeilen wie dort (Spalten umsortiert);
der Anomalie-Anteil ist exakt (mindestens eine Anomalie), die Anomalien sind die Touren mit den kleinsten Losen."""

from dataclasses import dataclass

import numpy as np

import dsvdd_constants as C

EXTRA_MAX = C.P_MAX - C.N_BASE_FEATURES
NOISE_MAX = C.N_NOISE_MAX


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray              # [n, p] Rohdaten in Einheiten
    z: np.ndarray              # [n, 2] latente Faktoren (Wahrheit, mit Betriebsart und Anomalie)
    anomaly: np.ndarray        # [n] bool
    mode: np.ndarray           # [n] Betriebsart 0 ... n_modes - 1 (bei Anomalien die der Normalziehung, ohne Bedeutung)
    n_modes: int
    kind: str
    p: int
    n_noise: int = 0           # angehängte reine Rauschmerkmale

    @property
    def n(self):
        return len(self.X)

    @property
    def names(self):
        base = [C.FEATURES[j][0] for j in C.FEATURE_ORDER[: min(self.p, C.N_BASE_FEATURES)]]
        names = base + [f"Zusatzmerkmal {k}" for k in range(C.N_BASE_FEATURES + 1, self.p + 1)]
        return names + [f"Rauschmerkmal {k}" for k in range(1, self.n_noise + 1)]


def _nonlinear_terms(z):
    n, q = z.shape
    terms = []
    for f in range(q):
        terms.append(np.sin(C.CURVATURE_FREQUENCY * z[:, f]))
        terms.append(np.cos(C.CURVATURE_FREQUENCY * z[:, f]) - np.exp(-0.5 * C.CURVATURE_FREQUENCY ** 2))
        terms.append(z[:, f] ** 2 - 1.0)
    for a in range(q):
        for b in range(a + 1, q):
            terms.append(z[:, a] * z[:, b])
    return np.stack(terms, axis=1)


def _n_terms(q):
    return 3 * q + q * (q - 1) // 2


def loading_matrix():
    """Feste 12 x 2-Ladungsmatrix (die der PCA-Demo mit q = 2)."""
    rng = np.random.default_rng(C.LAYOUT_SEED)
    cross = rng.uniform(-1.0, 1.0, size=(C.N_BASE_FEATURES, 4))
    L = np.zeros((C.N_BASE_FEATURES, C.Q))
    for j in range(C.N_BASE_FEATURES):
        group = C.GROUP_OF_FEATURE[j]
        L[j, group % C.Q] += C.WITHIN_LOADINGS[j % 3]
        for g in range(4):
            if g != group:
                L[j, g % C.Q] += C.CROSS_LOADING * cross[j, g]
    return L


def curvature_matrix():
    rng = np.random.default_rng(C.LAYOUT_SEED + 1)
    B = rng.standard_normal((C.N_BASE_FEATURES, _n_terms(C.Q)))
    return B / np.linalg.norm(B, axis=0, keepdims=True) * C.CURVATURE_AMPLITUDE


def extra_loading_matrix():
    """Feste Mischungen der Faktoren für die Zusatzmerkmale 13 ... 30 (Zeilen mit Länge 0.9)."""
    rng = np.random.default_rng(C.EXTRA_LAYOUT_SEED)
    E = rng.standard_normal((EXTRA_MAX, C.Q))
    return 0.9 * E / np.linalg.norm(E, axis=1, keepdims=True)


def mode_centers(n_modes):
    """Mittelpunkte der Betriebsarten im Faktorraum: eine im Ursprung, zwei gegenüber, drei im Dreieck (Radius `MODE_RADIUS`)."""
    if n_modes == 1:
        return np.zeros((1, C.Q))
    angles = 2 * np.pi * np.arange(n_modes) / n_modes
    return C.MODE_RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def generate_dataset(n_tours, p, n_modes, curvature, noise, contamination_pct, kind, strength, seed, n_noise=0):
    """`kind`: "scattered", "cluster", "gap" (braucht mindestens zwei Betriebsarten; bei einer fällt sie auf "scattered" zurück) oder "decorrelated" (Korrelationsbruch: Randverteilungen der Normalen, Abhängigkeit zerstört)."""
    if kind == "gap" and n_modes < 2:
        kind = "scattered"
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_tours, C.Q))
    eps = rng.standard_normal((n_tours, C.N_BASE_FEATURES))
    eps_extra = rng.standard_normal((n_tours, EXTRA_MAX))
    mode_draw = rng.random(n_tours)
    anomaly_draw = rng.random(n_tours)
    anomaly_noise = rng.standard_normal((n_tours, C.Q))
    anomaly_direction = rng.standard_normal((n_tours, C.Q))
    noise_columns = rng.standard_normal((n_tours, NOISE_MAX))                             # zuletzt gezogen: die bisherigen Spalten bleiben unverändert
    pick_draw = rng.random((n_tours, C.P_MAX + NOISE_MAX))                                # noch später: Zeilenwahl für den Korrelationsbruch (ändert keine bisherige Ziehung)
    mode = np.minimum((mode_draw * n_modes).astype(int), n_modes - 1)
    z_used = z.copy() if n_modes == 1 else mode_centers(n_modes)[mode] + C.MODE_SD * z          # eine Betriebsart: N(0, I) wie in der PCA-Demo
    n_anomalies = max(1, int(round(contamination_pct / 100.0 * n_tours)))                 # genau dieser Anteil (mindestens eine Anomalie): die mit den kleinsten Losen
    anomaly = np.zeros(n_tours, dtype=bool)
    anomaly[np.argsort(anomaly_draw, kind="stable")[:n_anomalies]] = True
    if kind == "scattered":
        direction = anomaly_direction / np.linalg.norm(anomaly_direction, axis=1, keepdims=True)
        radius = strength * (1.0 + 0.2 * anomaly_noise[:, :1])
        z_used[anomaly] = (direction * radius)[anomaly]
    elif kind == "cluster":
        u = np.array([np.cos(C.CLUSTER_ANGLE), np.sin(C.CLUSTER_ANGLE)])
        z_used[anomaly] = (strength * u + C.CLUSTER_SD * anomaly_noise)[anomaly]
    elif kind == "gap":
        z_used[anomaly] = (C.GAP_SD * anomaly_noise)[anomaly]
    signal = z_used @ loading_matrix().T
    if curvature > 0:
        signal = signal + curvature * _nonlinear_terms(z_used) @ curvature_matrix().T
    signal = signal + noise * eps
    means = np.array([f[2] for f in C.FEATURES])
    scales = np.array([f[3] for f in C.FEATURES])
    X = (means + signal * scales)[:, list(C.FEATURE_ORDER)]
    if p > C.N_BASE_FEATURES:
        k = p - C.N_BASE_FEATURES
        extra = z_used @ extra_loading_matrix()[:k].T + noise * eps_extra[:, :k]
        X = np.concatenate([X, C.EXTRA_MEAN + C.EXTRA_SCALE * extra], axis=1)
    else:
        X = X[:, :p]
    if n_noise > 0:
        X = np.concatenate([X, C.EXTRA_MEAN + C.EXTRA_SCALE * noise_columns[:, :n_noise]], axis=1)
    if kind == "decorrelated":
        # Korrelationsbruch: die Merkmale der Anomalien werden je Spalte unabhängig aus den Werten der normalen Touren gezogen - identische Randverteilungen, zerstörte Abhängigkeit
        normal_rows = np.flatnonzero(~anomaly)
        pick = normal_rows[np.minimum((pick_draw[anomaly][:, : X.shape[1]] * len(normal_rows)).astype(int), len(normal_rows) - 1)]
        X = X.copy()
        X[anomaly] = X[pick, np.arange(X.shape[1])[None, :]]
    return Dataset(X=X, z=z_used, anomaly=anomaly, mode=mode, n_modes=int(n_modes), kind=kind, p=int(p), n_noise=int(n_noise))
