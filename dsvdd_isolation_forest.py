"""Isolation Forest (numpy von Grund auf, nach Liu, Ting und Zhou): Wald aus zufälligen Isolationsbäumen. Jeder Baum wird auf einer kleinen Unterstichprobe gebaut, indem rekursiv ein zufälliges Merkmal
und ein gleichverteilter Schnittpunkt zwischen Minimum und Maximum des Knotens gewählt wird, bis jeder Punkt allein ist (oder das Höhenlimit erreicht ist). Anomalien liegen abseits und werden nach wenigen Schnitten
isoliert - kurze Pfade; der Anomalie-Wert ist 2^(-E[Pfadlänge] / c(psi))."""

import math
from dataclasses import dataclass

import numpy as np

EULER_GAMMA = 0.5772156649015329


def c_factor(n):
    """Mittlere Pfadlänge einer erfolglosen Suche im binären Suchbaum über n Punkte: c(n) = 2 H(n-1) - 2 (n-1) / n (H harmonisch, ln + Euler-Konstante); c(1) = 0, c(2) = 1. Normiert die Pfadlängen."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + EULER_GAMMA) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


@dataclass(frozen=True)
class Tree:
    feature: np.ndarray         # Merkmal des Schnitts (-1 = Blatt)
    threshold: np.ndarray       # Schnittpunkt: kleiner geht nach links
    left: np.ndarray
    right: np.ndarray
    size: np.ndarray            # Zahl der Unterstichproben-Punkte im Knoten
    depth: np.ndarray           # Tiefe des Knotens


@dataclass(frozen=True)
class Forest:
    trees: tuple
    psi: int                    # Größe der Unterstichprobe
    subsets: tuple              # Indizes der Unterstichprobe je Baum
    max_depth: int


def height_limit(psi):
    return int(math.ceil(math.log2(max(psi, 2))))


def build_tree(S, rng, max_depth):
    """Ein Isolationsbaum über die Unterstichprobe S (psi x p). Ebene für Ebene: je Knoten ein zufälliges nicht konstantes Merkmal und ein gleichverteilter Schnittpunkt zwischen Minimum und Maximum."""
    psi, p = S.shape
    feature, threshold, left, right, size, depth = [-1], [0.0], [-1], [-1], [psi], [0]
    node_of = np.zeros(psi, dtype=int)
    frontier = [0]
    for level in range(max_depth):
        if not frontier:
            break
        sizes = np.array([size[k] for k in frontier])
        active = [k for k, sz in zip(frontier, sizes) if sz > 1]
        if not active:
            break
        m = len(active)
        local = np.full(len(feature), -1, dtype=int)
        local[active] = np.arange(m)
        rows = np.flatnonzero(local[node_of] >= 0)
        loc = local[node_of[rows]]
        mins = np.full((m, p), np.inf)
        maxs = np.full((m, p), -np.inf)
        np.minimum.at(mins, loc, S[rows])
        np.maximum.at(maxs, loc, S[rows])
        valid = maxs > mins
        pick = rng.random((m, p))
        pick[~valid] = -1.0
        f = np.argmax(pick, axis=1)
        splittable = valid.any(axis=1)
        lo, hi = mins[np.arange(m), f], maxs[np.arange(m), f]
        thr = lo + rng.random(m) * (hi - lo)
        go_left = S[rows, f[loc]] < thr[loc]
        n_left = np.bincount(loc[go_left], minlength=m)
        n_right = np.bincount(loc[~go_left], minlength=m)
        splittable &= (n_left > 0) & (n_right > 0)
        next_frontier = []
        lid_of = np.full(m, -1, dtype=int)
        rid_of = np.full(m, -1, dtype=int)
        for j, node in enumerate(active):
            if not splittable[j]:
                continue
            lid, rid = len(feature), len(feature) + 1
            for cnt in (n_left[j], n_right[j]):
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(-1)
                size.append(int(cnt))
                depth.append(level + 1)
            feature[node], threshold[node], left[node], right[node] = int(f[j]), float(thr[j]), lid, rid
            lid_of[j], rid_of[j] = lid, rid
            next_frontier += [lid, rid]
        moved = lid_of[loc] >= 0
        node_of[rows[moved]] = np.where(go_left[moved], lid_of[loc[moved]], rid_of[loc[moved]])
        frontier = next_frontier
    return Tree(np.array(feature), np.array(threshold), np.array(left), np.array(right), np.array(size), np.array(depth))


def fit_forest(X, n_trees=100, psi=256, seed=0):
    """Wald aus `n_trees` Bäumen, je auf einer Unterstichprobe der Größe min(psi, n) ohne Zurücklegen (Seed getrennt von dem der Aufnahme)."""
    n = len(X)
    psi = int(min(psi, n))
    rng = np.random.default_rng([seed, 4242])
    limit = height_limit(psi)
    trees, subsets = [], []
    for _ in range(n_trees):
        idx = np.sort(rng.choice(n, size=psi, replace=False))
        trees.append(build_tree(X[idx], rng, limit))
        subsets.append(idx)
    return Forest(tuple(trees), psi, tuple(subsets), limit)


def tree_path_length(tree, X):
    """Pfadlänge jedes Punkts in einem Baum: Tiefe des Blatts plus c(Blattgröße) für die nicht mehr getrennten Punkte."""
    n = len(X)
    node = np.zeros(n, dtype=int)
    rows = np.arange(n)
    depth = np.zeros(n)
    while True:
        f = tree.feature[node]
        internal = f >= 0
        if not internal.any():
            break
        go_left = X[rows, np.maximum(f, 0)] < tree.threshold[node]
        node = np.where(internal, np.where(go_left, tree.left[node], tree.right[node]), node)
        depth += internal
    return depth + c_factor(tree.size[node])


def path_lengths(forest, X):
    """(Bäume, n) Pfadlängen aller Punkte in allen Bäumen."""
    return np.stack([tree_path_length(t, X) for t in forest.trees])


def score_from_paths(paths, psi):
    """Anomalie-Wert s = 2^(-E[h] / c(psi)) aus den Pfadlängen (Bäume x n): nahe 1 = leicht zu isolieren, um 0.5 = normal, nahe 0 = sehr dicht."""
    return 2.0 ** (-paths.mean(axis=0) / float(c_factor(psi)))


def score(forest, X):
    return score_from_paths(path_lengths(forest, X), forest.psi)


def score_convergence(paths, psi, steps):
    """Anomalie-Werte nach den ersten k Bäumen (Zeilen: k in `steps`) - wie schnell der Wert stabil wird."""
    cum = np.cumsum(paths, axis=0)
    return np.stack([2.0 ** (-(cum[k - 1] / k) / float(c_factor(psi))) for k in steps])


def flag_top(scores, share):
    """Markiert die round(share * n) Punkte mit dem größten Anomalie-Wert (mindestens einen)."""
    k = max(1, int(round(share * len(scores))))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores, kind="stable")[:k]] = True
    return flagged


# --- Darstellung ------------------------------------------------------------------------------------------------------------------


def point_path(tree, x):
    """Der Weg eines Punkts durch einen Baum: [(Merkmal, Schnittpunkt, ging nach links)] bis zum Blatt, dazu die Blattgröße."""
    node, steps = 0, []
    while tree.feature[node] >= 0:
        went_left = bool(x[tree.feature[node]] < tree.threshold[node])
        steps.append((int(tree.feature[node]), float(tree.threshold[node]), went_left))
        node = tree.left[node] if went_left else tree.right[node]
    return steps, int(tree.size[node])


def tree_segments(tree, bounds):
    """Schnittlinien eines Baums über zwei Merkmale als Strecken (x0, y0, x1, y1, Tiefe) im Rechteck `bounds` = (xmin, xmax, ymin, ymax) - zum Zeichnen der Zerlegung der Ebene."""
    segments = []

    def walk(node, xmin, xmax, ymin, ymax):
        if tree.feature[node] < 0:
            return
        f, t = tree.feature[node], tree.threshold[node]
        if f == 0:
            segments.append((t, ymin, t, ymax, int(tree.depth[node])))
            walk(tree.left[node], xmin, t, ymin, ymax)
            walk(tree.right[node], t, xmax, ymin, ymax)
        else:
            segments.append((xmin, t, xmax, t, int(tree.depth[node])))
            walk(tree.left[node], xmin, xmax, ymin, t)
            walk(tree.right[node], xmin, xmax, t, ymax)

    walk(0, *bounds)
    return segments
