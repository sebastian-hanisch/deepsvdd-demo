"""ECOD (Empirical Cumulative distribution functions for Outlier Detection, numpy von Grund auf, nach Li, Zhao, Hu, Botta, Ionescu und Chen).

Je Merkmal j werden die empirischen Verteilungsfunktionen der Randverteilung bestimmt: links F_l(x) = #{x_i <= x} / n und rechts F_r(x) = #{x_i >= x} / n (Bindungen zählen auf beiden Seiten mit, die
Werte liegen in [1/n, 1]). Der Beitrag eines Merkmals ist die negative Log-Schwanzwahrscheinlichkeit -log F - null in der Mitte, log n am äußersten Rand. Es gibt keinen einzigen Parameter.

Kombination der Merkmale (Variante):
- **paper**: O_left = sum_j -log F_l, O_right = sum_j -log F_r, O_auto = die Seite, auf die die Schiefe des Merkmals zeigt (links bei negativer Schiefe, sonst rechts); Wert = max(O_left, O_right, O_auto).
- **auto**: nur O_auto.
- **both**: je Merkmal der größere der beiden Schwänze, summiert (zweiseitig).
- **twosided**: je Merkmal -log(min(1, 2 * min(F_l, F_r))), summiert - zweiseitige p-Werte; unter Unabhängigkeit exakt Gamma(p, 1), die Grundlage einer Schwelle ohne Vorwissen.
- **pyod**: die Implementierung in PyOD (je Merkmal das Maximum aus links, rechts und einem Schiefe-Term, dabei zählt nur das Vorzeichen der Schiefe: bei genau 0 beide Seiten; summiert) - zum Abgleich.

Unter Unabhängigkeit und stetigen Randverteilungen ist -log F ~ Exp(1), eine Summe über p Merkmale also Gamma(p, 1) und 2 * Summe ~ chi^2(2p) (Fisher-Methode): daraus folgt eine Schwelle ohne Vorwissen über den Anteil."""

import numpy as np

VARIANTS = ("paper", "twosided", "both", "auto", "pyod")
VARIANT_LABELS = {"paper": "Paper: Maximum der drei Summen", "twosided": "zweiseitig mit p-Werten (Fisher)", "both": "je Merkmal der größere Schwanz", "auto": "nur nach Schiefe (O_auto)",
                  "pyod": "PyOD-Implementierung"}


def tail_probabilities(X):
    """(F_links, F_rechts) je Wert und Merkmal, beide [n, p] in [1/n, 1]; Bindungen zählen auf beiden Seiten."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    left = np.empty_like(X)
    right = np.empty_like(X)
    for j in range(X.shape[1]):
        col = np.sort(X[:, j])
        left[:, j] = np.searchsorted(col, X[:, j], side="right") / n                 # #{x_i <= x} / n
        right[:, j] = (n - np.searchsorted(col, X[:, j], side="left")) / n           # #{x_i >= x} / n
    return left, right


def skewness(X):
    """Stichproben-Schiefe je Merkmal (Moment-Schätzer, wie scipy.stats.skew); 0 bei konstantem Merkmal."""
    X = np.asarray(X, dtype=float)
    d = X - X.mean(axis=0)
    m2 = (d ** 2).mean(axis=0)
    m3 = (d ** 3).mean(axis=0)
    out = np.zeros(X.shape[1])
    ok = m2 > 1e-24
    out[ok] = m3[ok] / m2[ok] ** 1.5
    return out


def contributions(X):
    """Beiträge je Tour und Merkmal: (U_links, U_rechts) = (-log F_l, -log F_r), beide [n, p] >= 0."""
    left, right = tail_probabilities(X)
    return -np.log(left), -np.log(right)


def score(X, variant="paper"):
    """ECOD-Wert je Tour (größer = auffälliger)."""
    X = np.asarray(X, dtype=float)
    u_l, u_r = contributions(X)
    skew = skewness(X)
    if variant == "paper":
        u_auto = np.where(skew < 0, u_l, u_r)
        return np.maximum(np.maximum(u_l.sum(axis=1), u_r.sum(axis=1)), u_auto.sum(axis=1))
    if variant == "auto":
        return np.where(skew < 0, u_l, u_r).sum(axis=1)
    if variant == "both":
        return np.maximum(u_l, u_r).sum(axis=1)
    if variant == "twosided":
        return per_feature_contribution(X, "twosided").sum(axis=1)
    if variant == "pyod":
        sk = np.sign(skew)                                                                                      # PyOD rechnet mit dem Vorzeichen der Schiefe
        u_skew = u_l * -np.sign(sk - 1) + u_r * np.sign(sk + 1)
        return np.maximum(np.maximum(u_l, u_r), u_skew).sum(axis=1)
    raise ValueError(variant)


def per_feature_contribution(X, variant="paper"):
    """Beitrag jedes Merkmals zum Wert (für die Darstellung): bei 'paper' die Beiträge der Seite, die den Maximalwert liefert; sonst die Beiträge der jeweiligen Variante. [n, p]."""
    X = np.asarray(X, dtype=float)
    u_l, u_r = contributions(X)
    skew = skewness(X)
    if variant == "paper":
        u_auto = np.where(skew < 0, u_l, u_r)
        sums = np.stack([u_l.sum(axis=1), u_r.sum(axis=1), u_auto.sum(axis=1)])
        best = sums.argmax(axis=0)
        stack = np.stack([u_l, u_r, u_auto])
        return stack[best, np.arange(len(X))]
    if variant == "auto":
        return np.where(skew < 0, u_l, u_r)
    if variant == "both":
        return np.maximum(u_l, u_r)
    if variant == "twosided":
        left, right = tail_probabilities(X)
        return -np.log(np.minimum(1.0, 2.0 * np.minimum(left, right)))
    if variant == "pyod":
        sk = np.sign(skew)                                                                                      # PyOD rechnet mit dem Vorzeichen der Schiefe
        u_skew = u_l * -np.sign(sk - 1) + u_r * np.sign(sk + 1)
        return np.maximum(np.maximum(u_l, u_r), u_skew)
    raise ValueError(variant)
