"""Deep SVDD (numpy von Grund auf, Backpropagation von Hand, Adam, Vollbatch-Training) nach Ruff, Vandermeulen, Görnitz, Deecke, Siddiqui, Binder, Müller und Kloft.

Ein neuronales Netz phi(x; W) bildet die (standardisierten) Kennzahlen einer Tour auf d Ausgabewerte ab; die Normalen sollen in einer möglichst kleinen Kugel um ein **festes Zentrum c** landen.
Der Anomalie-Wert ist der quadrierte Abstand ||phi(x) - c||^2 (Soft-Boundary: minus R^2).

- **Netz:** Schichten der Breite `width` (Tiefe `depth`, tanh oder ReLU) und eine lineare Ausgabeschicht. **Ohne Bias-Terme** (Standard, wie im Paper): mit Bias könnte das Netz alle Eingaben auf c abbilden
  (Hypersphären-Kollaps, ||phi - c|| = 0 für alles); ohne Bias ist phi(0) = 0, und die triviale Lösung ist ausgeschlossen. `bias=True` zeigt den Kollaps.
- **Zentrum c:** Mittel der Ausgaben des **initialen** (zufälligen) Netzes auf den Trainingsdaten, danach fest. Komponenten mit |c_k| < eps werden auf +-eps gesetzt (wie im Paper, sonst wäre 0 eine triviale Lösung).
- **One-Class-Ziel:** (1/n) sum_i ||phi(x_i) - c||^2 + lambda / 2 * sum ||W||^2.
- **Soft-Boundary-Ziel:** R^2 + 1 / (nu n) * sum_i max(0, ||phi(x_i) - c||^2 - R^2) + lambda / 2 * sum ||W||^2; R^2 wird alle `r_every` Epochen (nach `warmup` Epochen mit R = 0) als (1 - nu)-Quantil der Abstände neu gesetzt.
- **Schwelle:** R^2 = (1 - nu)-Quantil der Trainingsabstände (One-Class) bzw. das gelernte R^2 (Soft-Boundary): wie bei der One-Class SVM ein eingebauter Anteil nu.
- Initialisierung Xavier (tanh) bzw. He (ReLU); der Netz-Seed ist vom Datensatz-Seed entkoppelt."""

from dataclasses import dataclass

import numpy as np

ACTIVATIONS = ("tanh", "relu")
VARIANTS = ("one_class", "soft_boundary")
SCALINGS = ("minmax", "standard")
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
CENTER_EPS = 0.1
WARMUP = 10
R_EVERY = 5


def activate(name, x):
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    raise ValueError(f"unbekannte Aktivierung {name!r}")


def activation_grad(name, out):
    """Ableitung der Aktivierung, ausgedrückt durch ihren AUSGABEwert."""
    if name == "tanh":
        return 1.0 - out ** 2
    if name == "relu":
        return (out > 0).astype(float)
    raise ValueError(f"unbekannte Aktivierung {name!r}")


def layer_sizes(n_features, depth, width, n_out):
    return [n_features] + [width] * depth + [n_out]


def n_parameters(sizes, bias=False):
    return int(sum(a * b + (b if bias else 0) for a, b in zip(sizes[:-1], sizes[1:])))


def init_parameters(sizes, activation, seed, bias=False):
    """(Gewichte, Biases oder None); Bias-Terme starten bei 0."""
    rng = np.random.default_rng(seed)
    weights = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if activation == "relu":
            weights.append(rng.normal(size=(fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
        else:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
    biases = [np.zeros(b) for b in sizes[1:]] if bias else None
    return weights, biases


def forward(weights, biases, Z, activation):
    """Ausgaben aller Schichten (Index 0 = Eingabe); die letzte Schicht ist linear."""
    outs = [Z]
    last = len(weights) - 1
    for i, w in enumerate(weights):
        z = outs[-1] @ w
        if biases is not None:
            z = z + biases[i]
        outs.append(activate(activation, z) if i < last else z)
    return outs


def output(weights, biases, Z, activation):
    return forward(weights, biases, Z, activation)[-1]


def make_center(weights, biases, Z, activation, eps=CENTER_EPS):
    """Zentrum: Mittel der Ausgaben des initialen Netzes; kleine Komponenten werden auf +-eps geschoben (Vorzeichen, bei 0: +)."""
    c = output(weights, biases, Z, activation).mean(axis=0)
    small = np.abs(c) < eps
    c = np.where(small, np.where(c < 0, -eps, eps), c)
    return c


def quantile_r2(d2, nu):
    """Das (1 - nu)-Quantil der quadrierten Abstände (untere Interpolation: höchstens ein Anteil nu der Touren liegt strikt darüber)."""
    return float(np.quantile(d2, 1.0 - nu, method="higher"))


def loss_and_gradients(weights, biases, Z, c, activation, variant="one_class", nu=0.1, r2=0.0, weight_decay=0.0):
    """Zielfunktion und Gradienten (Backpropagation von Hand); R^2 gilt als fest (Soft-Boundary: es wird zwischen den Aktualisierungen nicht mitgelernt)."""
    n = len(Z)
    outs = forward(weights, biases, Z, activation)
    diff = outs[-1] - c
    d2 = (diff ** 2).sum(axis=1)
    if variant == "one_class":
        loss = float(d2.mean())
        delta = 2.0 * diff / n
    elif variant == "soft_boundary":
        active = d2 > r2
        loss = float(r2 + np.maximum(d2 - r2, 0.0).sum() / (nu * n))
        delta = 2.0 * diff * active[:, None] / (nu * n)
    else:
        raise ValueError(f"variant in {VARIANTS}")
    loss += 0.5 * weight_decay * float(sum((w ** 2).sum() for w in weights))
    n_layers = len(weights)
    gw, gb = [None] * n_layers, ([None] * n_layers if biases is not None else None)
    for i in reversed(range(n_layers)):
        gw[i] = outs[i].T @ delta + weight_decay * weights[i]
        if gb is not None:
            gb[i] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ weights[i].T) * activation_grad(activation, outs[i])
    return loss, gw, gb


def snapshot_epochs(n_epochs, count=12):
    grid = np.unique(np.round(np.geomspace(1, max(n_epochs, 2), count)).astype(int))
    return sorted(set(int(e) for e in grid) | {0, int(n_epochs)})


@dataclass(frozen=True)
class Model:
    weights: list
    biases: object                   # Liste oder None (bias-frei)
    activation: str
    depth: int
    width: int
    n_out: int
    variant: str
    nu: float
    center: np.ndarray
    r2: float                        # Schwelle: gelernt (Soft-Boundary) bzw. (1 - nu)-Quantil der Trainingsabstände (One-Class)
    outputs: np.ndarray              # [n, d] Ausgaben der Trainings-Touren
    snapshots: dict                  # Epoche -> Ausgaben (Kopie), inkl. 0 und n_epochs
    r2_snapshots: dict               # Epoche -> R^2 (Quantil der Abstände zu diesem Zeitpunkt)
    loss_history: np.ndarray         # [n_epochs]
    mean: np.ndarray
    scale: np.ndarray
    Z: np.ndarray
    lr: float
    n_epochs: int
    seed: int
    weight_decay: float

    @property
    def n(self):
        return len(self.Z)

    @property
    def bias(self):
        return self.biases is not None

    @property
    def n_parameters(self):
        return n_parameters(layer_sizes(self.Z.shape[1], self.depth, self.width, self.n_out), self.bias)


def fit_deep_svdd(X, depth=2, width=32, n_out=8, activation="tanh", variant="one_class", nu=0.1, lr=0.001, n_epochs=300, seed=0, bias=False, weight_decay=1e-6, scaling="minmax"):
    """Trainiert Deep SVDD auf den Kennzahlen X [n, p] (werden intern standardisiert)."""
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation in {ACTIVATIONS}")
    if variant not in VARIANTS:
        raise ValueError(f"variant in {VARIANTS}")
    if depth < 0 or n_out < 1 or not 0 < nu <= 1:
        raise ValueError("depth >= 0, n_out >= 1, 0 < nu <= 1")
    X = np.asarray(X, dtype=float)
    if scaling == "minmax":
        mean = X.min(axis=0)
        scale = X.max(axis=0) - X.min(axis=0)
    elif scaling == "standard":
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=1)
    else:
        raise ValueError(f"scaling in {SCALINGS}")
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale
    sizes = layer_sizes(Z.shape[1], depth, width, n_out)
    weights, biases = init_parameters(sizes, activation, seed, bias)
    c = make_center(weights, biases, Z, activation)
    params = weights + (biases if bias else [])
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    n_layers = len(weights)
    wanted = set(snapshot_epochs(n_epochs))
    loss_history = np.zeros(n_epochs)
    r2 = 0.0

    def dist2():
        return ((output(weights, biases, Z, activation) - c) ** 2).sum(axis=1)
    snapshots = {0: output(weights, biases, Z, activation).copy()} if 0 in wanted else {}
    r2_snapshots = {0: quantile_r2(dist2(), nu)} if 0 in wanted else {}
    for epoch in range(1, n_epochs + 1):
        if variant == "soft_boundary" and epoch > WARMUP and epoch % R_EVERY == 0:
            r2 = quantile_r2(dist2(), nu)
        loss, gw, gb = loss_and_gradients(weights, biases, Z, c, activation, variant, nu, r2, weight_decay)
        loss_history[epoch - 1] = loss
        grads = gw + (gb if bias else [])
        lr_t = lr * np.sqrt(1.0 - BETA2 ** epoch) / (1.0 - BETA1 ** epoch)
        for k, g in enumerate(grads):
            m[k] = BETA1 * m[k] + (1 - BETA1) * g
            v[k] = BETA2 * v[k] + (1 - BETA2) * g ** 2
            params[k] -= lr_t * m[k] / (np.sqrt(v[k]) + ADAM_EPS)
        if epoch in wanted:
            snapshots[epoch] = output(weights, biases, Z, activation).copy()
            r2_snapshots[epoch] = quantile_r2(dist2(), nu)
    d2 = dist2()
    r2 = quantile_r2(d2, nu)                                                     # Schwelle: (1 - nu)-Quantil der Trainingsabstände (bei Soft-Boundary die letzte Aktualisierung von R^2)
    return Model(weights=weights, biases=biases, activation=activation, depth=depth, width=width, n_out=n_out, variant=variant, nu=float(nu), center=c, r2=float(r2), outputs=output(weights, biases, Z, activation),
                 snapshots=snapshots, r2_snapshots=r2_snapshots, loss_history=loss_history, mean=mean, scale=scale, Z=Z, lr=float(lr), n_epochs=int(n_epochs), seed=int(seed), weight_decay=float(weight_decay))


def transform(model, X_new):
    """Ausgaben des trainierten Netzes für neue Touren."""
    Zn = (np.asarray(X_new, dtype=float) - model.mean) / model.scale
    return output(model.weights, model.biases, Zn, model.activation)


def distances2(model, X_new=None):
    """Quadrierter Abstand zum Zentrum (Trainings-Touren, wenn X_new fehlt)."""
    out = model.outputs if X_new is None else transform(model, X_new)
    return ((out - model.center) ** 2).sum(axis=1)


def score(model, X_new=None):
    """Anomalie-Wert: ||phi(x) - c||^2 (One-Class); bei der Soft-Boundary-Variante zusätzlich - R^2 (gleiche Rangfolge, Schwelle bei 0)."""
    d2 = distances2(model, X_new)
    return d2 - model.r2 if model.variant == "soft_boundary" else d2


def output_spread(model):
    """Streuung der Ausgaben über die Touren (Wurzel der mittleren Varianz je Ausgabe): fällt beim Kollaps gegen 0."""
    return float(np.sqrt(model.outputs.var(axis=0).mean()))
