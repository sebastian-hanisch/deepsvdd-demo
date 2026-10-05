"""Unabhängiges Orakel für Deep SVDD und die kopierten Bausteine: Verlauf der Zielfunktion (Index k = nach k Schritten, der letzte Wert gehört zum fertigen Netz), Zielfunktion von Hand und Gradient jeder
einzelnen Komponente per zentraler Differenz (beide Varianten, mit und ohne Bias, mit Gewichtszerfall), Adam gegen die Lehrbuchformel, LOF und One-Class SVM gegen scikit-learn."""

import numpy as np
import pytest

import dsvdd_algorithm as dv
import dsvdd_lof as lof
import dsvdd_ocsvm as svm


def _data(n=60, p=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)) @ rng.standard_normal((p, p)) + 3.0


def test_objective_history_ends_with_the_objective_of_the_finished_network():
    """Früher maß die Schleife die Zielfunktion vor dem Schritt: der letzte Wert gehörte zum Netz nach n - 1 Schritten, nicht zum gezeigten Netz."""
    X = _data()
    for variant in dv.VARIANTS:
        m = dv.fit_deep_svdd(X, depth=1, width=8, n_out=3, variant=variant, nu=0.1, lr=0.01, n_epochs=40, seed=2, weight_decay=1e-3)
        assert len(m.loss_history) == 41
        d2 = dv.distances2(m)
        r2_in_force = m.loss_history[-1]
        if variant == "one_class":
            assert r2_in_force == pytest.approx(float(d2.mean()) + 0.5e-3 * sum((w ** 2).sum() for w in m.weights), rel=1e-10)
        else:
            assert np.isfinite(r2_in_force)
    short = dv.fit_deep_svdd(X, depth=1, width=8, n_out=3, lr=0.01, n_epochs=5, seed=3)
    long = dv.fit_deep_svdd(X, depth=1, width=8, n_out=3, lr=0.01, n_epochs=9, seed=3)
    assert np.allclose(long.loss_history[:6], short.loss_history)                      # Index k = Zielfunktion nach k Schritten, unabhängig von der Gesamtzahl der Epochen


def _full_gradient_error(loss_fn, arrays, grads, eps=1e-6):
    worst = 0.0
    for arr, g in zip(arrays, grads):
        for idx in np.ndindex(arr.shape):
            old = arr[idx]
            arr[idx] = old + eps
            up = loss_fn()
            arr[idx] = old - eps
            down = loss_fn()
            arr[idx] = old
            num = (up - down) / (2 * eps)
            worst = max(worst, abs(num - g[idx]) / max(1.0, abs(num), abs(g[idx])))
    return worst


@pytest.mark.parametrize("activation,depth,variant,bias,decay", [("tanh", 1, "one_class", False, 0.0), ("tanh", 2, "soft_boundary", True, 1e-2), ("relu", 1, "soft_boundary", False, 0.1),
                                                                   ("relu", 2, "one_class", True, 0.1), ("tanh", 0, "soft_boundary", False, 0.0)])
def test_objective_by_hand_and_every_gradient_component_match_central_differences(activation, depth, variant, bias, decay):
    rng = np.random.default_rng(20 + depth)
    Z = rng.standard_normal((9, 4))
    w, b = dv.init_parameters(dv.layer_sizes(4, depth, 4, 3), activation, 5, bias)
    if bias:
        b = [rng.normal(size=len(x)) * 0.3 for x in b]
    c = rng.standard_normal(3)
    nu = 0.3
    d2 = ((dv.output(w, b, Z, activation) - c) ** 2).sum(axis=1)
    r2 = float(np.quantile(d2, 0.7)) * 1.0123 + 1e-3 if variant == "soft_boundary" else 0.0            # Radius abseits der Knickstellen des Hinge
    loss, gw, gb = dv.loss_and_gradients(w, b, Z, c, activation, variant, nu, r2, decay)
    ref = d2.mean() if variant == "one_class" else r2 + np.maximum(d2 - r2, 0).sum() / (nu * len(Z))
    assert loss == pytest.approx(ref + 0.5 * decay * sum((x ** 2).sum() for x in w), rel=1e-10)
    f = lambda: dv.loss_and_gradients(w, b, Z, c, activation, variant, nu, r2, decay)[0]
    assert _full_gradient_error(f, w, gw) < 1e-6
    if bias:
        assert _full_gradient_error(f, b, gb) < 1e-6


def test_training_follows_textbook_adam():
    """Adam nach Kingma und Ba (Algorithmus 1) mit den Gradienten der Demo, 25 Schritte: gleiche Gewichte bis auf die Stellung von epsilon (< 1e-3)."""
    X = _data(40, 4, 1)
    m = dv.fit_deep_svdd(X, depth=1, width=6, n_out=3, lr=0.01, n_epochs=25, seed=4, weight_decay=1e-3)
    w, _ = dv.init_parameters(dv.layer_sizes(4, 1, 6, 3), "tanh", 4)
    c = dv.make_center(w, None, m.Z, "tanh")
    assert np.allclose(c, m.center)
    mom, var = [np.zeros_like(x) for x in w], [np.zeros_like(x) for x in w]
    for t in range(1, 26):
        _, gw, _ = dv.loss_and_gradients(w, None, m.Z, c, "tanh", "one_class", 0.1, 0.0, 1e-3)
        for k, g in enumerate(gw):
            mom[k] = 0.9 * mom[k] + 0.1 * g
            var[k] = 0.999 * var[k] + 0.001 * g ** 2
            w[k] -= 0.01 * (mom[k] / (1 - 0.9 ** t)) / (np.sqrt(var[k] / (1 - 0.999 ** t)) + 1e-8)
    assert max(np.abs(a - b).max() for a, b in zip(m.weights, w)) < 1e-3


def test_threshold_flags_at_most_the_share_nu():
    rng = np.random.default_rng(6)
    for i in range(400):
        n, nu = int(rng.integers(2, 120)), float(rng.choice([0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0]))
        d2 = rng.exponential(size=n)
        if i % 3 == 0:
            d2 = np.round(d2, 1)
        assert (d2 > dv.quantile_r2(d2, nu)).sum() <= nu * n + 1e-9


def test_lof_equals_scikit_learn_including_new_points():
    neighbors = pytest.importorskip("sklearn.neighbors")
    rng = np.random.default_rng(7)
    for _ in range(40):
        n, p = int(rng.integers(6, 60)), int(rng.integers(1, 4))
        X = rng.standard_normal((n, p)) * rng.uniform(0.5, 3, p)
        X[: max(1, n // 10)] += rng.uniform(3, 8)
        k = lof.clamp_k(int(rng.integers(1, n)), n)
        fit = lof.fit_lof(X, k)
        sk = neighbors.LocalOutlierFactor(n_neighbors=k, algorithm="brute").fit(X)
        assert np.allclose(fit.lof, -sk.negative_outlier_factor_, rtol=1e-8)
        Q = rng.standard_normal((6, p)) * 3
        skn = neighbors.LocalOutlierFactor(n_neighbors=k, algorithm="brute", novelty=True).fit(X)
        assert np.allclose(lof.score_new(X, fit, Q), -skn.score_samples(Q), rtol=1e-8)


def test_one_class_svm_equals_scikit_learn_decision_function_and_the_nu_properties():
    sk_svm = pytest.importorskip("sklearn.svm")
    rng = np.random.default_rng(8)
    for nu, gamma in ((0.05, 0.2), (0.2, 1.0), (0.5, 0.05), (0.8, 3.0), (0.3, 1.0), (0.1, 0.5)):
        n = int(rng.integers(12, 60))
        X = rng.standard_normal((n, 2))
        X[: max(1, n // 8)] += rng.uniform(2, 5)
        fit = svm.fit_ocsvm(X, nu, gamma, tol=1e-7)
        sk = sk_svm.OneClassSVM(kernel="rbf", nu=nu, gamma=gamma, tol=1e-6).fit(X)
        f = svm.train_decision(fit)
        assert fit.converged and np.abs(f - sk.decision_function(X)).max() < 1e-3
        assert (f < -1e-5).mean() <= nu + 1e-9 and fit.n_support / n >= nu - 1e-9               # nu: obere Schranke der Ausreißer, untere der Stützvektoren
        assert fit.alpha.sum() == pytest.approx(nu * n)
