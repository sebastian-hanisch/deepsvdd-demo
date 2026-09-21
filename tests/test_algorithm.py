"""Deep SVDD: Gradienten gegen Finite Differenzen (beide Ziele, tanh und ReLU, mit und ohne Bias), Netz ohne Bias, Zentrum (initiales Netz, fest, eps-Regel), Quantil-Schwelle, die Falle zentrierter Eingaben,
Verkleinerung der Ausgaben-Streuung, Determinismus und Netz-Seed; die kopierten Bausteine der Vorgänger (One-Class SVM mit sklearn-Kreuzprüfung, LOF, Isolation Forest, χ², MCD, ECOD)."""

import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.svm import OneClassSVM

import dsvdd_algorithm as dv
import dsvdd_ecod as ecod
import dsvdd_ee_algorithm as alg
import dsvdd_isolation_forest as isf
import dsvdd_lof as lof
import dsvdd_ocsvm as svm


def _data(n=150, p=5, n_out=8, shift=4.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.standard_normal((n - n_out, p)), shift + 0.5 * rng.standard_normal((n_out, p))])
    return X, np.arange(n) >= n - n_out


def _auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --- Gradienten ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["one_class", "soft_boundary"])
@pytest.mark.parametrize("activation", ["tanh", "relu"])
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_backpropagation_matches_finite_differences(variant, activation, depth, bias):
    rng = np.random.default_rng(3)
    Z = rng.random((25, 4))
    sizes = dv.layer_sizes(4, depth, 6, 3)
    w, b = dv.init_parameters(sizes, activation, 1, bias)
    if bias:
        b = [rng.normal(size=len(x)) * 0.1 for x in b]
    c = dv.make_center(w, b, Z, activation)
    d2 = ((dv.output(w, b, Z, activation) - c) ** 2).sum(axis=1)
    ds = np.sort(d2)
    r2 = float(0.5 * (ds[len(ds) // 2] + ds[len(ds) // 2 + 1]))                                    # zwischen zwei Abständen: kein Knick des Hinge in der Nähe
    kwargs = dict(activation=activation, variant=variant, nu=0.2, r2=r2, weight_decay=1e-3)
    loss, gw, gb = dv.loss_and_gradients(w, b, Z, c, **kwargs)
    eps = 1e-6

    def numeric(arr, idx):
        old = arr[idx]
        arr[idx] = old + eps
        up = dv.loss_and_gradients(w, b, Z, c, **kwargs)[0]
        arr[idx] = old - eps
        down = dv.loss_and_gradients(w, b, Z, c, **kwargs)[0]
        arr[idx] = old
        return (up - down) / (2 * eps)
    checked = 0
    for layer in range(len(w)):
        for idx in [(0, 0), (w[layer].shape[0] - 1, w[layer].shape[1] - 1), (1 % w[layer].shape[0], 2 % w[layer].shape[1])]:
            assert numeric(w[layer], idx) == pytest.approx(gw[layer][idx], rel=1e-4, abs=1e-7)
            checked += 1
        if bias:
            assert numeric(b[layer], (0,)) == pytest.approx(gb[layer][0], rel=1e-4, abs=1e-7)
    assert checked >= 3 and (gb is not None) == bias


def test_activation_derivatives_match_finite_differences():
    x = np.linspace(-2, 2, 9) + 0.0123
    for name in dv.ACTIVATIONS:
        h = 1e-6
        numeric = (dv.activate(name, x + h) - dv.activate(name, x - h)) / (2 * h)
        assert np.allclose(dv.activation_grad(name, dv.activate(name, x)), numeric, atol=1e-6)


# --- Netz, Zentrum, Schwelle ---------------------------------------------------------------------------------------------------------------


def test_the_network_has_no_bias_terms_and_maps_the_origin_to_zero():
    X, _ = _data()
    m = dv.fit_deep_svdd(X, n_epochs=20, seed=1)
    assert m.biases is None and not m.bias
    for act in dv.ACTIVATIONS:
        w, b = dv.init_parameters(dv.layer_sizes(5, 2, 8, 4), act, 0)
        assert b is None and np.array_equal(dv.output(w, b, np.zeros((1, 5)), act), np.zeros((1, 4)))
    assert dv.n_parameters(dv.layer_sizes(5, 2, 8, 4)) == 5 * 8 + 8 * 8 + 8 * 4 and dv.n_parameters(dv.layer_sizes(5, 2, 8, 4), True) == 5 * 8 + 8 * 8 + 8 * 4 + 8 + 8 + 4


def test_center_is_the_mean_of_the_initial_outputs_and_stays_fixed():
    X, _ = _data()
    m0 = dv.fit_deep_svdd(X, n_epochs=0, seed=4)
    m1 = dv.fit_deep_svdd(X, n_epochs=60, seed=4)
    assert np.array_equal(m0.center, m1.center) and np.allclose(m1.snapshots[0], m0.outputs)
    w, b = dv.init_parameters(dv.layer_sizes(5, 2, 32, 8), "tanh", 4)
    Z = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    raw = dv.output(w, b, Z, "tanh").mean(axis=0)
    assert np.allclose(m0.center, np.where(np.abs(raw) < dv.CENTER_EPS, np.where(raw < 0, -dv.CENTER_EPS, dv.CENTER_EPS), raw))
    assert (np.abs(m0.center) >= dv.CENTER_EPS - 1e-12).all()


def test_small_center_components_are_pushed_to_plus_or_minus_eps():
    Z = np.random.default_rng(0).random((10, 3))
    zero_w = [np.zeros((3, 2))]
    assert np.array_equal(dv.make_center(zero_w, None, Z, "tanh"), [dv.CENTER_EPS, dv.CENTER_EPS])
    neg_w = [np.array([[-0.001, 0.5], [-0.001, 0.5], [-0.001, 0.5]])]
    c = dv.make_center(neg_w, None, Z, "tanh")
    assert c[0] == -dv.CENTER_EPS and c[1] > dv.CENTER_EPS


def test_threshold_is_the_one_minus_nu_quantile_and_flags_at_most_nu():
    X, _ = _data(n=200)
    for nu in (0.05, 0.1, 0.3):
        m = dv.fit_deep_svdd(X, nu=nu, n_epochs=30, seed=2)
        d2 = dv.distances2(m)
        assert m.r2 == dv.quantile_r2(d2, nu) and float((d2 > m.r2).mean()) <= nu + 1e-9 and float((d2 >= m.r2).mean()) >= nu - 1e-9


def test_scores_for_both_variants():
    X, _ = _data()
    a = dv.fit_deep_svdd(X, n_epochs=20, seed=0)
    b = dv.fit_deep_svdd(X, n_epochs=20, seed=0, variant="soft_boundary")
    assert np.allclose(dv.score(a), dv.distances2(a)) and np.allclose(dv.score(b), dv.distances2(b) - b.r2)
    assert np.allclose(dv.distances2(a, X[:7]), dv.distances2(a)[:7])                       # neue Touren durch dasselbe Netz


# --- Training --------------------------------------------------------------------------------------------------------------------------------


def test_the_objective_falls_during_training_and_the_outputs_contract():
    X, _ = _data()
    m = dv.fit_deep_svdd(X, n_epochs=200, seed=1)
    assert m.loss_history[-1] < 0.5 * m.loss_history[0] and np.isfinite(m.loss_history).all()
    assert dv.output_spread(m) < 0.6 * float(np.sqrt(m.snapshots[0].var(axis=0).mean()))
    assert set(m.snapshots) >= {0, 200} and m.n_epochs == 200 and m.r2_snapshots[200] == m.r2


def test_soft_boundary_uses_the_learned_radius_and_the_hinge():
    X, _ = _data()
    m = dv.fit_deep_svdd(X, n_epochs=80, seed=1, variant="soft_boundary", nu=0.1)
    assert m.r2 == dv.quantile_r2(dv.distances2(m), 0.1) and m.r2 > 0 and np.isfinite(m.loss_history).all()
    w, b = m.weights, m.biases
    Z = m.Z
    loss_small, gw, _ = dv.loss_and_gradients(w, b, Z, m.center, m.activation, "soft_boundary", 0.1, 1e-12)
    loss_big, gw2, _ = dv.loss_and_gradients(w, b, Z, m.center, m.activation, "soft_boundary", 0.1, 1e6)
    assert all(np.abs(g).max() > 0 for g in gw) and all(np.abs(g).max() == 0 for g in gw2)                                 # große Kugel: kein Punkt außerhalb, kein Gradient aus dem Hinge
    assert loss_big == pytest.approx(1e6)


def test_zero_epochs_is_the_untrained_network():
    X, y = _data()
    m = dv.fit_deep_svdd(X, n_epochs=0, seed=3)
    assert len(m.loss_history) == 0 and np.array_equal(m.snapshots[0], m.outputs) and m.n_epochs == 0
    assert _auc(dv.score(m), y) > 0.85                                                           # das zufällige Netz trennt weit entfernte Anomalien schon


def test_deterministic_for_a_seed_and_different_for_another():
    X, _ = _data()
    a = dv.fit_deep_svdd(X, n_epochs=40, seed=5)
    b = dv.fit_deep_svdd(X, n_epochs=40, seed=5)
    c = dv.fit_deep_svdd(X, n_epochs=40, seed=6)
    assert np.array_equal(a.outputs, b.outputs) and np.array_equal(a.loss_history, b.loss_history)
    assert not np.allclose(a.outputs, c.outputs) and not np.allclose(a.center, c.center)


# --- Skalierung, Bias und Kollaps ----------------------------------------------------------------------------------------------------------


def test_centered_inputs_put_the_normal_tours_on_the_origin_that_a_bias_free_network_maps_to_zero():
    X, _ = _data()
    m = dv.fit_deep_svdd(X, n_epochs=80, seed=1, scaling="standard")
    origin = dv.transform(m, X.mean(axis=0, keepdims=True))
    assert np.array_equal(origin, np.zeros_like(origin))                                         # der Mittelpunkt der Daten landet immer auf 0
    assert float(((origin - m.center) ** 2).sum()) == pytest.approx(float((m.center ** 2).sum())) and float((m.center ** 2).sum()) >= m.n_out * dv.CENTER_EPS ** 2 - 1e-12
    mm = dv.fit_deep_svdd(X, n_epochs=10, seed=1, scaling="minmax")
    assert not np.allclose(dv.transform(mm, X.mean(axis=0, keepdims=True)), 0.0)                 # Min-Max: der Mittelpunkt ist nicht der Ursprung


def test_bias_terms_let_the_outputs_contract_further_than_the_bias_free_network():
    X, _ = _data()
    free = dv.fit_deep_svdd(X, n_epochs=400, lr=0.01, seed=1, bias=False)
    with_bias = dv.fit_deep_svdd(X, n_epochs=400, lr=0.01, seed=1, bias=True)
    start = float(np.sqrt(free.snapshots[0].var(axis=0).mean()))
    assert with_bias.bias and dv.output_spread(with_bias) < 0.2 * start and dv.output_spread(with_bias) < dv.output_spread(free) * 1.0 + 1e-12
    assert all(np.isfinite(b).all() for b in with_bias.biases)


def test_invalid_arguments_are_rejected():
    X, _ = _data(n=30)
    for kw in ({"activation": "sigmoid"}, {"variant": "x"}, {"nu": 0.0}, {"nu": 1.5}, {"depth": -1}, {"n_out": 0}, {"scaling": "z"}):
        with pytest.raises(ValueError):
            dv.fit_deep_svdd(X, n_epochs=2, **kw)


# --- Kopierte Komponenten der Vorgänger -----------------------------------------------------------------------------------------------------


def test_one_class_svm_still_matches_scikit_learn():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((120, 5))
    X[:10] += 4
    g = svm.gamma_scale(X)
    f = svm.fit_ocsvm(X, 0.1, g, tol=1e-9)
    m = OneClassSVM(kernel="rbf", gamma=g, nu=0.1, tol=1e-9).fit(X)
    Z = rng.standard_normal((30, 5))
    assert np.abs(svm.decision_function(f, Z) - m.decision_function(Z)).max() < 1e-6


def test_other_copied_components_still_behave_like_their_predecessors():
    rng = np.random.default_rng(8)
    X = np.concatenate([rng.standard_normal((200, 3)), 7 + rng.standard_normal((10, 3))])
    y = np.arange(210) >= 200
    f = lof.fit_lof(X, 20)
    assert f.lof[y].min() > 1.5 and np.median(f.lof[~y]) < 1.2
    forest = isf.fit_forest(X, 100, 128, 0)
    sc = isf.score_from_paths(isf.path_lengths(forest, X), forest.psi)
    assert sc[y].min() > sc[~y].mean() and (sc >= 0).all() and (sc <= 1).all()
    r = alg.fit_mcd(X, 0.75, 0)
    assert r.d2[y].min() > alg.threshold(r, 0.975)
    assert alg.chi2_ppf(0.975, 12) == pytest.approx(chi2.ppf(0.975, 12), rel=1e-6)
    left, right = ecod.tail_probabilities(np.array([[1.0], [2.0], [2.0], [3.0], [10.0]]))
    assert np.allclose(left[:, 0], [0.2, 0.6, 0.6, 0.8, 1.0]) and np.allclose(right[:, 0], [1.0, 0.8, 0.8, 0.4, 0.2])
