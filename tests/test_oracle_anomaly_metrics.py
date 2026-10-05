"""Unabhängiges Orakel für die Kennzahlen der Auswertung: AUC, mittlere Präzision, ROC-Kurve und Flag-Kennzahlen gegen scikit-learn.

Gerade bei Bindungen (gleiche Score-Werte: Plateaus, ganzzahlige Werte, alle gleich) hing die mittlere Präzision früher von der Zeilenreihenfolge ab und wich von
scikit-learn ab (z. B. 1,0 statt 0,25); gleiche Werte bilden jetzt EINE Schwelle (auch bei der ROC-Kurve: Diagonale statt Treppe).
"""

import numpy as np
import pytest

import dsvdd_evaluation as ev

metrics = pytest.importorskip("sklearn.metrics")


def _cases(n_cases=400, seed=1):
    rng = np.random.default_rng(seed)
    for i in range(n_cases):
        n = int(rng.integers(2, 40))
        kind = i % 5
        if kind == 0:
            s = rng.integers(0, 4, n).astype(float)             # ganzzahlig: viele Bindungen
        elif kind == 1:
            s = np.round(rng.normal(size=n), 1)                 # gerundet: Plateaus
        elif kind == 2:
            s = np.full(n, 0.7)                                 # alle gleich
        elif kind == 3:
            s = rng.normal(size=n)                              # keine Bindungen
        else:
            s = np.repeat(rng.normal(size=3), 20)[:n]           # drei große Plateaus
        pos = rng.random(n) < rng.choice([0.1, 0.3, 0.5, 0.8])
        if kind == 2 and i % 10 == 2:
            pos[:] = False
            pos[int(rng.integers(n))] = True                    # genau ein Treffer
        yield s, pos


def test_average_precision_equals_scikit_learn_also_with_ties():
    for s, y in _cases():
        if y.any():
            assert ev.average_precision(s, y) == pytest.approx(metrics.average_precision_score(y, s), abs=1e-12)
        else:
            assert np.isnan(ev.average_precision(s, y))


def test_average_precision_does_not_depend_on_row_order():
    rng = np.random.default_rng(5)
    for s, y in _cases(60, seed=2):
        if not y.any():
            continue
        perm = rng.permutation(len(s))
        assert ev.average_precision(s[perm], y[perm]) == pytest.approx(ev.average_precision(s, y), abs=1e-12)


def test_average_precision_tie_hand_instance():
    # Alle Werte gleich: eine einzige Schwelle, Präzision = Anteil der Treffer (nicht 1,0, nur weil der Treffer zufällig vorn steht).
    assert ev.average_precision(np.ones(4), np.array([True, False, False, False])) == pytest.approx(0.25)
    assert ev.average_precision(np.ones(4), np.array([False, False, False, True])) == pytest.approx(0.25)


def test_roc_auc_and_curve_equal_scikit_learn_also_with_ties():
    for s, y in _cases():
        if not (y.any() and (~y).any()):
            continue
        auc = metrics.roc_auc_score(y, s)
        assert ev.roc_auc(s, y) == pytest.approx(auc, abs=1e-12)
        fpr, tpr = ev.roc_curve(s, y)
        ref_f, ref_t, _ = metrics.roc_curve(y, s, drop_intermediate=False)
        assert set(zip(np.round(fpr, 12), np.round(tpr, 12))) == set(zip(np.round(ref_f, 12), np.round(ref_t, 12)))
        assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
        assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
        assert np.trapezoid(tpr, fpr) == pytest.approx(auc, abs=1e-12)


def test_flag_metrics_equal_scikit_learn():
    rng = np.random.default_rng(3)
    for s, y in _cases(200, seed=4):
        flagged = rng.random(len(y)) < 0.4
        if not y.any() or not flagged.any():
            continue
        p, r, f, _ = metrics.precision_recall_fscore_support(y, flagged, average="binary", zero_division=0)
        m = ev.flag_metrics(flagged, y)
        assert (m["precision"], m["recall"], m["f1"]) == pytest.approx((p, r, f), abs=1e-12)
        assert m["n_flagged"] == int(flagged.sum())
