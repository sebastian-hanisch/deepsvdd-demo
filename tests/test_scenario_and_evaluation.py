"""Szenario (identisch zu den Vorgänger-Demos, dazu der Korrelationsbruch), Kennzahlen, Analyse und Schwellen für vier Detektoren und die Bezugsdetektoren, Sweeps und Tabellen, Urteil."""

import numpy as np
import pytest

import dsvdd_algorithm as dv
import dsvdd_constants as C
import dsvdd_ee_algorithm as alg
import dsvdd_evaluation as ev
import dsvdd_isolation_forest as isf
import dsvdd_lof as lof
import dsvdd_ocsvm as svm

# Zeilensummen der ersten acht Zeilen der PCA-Demo (dieselben normalen Zeilen wie in den Vorgänger-Demos): permutationsinvariant, eingefroren
PCA_ROW_SUMS = [39967.27507413389, 42083.657538741994, 40172.02304566072, 44766.540605465496, 45034.44402326185, 55757.45917619934, 50727.55911151846, 43923.05872324106]


# --- Szenario ------------------------------------------------------------------------------------------------------------------------


def test_normal_rows_equal_the_pca_and_the_predecessor_rows():
    ds = ev.make_dataset(contamination=1)
    assert not ds.anomaly[:8].any() and ds.X.shape == (300, 12)
    assert np.allclose(ds.X[:8].sum(axis=1), PCA_ROW_SUMS, rtol=1e-12)


def test_default_dataset_is_frozen_at_the_predecessor_values():
    ds = ev.make_dataset()
    assert float(ds.X.sum()) == pytest.approx(13396385.215115668, rel=1e-12) and int(ds.anomaly.sum()) == 30 and ds.X[0, 0] == pytest.approx(19701.54827513291, rel=1e-12)


def test_noise_features_are_appended_and_change_nothing_else():
    base = ev.make_dataset()
    for n_noise in (1, 10, 40):
        ds = ev.make_dataset(n_noise=n_noise)
        assert ds.X.shape == (300, 12 + n_noise) and np.array_equal(ds.X[:, :12], base.X) and np.array_equal(ds.anomaly, base.anomaly)
        assert len(ds.names) == 12 + n_noise and ds.names[-1] == f"Rauschmerkmal {n_noise}" and ds.n_noise == n_noise
    assert np.array_equal(ev.make_dataset(p=30, n_noise=5).X[:, :30], ev.make_dataset(p=30).X)


@pytest.mark.parametrize("n,pct,expected", [(300, 10, 30), (300, 1, 3), (20, 1, 1), (30, 10, 3), (600, 45, 270)])
def test_contamination_is_exact_with_at_least_one_anomaly(n, pct, expected):
    assert int(ev.make_dataset(n=n, contamination=pct).anomaly.sum()) == expected


def test_gap_needs_two_modes_and_kinds_have_their_geometry():
    assert ev.make_dataset(kind="gap").kind == "scattered"
    gap = ev.make_dataset(n_modes=2, kind="gap")
    assert np.abs(gap.z[gap.anomaly]).max() < 1.2
    clu = ev.make_dataset(kind="cluster", contamination=20)
    assert np.linalg.norm(clu.z[clu.anomaly].mean(axis=0) - 6.0 * np.array([np.cos(C.CLUSTER_ANGLE), np.sin(C.CLUSTER_ANGLE)])) < 0.15


def test_decorrelated_anomalies_keep_the_marginals_and_lose_the_dependence():
    base = ev.make_dataset()
    ds = ev.make_dataset(kind="decorrelated", contamination=20)
    assert ds.kind == "decorrelated" and int(ds.anomaly.sum()) == 60 and ds.X.shape == base.X.shape
    normal = ds.X[~ds.anomaly]
    anom = ds.X[ds.anomaly]
    for j in range(ds.p):
        assert set(np.round(anom[:, j], 9)) <= set(np.round(normal[:, j], 9))                          # jede Spalte stammt aus den Normalen
    assert np.array_equal(normal, ev.make_dataset(kind="scattered", contamination=20).X[~ev.make_dataset(kind="scattered", contamination=20).anomaly])
    ds = ev.make_dataset(kind="decorrelated", contamination=40, n=600)
    c_norm = np.corrcoef(ds.X[~ds.anomaly].T)
    c_anom = np.corrcoef(ds.X[ds.anomaly].T)
    off = ~np.eye(ds.p, dtype=bool)
    assert np.abs(c_norm[off]).mean() > 0.4 and np.abs(c_anom[off]).mean() < 0.15                       # Abhängigkeit zerstört
    for j in range(ds.p):                                                                                 # gleiche Randverteilung: Quantile der Anomalien im Bereich der Normalen
        q_n, q_a = np.quantile(ds.X[~ds.anomaly][:, j], [0.05, 0.5, 0.95]), np.quantile(ds.X[ds.anomaly][:, j], [0.05, 0.5, 0.95])
        assert np.all(np.abs(q_n - q_a) < 0.35 * (q_n[2] - q_n[0]))


def test_decorrelated_is_drawn_last_so_other_kinds_and_defaults_stay_bit_identical():
    a = ev.make_dataset(seed=11, kind="scattered")
    b = ev.make_dataset(seed=11, kind="decorrelated")
    assert np.array_equal(a.X[~a.anomaly], b.X[~b.anomaly]) and np.array_equal(a.anomaly, b.anomaly)
    assert not np.array_equal(a.X[a.anomaly], b.X[b.anomaly])


def test_dataset_is_deterministic():
    assert np.array_equal(ev.make_dataset(seed=3, n_noise=4, kind="decorrelated").X, ev.make_dataset(seed=3, n_noise=4, kind="decorrelated").X)


# --- Kennzahlen: Handinstanzen ------------------------------------------------------------------------------------------------------------


def test_roc_auc_and_average_precision_hand_instances():
    assert ev.roc_auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert ev.roc_auc(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1], bool)) == 0.5
    assert ev.roc_auc(np.array([1, 4, 2, 3.0]), np.array([0, 1, 1, 0], bool)) == pytest.approx(0.75)
    assert np.isnan(ev.roc_auc(np.array([1.0, 2.0]), np.array([False, False])))
    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(200), rng.random(200) < 0.3
    assert ev.roc_auc(s, y) == pytest.approx(np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]))
    assert ev.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 0, 1, 0], bool)) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_curve_and_flag_metrics():
    rng = np.random.default_rng(1)
    s, y = rng.standard_normal(300), rng.random(300) < 0.2
    fpr, tpr = ev.roc_curve(s, y)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0) and np.trapezoid(tpr, fpr) == pytest.approx(ev.roc_auc(s, y), abs=1e-9)
    m = ev.flag_metrics(np.array([1, 1, 0, 0, 1, 0], bool), np.array([1, 0, 1, 0, 0, 0], bool))
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5 and m["false_alarm"] == pytest.approx(0.5) and m["f1"] == pytest.approx(0.4)


def test_lof_k_is_twenty_but_at_most_half_the_tours():
    assert [ev.lof_k(n) for n in (8, 20, 30, 100, 300, 600)] == [4, 10, 15, 20, 20, 20]


# --- Analyse und Schwellen ----------------------------------------------------------------------------------------------------------------


def _params(**kw):
    p = {**ev.DEFAULT_DATA, **kw}
    return (p["n"], p["p"], p["n_noise"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], 7)


def test_analysis_fields_and_consistency():
    a = ev.analyse_for(_params())
    assert set(a.scores) == set(ev.DETECTORS) | set(ev.EXTRAS) and set(a.oracle_f1) == set(ev.DETECTORS) | set(ev.EXTRAS) and set(a.flags) == set(a.values) - set(ev.EXTRAS) == set(ev.DETECTORS)
    assert a.p_total == 12 and a.forest_if.psi == 256 and a.robust.h == 156 and a.net.n_epochs == 100 and a.net.width == 32 and a.net.depth == 2 and a.net.n_out == 8 and not a.net.bias
    assert np.allclose(a.values["dsvdd"], dv.distances2(a.net)) and np.allclose(a.values["untrained"], ((a.net.snapshots[0] - a.net.center) ** 2).sum(axis=1))
    Z = ev.standardise(a.ds.X)
    assert np.allclose(a.values["ocsvm"], -svm.decision_function(a.ocsvm_fit, Z)) and a.ocsvm_fit.gamma == pytest.approx(1.0 / 12) and np.allclose(a.values["lof"], lof.fit_lof(Z, 20).lof)
    assert np.allclose(a.values["iforest"], isf.score_from_paths(isf.path_lengths(a.forest_if, a.ds.X), a.forest_if.psi))
    for d in ev.DETECTORS:
        assert a.scores[d]["n_flagged"] == int(a.flags[d].sum()) and 0 <= a.oracle_f1[d] <= 1
    assert a.scores["dsvdd"]["threshold"] == pytest.approx(a.net.r2) and (a.flags["dsvdd"] == (a.values["dsvdd"] > a.net.r2)).all()
    assert a.scores["ocsvm"]["threshold"] == C.FLAG_EPS and a.scores["iforest"]["threshold"] == 0.5 and a.scores["robust"]["threshold"] == pytest.approx(alg.threshold(a.robust, 0.975))
    assert a.scores["dsvdd"]["n_flagged"] / a.ds.n <= 0.1 + 1e-9 and a.seconds["dsvdd"] > 0 and a.spread == pytest.approx(dv.output_spread(a.net))


def test_share_threshold_flags_exactly_the_assumed_share_for_all_four_detectors():
    for share in (2, 10, 40):
        a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=share))
        for d in ev.DETECTORS:
            assert int(a.flags[d].sum()) == max(1, round(share / 100 * 300))
            assert a.flags[d][np.argsort(-a.values[d])[:3]].all()
    a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=10))
    assert all(a.scores[d]["f1"] == pytest.approx(a.oracle_f1[d]) for d in ev.DETECTORS)                    # 10 % = wahrer Anteil


def test_net_settings_move_only_deep_svdd_and_its_untrained_reference():
    base = ev.analyse_for(_params())
    for st in (ev.Settings(width=8), ev.Settings(depth=1), ev.Settings(n_out=4), ev.Settings(epochs=300), ev.Settings(activation="relu"), ev.Settings(bias=True), ev.Settings(scaling="standard"), ev.Settings(net_seed=3),
               ev.Settings(variant="soft_boundary")):
        b = ev.analyse_for(_params(), st)
        assert b.scores["dsvdd"] != base.scores["dsvdd"] and all(b.scores[d] == base.scores[d] for d in ("ocsvm", "iforest", "robust", "lof", "ecod", "ocsvm_wide"))
    n = ev.analyse_for(_params(), ev.Settings(nu=0.3))
    assert n.scores["dsvdd"]["auc"] == base.scores["dsvdd"]["auc"] and n.scores["dsvdd"]["n_flagged"] > base.scores["dsvdd"]["n_flagged"] and n.scores["ocsvm"] == base.scores["ocsvm"]
    e = ev.analyse_for(_params(), ev.Settings(epochs=300))
    assert np.array_equal(e.values["untrained"], base.values["untrained"])                                   # dasselbe untrainierte Netz und Zentrum, unabhängig von den Epochen


def test_zero_epochs_makes_the_trained_and_untrained_detector_identical():
    a = ev.analyse_for(_params(), ev.Settings(epochs=0))
    assert np.array_equal(a.values["dsvdd"], a.values["untrained"]) and a.scores["dsvdd"]["auc"] == a.scores["untrained"]["auc"]


def test_chi2_quantile_moves_only_the_robust_detector():
    base = ev.analyse_for(_params())
    q = ev.analyse_for(_params(), ev.Settings(quantile=0.999))
    assert q.scores["dsvdd"] == base.scores["dsvdd"] and q.scores["robust"]["false_alarm"] <= base.scores["robust"]["false_alarm"] and q.scores["robust"]["auc"] == base.scores["robust"]["auc"]


def test_forest_seed_is_separate_from_the_data_and_net_seeds_and_deep_svdd_is_deterministic():
    a, b = ev.analyse_for(_params()), ev.analyse_for(_params(), ev.Settings(start=5))
    assert np.array_equal(a.ds.X, b.ds.X) and a.values["iforest"].tolist() != b.values["iforest"].tolist() and np.array_equal(a.values["dsvdd"], b.values["dsvdd"])
    c = ev.analyse_for(_params(), ev.Settings(net_seed=1))
    assert np.array_equal(a.ds.X, c.ds.X) and not np.array_equal(a.values["dsvdd"], c.values["dsvdd"]) and np.array_equal(a.values["iforest"], c.values["iforest"])


def test_noise_features_are_used_and_few_tours_many_features_work():
    b = ev.analyse_for(_params(n_noise=10))
    assert b.ds.X.shape[1] == 22 and b.p_total == 22 and b.ocsvm_fit.gamma == pytest.approx(1.0 / 22)
    c = ev.analyse_for(_params(n=20, p=30))
    assert 0 <= c.scores["dsvdd"]["auc"] <= 1 and c.forest_if.psi == 20 and np.isfinite(c.net.loss_history).all()


def test_sweep_rows_labels_and_the_setting_sweeps():
    rows = ev.sweep("contamination", values=(5, 10))
    assert [r["x"] for r in rows] == [5, 10]
    for r in rows:
        for d in ev.DETECTORS + ev.EXTRAS:
            assert r[f"{d}_auc_min"] <= r[f"{d}_auc"] <= r[f"{d}_auc_max"] and r[f"{d}_auc_std"] >= 0 and f"{d}_oracle_f1" in r
        assert "dsvdd_spread" in r and "dsvdd_seconds" in r
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    e = ev.sweep("epochs", values=(0, 300))
    assert e[0]["dsvdd_auc"] > e[1]["dsvdd_auc"] and e[0]["dsvdd_spread"] > e[1]["dsvdd_spread"] and e[0]["iforest_auc"] == e[1]["iforest_auc"] and e[0]["untrained_auc"] == e[1]["untrained_auc"]
    nu = ev.sweep("nu", values=(0.05, 0.3))
    assert nu[0]["dsvdd_recall"] < nu[1]["dsvdd_recall"] and nu[0]["dsvdd_auc"] == nu[1]["dsvdd_auc"]
    n = ev.sweep("n", values=(20,))
    assert n[0]["lof_auc"] > 0.9                                                                        # k = 20 würde bei n = 20 klemmen; der Vergleich begrenzt k auf n / 2


def test_settings_defaults_agree_with_the_constants():
    s = ev.Settings()
    assert (s.variant, s.width, s.depth, s.n_out, s.epochs, s.nu, s.activation, s.bias, s.scaling, s.net_seed) == (C.DEFAULT_VARIANT, C.DEFAULT_WIDTH, C.DEFAULT_DEPTH, C.DEFAULT_N_OUT, C.DEFAULT_EPOCHS, C.DEFAULT_NU,
                                                                                                                    C.DEFAULT_ACTIVATION, C.DEFAULT_BIAS, C.DEFAULT_SCALING, C.DEFAULT_NET_SEED)
    assert (s.threshold_kind, s.quantile, s.share) == (C.DEFAULT_THRESHOLD_KIND, C.DEFAULT_QUANTILE, C.DEFAULT_SHARE) and set(ev.DEFAULT_DATA) == set(ev.DATA_KEYS) and C.SWEEP_SEEDS == tuple(range(100000, 100005))


def test_score_maps_show_the_learned_sphere():
    ds = ev.make_dataset(p=2)
    X2 = ds.X[~ds.anomaly]
    xs, ys, F, S, outside = ev.score_maps(X2, ev.Settings(epochs=30), grid=40)
    assert F.shape == S.shape == (40, 40) and len(xs) == len(ys) == 40 and outside.shape == (len(X2),)
    assert (F > 0).any() and (F < 0).any() and 0 < outside.mean() <= 0.1 + 1e-9                          # ein Anteil ν der Trainingstouren liegt außerhalb


# --- Experimente: Form der Tabellen ----------------------------------------------------------------------------------------------------------


def test_scenario_curve_seed_collapse_width_cost_and_threshold_tables_have_their_documented_shape():
    assert len(ev.SCENARIOS) == 8
    sc = ev.scenario_table(scenarios=ev.SCENARIOS[:2])
    assert [r["scenario"] for r in sc] == [s[0] for s in ev.SCENARIOS[:2]] and all(f"{d}_auc" in sc[0] for d in ev.DETECTORS + ev.EXTRAS)
    cv = ev.epoch_curves(scenarios=ev.SCENARIOS[:1], net_seeds=(0,), max_epochs=60)
    assert cv[0]["epochs"][0] == 0 and cv[0]["epochs"][-1] == 60 and len(cv[0]["auc"]) == len(cv[0]["spread"]) == len(cv[0]["epochs"]) and cv[0]["spread"][0] > cv[0]["spread"][-1]
    st = ev.seed_table(epochs_list=(0, 30), net_seeds=(0, 1))
    assert [r["epochs"] for r in st] == [0, 30] and all(r["auc_min"] <= r["auc"] <= r["auc_max"] and r["auc_std"] >= 0 for r in st)
    cl = ev.collapse_table(max_epochs=60, net_seeds=(0,))
    assert [(r["bias"], r["scaling"]) for r in cl] == [(False, "minmax"), (False, "standard"), (True, "minmax"), (True, "standard")] and all(r["epochs"][-1] == 60 for r in cl)
    wt = ev.width_table(widths=(4, 32), net_seeds=(0,), max_epochs=60)
    assert [r["width"] for r in wt] == [4, 32] and wt[0]["epochs"][0] == 0
    ct = ev.cost_table()["times"]
    assert [(t["n"], t["p"]) for t in ct] == [(100, 12), (300, 12), (600, 12), (100, 52), (300, 52), (600, 52)] and all(t["dsvdd_1000"] > t["dsvdd"] for t in ct)
    tt = ev.threshold_table(ev.Settings())
    assert [r["x"] for r in tt["cutoff"]] == list(ev.SWEEP_VALUES["nu"]) and [r["factor"] for r in tt["wrong_share"]] == [0.5, 1.0, 2.0] and [r["x"] for r in tt["wrong_share"]] == [5, 10, 20]


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------


def _verdict(**kw):
    settings = ev.Settings(**{k: kw.pop(k) for k in ("epochs", "activation", "scaling", "nu", "net_seed") if k in kw})
    return ev.verdict(ev.analyse_for(_params(**kw), settings))


def test_verdict_codes_for_the_presets_and_edge_cases():
    assert _verdict()[1] in ("training_hurts", "comparable")
    assert _verdict(epochs=0)[:2] == ("success", "comparable")
    assert _verdict(activation="relu", epochs=1000)[:2] == ("warning", "training_hurts")
    assert _verdict(scaling="standard", epochs=300)[:2] == ("warning", "training_hurts")
    assert _verdict(contamination=45)[:2] == ("warning", "training_hurts")
    assert _verdict(kind="decorrelated")[:2] == ("warning", "others_win")
    assert _verdict(curvature=1.0)[1] in ("comparable", "training_hurts")


def test_verdict_data_carries_the_numbers_the_messages_use():
    kind, code, data = _verdict()
    for key in ("dsvdd_auc", "untrained_auc", "untrained_f1", "ocsvm_auc", "iforest_auc", "robust_auc", "ocsvm_f1", "iforest_f1", "robust_f1", "best_other_auc", "dsvdd_recall", "dsvdd_f1", "oracle_dsvdd", "spread",
                "epochs", "nu", "contamination", "n_anomalies", "n", "p", "kind"):
        assert key in data
    assert data["p"] == 12 and data["n_anomalies"] == 30 and data["contamination"] == pytest.approx(10.0) and data["epochs"] == 100


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["dsvdd"] < 3.0 and a.seconds["ocsvm"] < 1.5 and a.seconds["lof"] < 1.5 and a.seconds["iforest"] < 4.0 and a.seconds["robust"] < 10.0
