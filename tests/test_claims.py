"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel). Das untrainierte Netz und die anderen Detektoren sind fast deterministisch (Toleranz ±0.02);
trainierte Netze werden über viele Epochen gerechnet und tragen Rundungsunterschiede zwischen numpy-/BLAS-Versionen mit: dort sind die Bänder weiter (±0.04 AUC, ±0.07 F1, ±0.06 bei langem Training).
Positive UND negative Aussagen: wo Deep SVDD gegen das untrainierte Netz, die One-Class SVM, den Isolation Forest oder die Wurzel verliert, steht das hier ebenso als Test wie dort, wo es gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import dsvdd_constants as C
import dsvdd_evaluation as ev

TOL = 0.02
TRAINED = 0.04                     # AUC nach 100 Epochen
TRAINED_F1 = 0.07
LONG = 0.06                        # nach 300 oder mehr Epochen und bei schlecht konditionierten Konfigurationen
STANDARD = ev.Settings()


def S(**kw):
    return ev.Settings(**kw)


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev._analyse_seed(s, settings, dict(items)) for s in C.SWEEP_SEEDS)


def runs(settings=STANDARD, **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=STANDARD, **kw):
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def oracle(det, settings=STANDARD, **kw):
    return float(np.mean([a.oracle_f1[det] for a in runs(settings, **kw)]))


def spread(settings=STANDARD, **kw):
    return float(np.mean([a.spread for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


@lru_cache(maxsize=None)
def curves():
    return {r["scenario"]: dict(zip(r["epochs"], r["auc"])) for r in ev.epoch_curves()}


@lru_cache(maxsize=None)
def spreads():
    return {r["scenario"]: dict(zip(r["epochs"], r["spread"])) for r in ev.epoch_curves(scenarios=ev.SCENARIOS[:1])}


@lru_cache(maxsize=None)
def seeds():
    return {r["epochs"]: r for r in ev.seed_table()}


@lru_cache(maxsize=None)
def collapse():
    return {(r["bias"], r["scaling"]): dict(zip(r["epochs"], zip(r["auc"], r["spread"]))) for r in ev.collapse_table()}


@lru_cache(maxsize=None)
def width_tab():
    return {r["width"]: dict(zip(r["epochs"], r["auc"])) for r in ev.width_table()}


# --- Das untrainierte Netz ist der Bezug: Sweep über die Epochen ----------------------------------------------------------------------------


@pytest.mark.parametrize("ep,auc,f1,spr", [(0, 0.995, 0.90, 0.081), (10, 0.99, 0.86, 0.048), (30, 0.95, 0.77, 0.037), (100, 0.92, 0.62, 0.019), (300, 0.91, 0.61, 0.010), (1000, 0.89, 0.63, 0.003)])
def test_epochs_sweep_the_ranking_gets_worse_while_the_sphere_shrinks(ep, auc, f1, spr):
    s = S(epochs=ep)
    near(m("dsvdd", "auc", s), auc, 0.012 if ep <= 10 else (TRAINED if ep <= 100 else LONG))
    near(m("dsvdd", "f1", s), f1, 0.03 if ep <= 10 else (TRAINED_F1 if ep <= 100 else 0.09))
    near(spread(s), spr, 0.02)


def test_untrained_reference_and_the_other_detectors_in_the_standard_case():
    near(m("untrained", "auc"), 0.995, 0.01)
    near(m("untrained", "f1"), 0.90, 0.03)
    near(m("untrained", "recall"), 0.89, 0.03)
    near(m("untrained", "false_alarm"), 0.009, 0.01)
    near(m("iforest", "auc"), 1.00, 0.01)
    near(m("iforest", "f1"), 0.96)
    near(m("ocsvm", "auc"), 0.95, 0.012)
    near(m("ocsvm", "f1"), 0.67)
    near(m("ocsvm_wide", "auc"), 1.00, 0.01)
    near(m("ocsvm_wide", "f1"), 0.95)
    near(m("robust", "f1"), 0.84)
    near(oracle("dsvdd"), 0.63, 0.08)
    near(oracle("untrained"), 0.90, 0.03)


def test_training_hurts_at_every_length_in_the_standard_case():
    for ep in (30, 100, 300, 1000):
        assert m("untrained", "auc") - m("dsvdd", "auc", S(epochs=ep)) >= 0.03


# --- Seitenleiste: Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil -------------------------------------------------


@pytest.mark.parametrize("n,auc,un,f1", [(20, 0.72, 0.99, 0.13), (30, 0.82, 0.97, 0.48), (50, 0.91, 0.99, 0.71), (100, 0.94, 1.00, 0.74), (200, 0.94, 0.99, 0.70), (400, 0.92, 1.00, 0.70), (600, 0.93, 1.00, 0.70)])
def test_tours_sweep(n, auc, un, f1):
    near(m("dsvdd", "auc", n=n), auc, TRAINED + 0.02)
    near(m("untrained", "auc", n=n), un, 0.015)
    near(m("dsvdd", "f1", n=n), f1, 0.10)


@pytest.mark.parametrize("p,auc,un", [(2, 0.91, 0.96), (5, 0.92, 1.00), (8, 0.94, 0.99), (12, 0.92, 1.00), (20, 0.82, 0.99), (30, 0.92, 1.00)])
def test_features_sweep(p, auc, un):
    near(m("dsvdd", "auc", p=p), auc, 0.06)
    near(m("untrained", "auc", p=p), un, 0.015)


@pytest.mark.parametrize("nn,auc,un", [(0, 0.92, 0.995), (5, 0.86, 0.97), (10, 0.87, 0.90), (20, 0.73, 0.77), (30, 0.72, 0.83), (40, 0.71, 0.73)])
def test_noise_features_sweep(nn, auc, un):
    near(m("dsvdd", "auc", n_noise=nn), auc, 0.06)
    near(m("untrained", "auc", n_noise=nn), un, 0.03)


def test_noise_features_the_others():
    near(m("ocsvm_wide", "auc", n_noise=40), 0.99, 0.015)
    near(m("iforest", "auc", n_noise=40), 0.99, 0.015)
    near(m("robust", "auc", n_noise=40), 0.88, 0.015)
    near(m("ocsvm_wide", "f1", n_noise=40), 0.92)


@pytest.mark.parametrize("modes,auc,un,robust", [(1, 0.92, 0.995, 1.00), (2, 0.94, 0.94, 0.95), (3, 0.87, 0.97, 0.85)])
def test_modes_sweep(modes, auc, un, robust):
    near(m("dsvdd", "auc", n_modes=modes), auc, 0.06)
    near(m("untrained", "auc", n_modes=modes), un, 0.02)
    near(m("robust", "auc", n_modes=modes), robust, 0.015)


@pytest.mark.parametrize("curv,auc,f1", [(0.0, 0.92, 0.62), (0.25, 0.96, 0.77), (0.5, 0.97, 0.78), (0.75, 0.98, 0.80), (1.0, 0.98, 0.83)])
def test_curvature_sweep(curv, auc, f1):
    near(m("dsvdd", "auc", curvature=curv), auc, TRAINED)
    near(m("dsvdd", "f1", curvature=curv), f1, 0.09)
    near(m("untrained", "auc", curvature=curv), 0.99, 0.015)


@pytest.mark.parametrize("noise,auc,f1", [(0.0, 0.99, 0.89), (0.25, 0.92, 0.62), (0.5, 0.88, 0.56), (0.75, 0.89, 0.59), (1.0, 0.91, 0.62)])
def test_measurement_noise_sweep(noise, auc, f1):
    near(m("dsvdd", "auc", noise=noise), auc, 0.06)
    near(m("dsvdd", "f1", noise=noise), f1, 0.12)


@pytest.mark.parametrize("c,auc", [(2, 0.97), (5, 0.95), (10, 0.92), (20, 0.89), (30, 0.86), (40, 0.84), (45, 0.83)])
def test_contamination_sweep_the_training_learns_the_anomalies(c, auc):
    near(m("dsvdd", "auc", contamination=c), auc, 0.06)
    assert m("untrained", "auc", contamination=c) >= 0.985


def test_contamination_the_others():
    near(m("ocsvm", "auc", contamination=45), 0.32, 0.02)
    near(m("ocsvm", "auc", contamination=2), 1.00, 0.01)
    near(m("ocsvm_wide", "auc", contamination=45), 1.00, 0.01)
    near(m("iforest", "auc", contamination=45), 1.00, 0.01)


# --- Art der Anomalien ---------------------------------------------------------------------------------------------------------------------


def test_dense_group_trained_network_is_below_chance():
    kw = dict(kind="cluster", contamination=30)
    near(m("dsvdd", "auc", **kw), 0.18, 0.10)
    assert m("dsvdd", "auc", **kw) < 0.5
    near(m("untrained", "auc", **kw), 0.85, 0.03)
    near(m("ocsvm", "auc", **kw), 0.74, 0.02)
    near(m("ocsvm_wide", "auc", **kw), 0.84, 0.02)
    near(m("iforest", "auc", **kw), 0.70, 0.02)
    near(m("robust", "auc", **kw), 0.49, 0.02)
    near(m("lof", "auc", **kw), 0.47, 0.02)
    near(m("ecod", "auc", **kw), 0.84, 0.02)


@pytest.mark.parametrize("nm,ds_auc,un,ocsvm,iforest,robust", [(2, 0.45, 0.02, 0.64, 0.54, 0.40), (3, 0.53, 0.03, 0.82, 0.27, 0.38)])
def test_gap_training_helps_but_stays_near_chance(nm, ds_auc, un, ocsvm, iforest, robust):
    kw = dict(kind="gap", n_modes=nm)
    near(m("dsvdd", "auc", **kw), ds_auc, 0.10)
    assert m("dsvdd", "auc", **kw) < 0.65
    near(m("untrained", "auc", **kw), un, 0.02)
    near(m("ocsvm", "auc", **kw), ocsvm, 0.02)
    near(m("iforest", "auc", **kw), iforest, 0.02)
    near(m("robust", "auc", **kw), robust, 0.02)


def test_correlation_break_the_training_helps():
    kw = dict(kind="decorrelated")
    near(m("dsvdd", "auc", **kw), 0.92, 0.06)
    near(m("dsvdd", "f1", **kw), 0.54, 0.12)
    near(m("untrained", "auc", **kw), 0.70, 0.03)
    near(m("ocsvm", "auc", **kw), 0.88, 0.02)
    near(m("iforest", "auc", **kw), 0.80, 0.02)
    near(m("ecod", "auc", **kw), 0.56, 0.03)
    near(m("lof", "auc", **kw), 0.99, 0.02)
    near(m("robust", "auc", **kw), 1.00, 0.02)


# --- Regler des Netzes ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ep,one,soft", [(0, 0.995, 0.995), (10, 0.99, 0.99), (30, 0.95, 0.97), (100, 0.92, 0.89), (300, 0.91, 0.83), (1000, 0.89, 0.83)])
def test_variant_sweep_soft_boundary_has_no_advantage(ep, one, soft):
    tol = 0.012 if ep <= 10 else (TRAINED if ep <= 100 else 0.07)
    near(m("dsvdd", "auc", S(epochs=ep)), one, tol)
    near(m("dsvdd", "auc", S(epochs=ep, variant="soft_boundary")), soft, tol + 0.01)


@pytest.mark.parametrize("w,auc", [(4, 0.98), (8, 0.98), (16, 0.95), (32, 0.92), (64, 0.88)])
def test_width_sweep_wider_networks_lose_more(w, auc):
    near(m("dsvdd", "auc", S(width=w)), auc, 0.05)
    assert m("untrained", "auc", S(width=w)) >= 0.985


@pytest.mark.parametrize("d,auc,un", [(0, 0.98, 1.00), (1, 0.92, 1.00), (2, 0.92, 0.99), (3, 0.95, 0.98)])
def test_depth_sweep(d, auc, un):
    near(m("dsvdd", "auc", S(depth=d)), auc, 0.05)
    near(m("untrained", "auc", S(depth=d)), un, 0.02)


@pytest.mark.parametrize("o,auc,un", [(1, 0.70, 0.66), (2, 0.79, 0.93), (4, 0.82, 0.96), (8, 0.92, 0.99), (16, 0.97, 0.98)])
def test_output_dimension_sweep(o, auc, un):
    near(m("dsvdd", "auc", S(n_out=o)), auc, 0.06)
    near(m("untrained", "auc", S(n_out=o)), un, 0.03)


@pytest.mark.parametrize("nu,f1,recall,fa", [(0.01, 0.13, 0.07, 0.000), (0.02, 0.29, 0.17, 0.000), (0.05, 0.59, 0.43, 0.004), (0.10, 0.62, 0.61, 0.039), (0.20, 0.55, 0.82, 0.127), (0.30, 0.44, 0.88, 0.232), (0.50, 0.32, 0.94, 0.447)])
def test_nu_sweep_the_threshold_marks_about_nu_of_the_tours(nu, f1, recall, fa):
    s = S(nu=nu)
    near(m("dsvdd", "f1", s), f1, 0.09)
    near(m("dsvdd", "recall", s), recall, 0.08)
    near(m("dsvdd", "false_alarm", s), fa, 0.03)
    near(m("dsvdd", "auc", s), 0.92, TRAINED)                                                # die Rangfolge ändert sich mit ν nicht


def test_nu_flags_at_most_nu_of_the_tours():
    for nu in (0.05, 0.1, 0.3):
        assert all(a.scores["dsvdd"]["n_flagged"] / a.ds.n <= nu + 1e-9 for a in runs(S(nu=nu)))


@pytest.mark.parametrize("ep,auc", [(0, 0.98), (10, 0.99), (30, 0.99), (100, 0.92), (300, 0.73), (1000, 0.49)])
def test_relu_is_good_early_and_falls_below_chance_late(ep, auc):
    near(m("dsvdd", "auc", S(activation="relu", epochs=ep)), auc, 0.02 if ep <= 30 else 0.10)
    assert m("dsvdd", "auc", S(activation="relu", epochs=1000)) < 0.65


@pytest.mark.parametrize("ep,with_bias,free", [(0, 0.995, 0.995), (10, 0.99, 0.99), (30, 0.95, 0.95), (100, 0.85, 0.92), (300, 0.87, 0.91), (1000, 0.90, 0.89)])
def test_bias_sweep(ep, with_bias, free):
    near(m("dsvdd", "auc", S(bias=True, epochs=ep)), with_bias, 0.012 if ep <= 10 else 0.08)
    near(m("dsvdd", "auc", S(epochs=ep)), free, 0.012 if ep <= 10 else 0.07)


@pytest.mark.parametrize("ep,auc", [(0, 0.95), (10, 0.89), (30, 0.65), (100, 0.58), (300, 0.48), (1000, 0.32)])
def test_centered_inputs_invert_the_ranking(ep, auc):
    near(m("dsvdd", "auc", S(scaling="standard", epochs=ep)), auc, 0.02 if ep == 0 else 0.10)
    assert m("dsvdd", "auc", S(scaling="standard", epochs=1000)) < 0.5


@pytest.mark.parametrize("q,f1", [(0.9, 0.67), (0.95, 0.77), (0.975, 0.84), (0.99, 0.89), (0.999, 0.90)])
def test_chi2_quantile_help_numbers(q, f1):
    near(m("robust", "f1", S(quantile=q)), f1)


@pytest.mark.parametrize("share,dsvdd,iforest,robust", [(2, 0.33, 0.33, 0.33), (5, 0.59, 0.67, 0.67), (10, 0.63, 0.97, 0.90), (20, 0.55, 0.67, 0.66), (40, 0.37, 0.40, 0.40)])
def test_assumed_share_costs_all_detectors(share, dsvdd, iforest, robust):
    s = S(threshold_kind="share", share=share)
    near(m("dsvdd", "f1", s), dsvdd, 0.08)
    near(m("iforest", "f1", s), iforest)
    near(m("robust", "f1", s), robust)


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_standard_and_untrained_presets():
    near(m("dsvdd", "auc"), 0.92, TRAINED)
    near(m("dsvdd", "f1"), 0.62, TRAINED_F1)
    near(m("untrained", "auc"), 0.995, 0.01)
    s0 = S(epochs=0)
    near(m("dsvdd", "auc", s0), 0.995, 0.01)
    near(m("dsvdd", "f1", s0), 0.90, 0.03)
    near(m("dsvdd", "recall", s0), 0.89, 0.03)
    near(m("dsvdd", "false_alarm", s0), 0.009, 0.01)


def test_relu_and_centered_presets():
    relu = S(activation="relu", epochs=1000)
    near(m("dsvdd", "auc", relu), 0.49, 0.12)
    near(m("dsvdd", "f1", relu), 0.13, 0.10)
    near(m("untrained", "auc", relu), 0.98, 0.02)
    near(spread(relu), 0.016, 0.02)
    cen = S(scaling="standard", epochs=300)
    near(m("dsvdd", "auc", cen), 0.48, 0.12)
    near(m("dsvdd", "f1", cen), 0.19, 0.10)
    near(m("untrained", "auc", cen), 0.95, 0.02)
    near(m("dsvdd", "auc", S(scaling="standard", epochs=1000)), 0.32, 0.12)


def test_curved_and_many_anomalies_presets():
    near(m("dsvdd", "auc", curvature=1.0), 0.98, TRAINED)
    near(m("dsvdd", "f1", curvature=1.0), 0.83, 0.09)
    near(m("dsvdd", "false_alarm", curvature=1.0), 0.017, 0.02)
    near(m("untrained", "auc", curvature=1.0), 0.99, 0.015)
    near(m("untrained", "f1", curvature=1.0), 0.90, 0.04)
    near(m("ocsvm", "auc", curvature=1.0), 0.98, 0.02)
    near(m("ocsvm", "f1", curvature=1.0), 0.85)
    near(m("ocsvm_wide", "auc", curvature=1.0), 0.79, 0.02)
    near(m("ocsvm_wide", "f1", curvature=1.0), 0.73)
    near(m("dsvdd", "f1", contamination=45), 0.34, 0.10)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario,ds_auc,un,ocsvm,wide,iforest,robust", [("Standardfall", 0.92, 0.995, 0.95, 1.00, 1.00, 1.00), ("gekrümmter Normalbereich", 0.98, 0.99, 0.98, 0.79, 1.00, 1.00),
                                                                           ("dichte Gruppe 30 %", 0.18, 0.85, 0.74, 0.84, 0.70, 0.49), ("Lücke, 2 Betriebsarten", 0.45, 0.02, 0.64, 0.02, 0.54, 0.40),
                                                                           ("Lücke, 3 Betriebsarten", 0.53, 0.03, 0.82, 0.00, 0.27, 0.38), ("Korrelationsbruch", 0.92, 0.70, 0.88, 0.69, 0.80, 1.00),
                                                                           ("45 % verstreut", 0.83, 0.995, 0.32, 1.00, 1.00, 0.87), ("40 Rauschmerkmale", 0.71, 0.73, 0.91, 0.99, 0.99, 0.88)])
def test_scenario_table_auc(scenario, ds_auc, un, ocsvm, wide, iforest, robust):
    kw = dict(dict(ev.SCENARIOS)[scenario])
    near(m("dsvdd", "auc", **kw), ds_auc, 0.10 if ds_auc < 0.6 else 0.06)
    near(m("untrained", "auc", **kw), un, 0.03)
    near(m("ocsvm", "auc", **kw), ocsvm, 0.02)
    near(m("ocsvm_wide", "auc", **kw), wide, 0.02)
    near(m("iforest", "auc", **kw), iforest, 0.02)
    near(m("robust", "auc", **kw), robust, 0.02)


def test_scenario_f1_statements():
    near(oracle("dsvdd"), 0.63, 0.08)
    near(oracle("untrained"), 0.90, 0.03)
    near(oracle("ocsvm_wide"), 0.95)


@pytest.mark.parametrize("scenario,start,end,at", [("Standardfall", 1.00, 0.88, {33: 0.94}), ("gekrümmter Normalbereich", 1.00, 0.95, {}), ("45 % verstreut", 0.99, 0.62, {33: 0.95}), ("40 Rauschmerkmale", 0.73, 0.65, {}),
                                                   ("Korrelationsbruch", 0.65, 0.79, {105: 0.91})])
def test_epoch_curves(scenario, start, end, at):
    c = curves()[scenario]
    near(c[0], start, 0.03)
    near(c[600], end, 0.08)
    for e, v in at.items():
        near(c[e], v, 0.06)


def test_epoch_curves_dense_group_and_gap():
    c = curves()["dichte Gruppe 30 %"]
    near(c[0], 0.80, 0.05)
    near(c[3], 0.27, 0.12)
    assert min(c.values()) < 0.25 and c[59] < 0.30
    for scen, at188 in (("Lücke, 2 Betriebsarten", 0.47), ("Lücke, 3 Betriebsarten", 0.59)):
        g = curves()[scen]
        assert g[0] < 0.10 and max(g.values()) < 0.7
        near(g[188], at188, 0.12)


def test_epoch_curve_best_epoch_depends_on_the_scenario():
    best = {s: max(c, key=c.get) for s, c in curves().items()}
    assert best["Standardfall"] <= 6 and best["Korrelationsbruch"] >= 59 and best["Lücke, 3 Betriebsarten"] >= 59


@pytest.mark.parametrize("ep,mean,std,mn,mx,f1", [(0, 0.99, 0.02, 0.91, 1.00, 0.90), (30, 0.95, 0.02, 0.90, 0.99, 0.75), (100, 0.91, 0.05, 0.76, 0.98, 0.65), (300, 0.89, 0.05, 0.76, 0.97, 0.61)])
def test_seed_table(ep, mean, std, mn, mx, f1):
    r = seeds()[ep]
    near(r["auc"], mean, 0.03)
    near(r["auc_std"], std, 0.03)
    near(r["auc_min"], mn, 0.06)
    near(r["auc_max"], mx, 0.03)
    near(r["f1"], f1, 0.08)


def test_seed_spread_grows_with_training():
    r = seeds()
    assert r[0]["auc_std"] < r[100]["auc_std"] and r[0]["auc_min"] > r[100]["auc_min"] + 0.05


def test_collapse_table_bias_and_scaling():
    cl = collapse()
    for cfg, spr_end in (((False, "minmax"), 0.004), ((True, "minmax"), 0.002), ((False, "standard"), 0.023), ((True, "standard"), 0.009)):
        assert cl[cfg][0][1] > 4 * cl[cfg][1000][1]                                             # in allen vier Konfigurationen ziehen sich die Ausgaben zusammen
        near(cl[cfg][1000][1], spr_end, 0.015)
    near(cl[(False, "minmax")][0][1], 0.082, 0.02)
    near(cl[(False, "minmax")][1000][0], 0.87, 0.07)
    near(cl[(True, "minmax")][1000][0], 0.90, 0.07)
    near(cl[(False, "standard")][0][0], 0.94, 0.03)
    near(cl[(False, "standard")][1000][0], 0.31, 0.12)
    near(cl[(True, "standard")][1000][0], 0.95, 0.06)
    assert min(v[0] for v in cl[(True, "standard")].values()) < 0.85
    near(cl[(True, "standard")][81][0], 0.76, 0.10)


def test_width_table():
    wt = width_tab()
    for w in (4, 8, 16, 32, 64):
        assert min(wt[w][0], wt[w][1]) >= 0.97
    for w, v in ((4, 0.96), (8, 0.97), (16, 0.92), (32, 0.90), (64, 0.87)):
        near(wt[w][105], v, 0.06)
    for w, v in ((4, 0.86), (8, 0.87), (16, 0.90), (32, 0.88), (64, 0.84)):
        near(wt[w][600], v, 0.07)
    assert max(wt[w][600] for w in wt) < wt[4][0] - 0.05


def test_wrong_share_table():
    tt = ev.threshold_table(STANDARD)
    by_factor = {r["factor"]: r for r in tt["wrong_share"]}
    near(by_factor[0.5]["dsvdd_f1"], 0.59, 0.08)
    near(by_factor[1.0]["dsvdd_f1"], 0.63, 0.08)
    near(by_factor[2.0]["dsvdd_f1"], 0.55, 0.08)
    near(by_factor[0.5]["iforest_f1"], 0.67)
    near(by_factor[1.0]["iforest_f1"], 0.97)
    near(by_factor[2.0]["iforest_f1"], 0.67)


def test_cost_table():
    times = {(t["n"], t["p"]): t for t in ev.cost_table()["times"]}
    for t in times.values():
        assert t["dsvdd"] < 0.5 and t["dsvdd_1000"] > 4 * t["dsvdd"] and t["dsvdd_1000"] < 3.0 and t["dsvdd"] > 1.5 * t["ocsvm"]          # größenordnungsmäßig: einige zehn ms, das Mehrfache der One-Class SVM
        assert t["dsvdd"] < t["iforest"] * 1.5 + 0.02
    assert times[(600, 12)]["dsvdd"] > times[(100, 12)]["dsvdd"]


# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------


def test_limits_table_numbers():
    near(m("untrained", "auc"), 0.995, 0.01)
    near(m("dsvdd", "auc"), 0.92, TRAINED)
    near(m("dsvdd", "auc", S(epochs=1000)), 0.89, LONG)
    near(m("dsvdd", "f1", S(epochs=0)), 0.90, 0.03)
    near(m("dsvdd", "auc", contamination=45), 0.83, 0.06)
    near(m("dsvdd", "auc", contamination=2), 0.97, 0.03)
    near(m("dsvdd", "f1", S(nu=0.01)), 0.13, 0.09)
    near(m("dsvdd", "f1", S(nu=0.5)), 0.32, 0.09)
    near(m("dsvdd", "auc", n=20), 0.72, 0.08)
    near(m("dsvdd", "auc", n=30), 0.82, 0.08)
    near(m("untrained", "auc", n=20), 0.99, 0.015)
    near(m("untrained", "auc", n=30), 0.97, 0.02)
