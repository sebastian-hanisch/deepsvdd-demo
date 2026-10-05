"""Auswertung der Deep-SVDD-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate; aus der Wurzel-Demo übernommen), Analyse einer Aufnahme für Deep SVDD, die One-Class SVM,
den Isolation Forest und die robuste Schätzung der Wurzel (dazu das untrainierte Netz, die One-Class SVM mit breitem Kernel, LOF und ECOD als Bezug), Sweeps, Experimente auf Abruf (Szenarien, Epochenverlauf, Netz-Seeds,
Kollaps, Breite, Schwelle, Kosten) und Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import dsvdd_algorithm as dv
import dsvdd_ocsvm as svm
import dsvdd_ecod as ecod
import dsvdd_lof as lof
import dsvdd_isolation_forest as isf
import dsvdd_ee_algorithm as alg
import dsvdd_constants as C
import dsvdd_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _group_ends(sorted_scores):
    """Index des letzten Elements jeder Gruppe gleicher Werte in einer sortierten Reihe (jede Schwelle trennt nur zwischen verschiedenen Werten)."""
    return np.flatnonzero(np.concatenate([sorted_scores[1:] != sorted_scores[:-1], [True]]))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    ends = _group_ends(np.asarray(score)[order])                       # gleiche Werte bilden EINE Schwelle (sonst hinge der Wert von der Zeilenreihenfolge ab)
    tp = np.cumsum(hits)[ends]
    precision = tp / (ends + 1.0)
    recall = tp / positive.sum()
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    ends = _group_ends(np.asarray(score)[order])                       # bei gleichen Werten nur die Punkte an den Schwellen (Diagonale statt Treppe in Zeilenreihenfolge)
    tpr = np.concatenate([[0.0], np.cumsum(hits)[ends] / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits)[ends] / max((~hits).sum(), 1)])
    return fpr, tpr


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


def standardise(X):
    """Kennzahlen auf Mittelwert 0 und Streuung 1 (One-Class SVM und LOF rechnen mit Abständen; Isolation Forest, robuste Schätzung und Deep SVDD skalieren selbst bzw. gar nicht)."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


@dataclass(frozen=True)
class Settings:
    variant: str = C.DEFAULT_VARIANT                    # Deep SVDD: one_class oder soft_boundary
    width: int = C.DEFAULT_WIDTH
    depth: int = C.DEFAULT_DEPTH
    n_out: int = C.DEFAULT_N_OUT
    epochs: int = C.DEFAULT_EPOCHS
    nu: float = C.DEFAULT_NU                            # Anteil der Trainingstouren außerhalb der Kugel: Schwelle R^2 = (1 - nu)-Quantil der Abstände
    activation: str = C.DEFAULT_ACTIVATION
    bias: bool = C.DEFAULT_BIAS
    scaling: str = C.DEFAULT_SCALING                    # Eingabeskalierung: minmax [0, 1] oder standard (zentriert)
    net_seed: int = C.DEFAULT_NET_SEED                  # Initialisierung des Netzes (getrennt vom Seed der Aufnahme)
    threshold_kind: str = C.DEFAULT_THRESHOLD_KIND      # "standard": R^2 (Deep SVDD), f < 0 (One-Class SVM), LOF 1.5, Score 0.5 (Isolation Forest), chi²-Quantil (robust); "share": der erwartete Anteil für alle vier
    quantile: float = C.DEFAULT_QUANTILE                # chi²-Quantil der Wurzel
    share: int = C.DEFAULT_SHARE                        # erwarteter Anteil der Anomalien [%]
    start: int = 0                                      # Seed des Isolation Forest und der MCD-Starts


DATA_KEYS = ("n", "p", "n_noise", "n_modes", "curvature", "noise", "contamination", "kind", "strength")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                    contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH)
DETECTORS = ("dsvdd", "ocsvm", "iforest", "robust")
EXTRAS = ("untrained", "ocsvm_wide", "lof", "ecod")                # Bezugsdetektoren, die im Urteil und in den Balken nicht auftauchen
DETECTOR_NAMES = {"dsvdd": "Deep SVDD", "ocsvm": "One-Class SVM", "iforest": "Isolation Forest", "robust": "robust (MCD)", "untrained": "Deep SVDD ohne Training", "ocsvm_wide": "One-Class SVM, γ = 0.1 · γ₀",
                  "lof": "LOF", "ecod": "ECOD (zweiseitig)"}
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm")
IF_TREES, IF_PSI, IF_CUTOFF, LOF_CUTOFF, LOF_K = C.DEFAULT_TREES, C.DEFAULT_PSI, C.DEFAULT_CUTOFF_IF, C.DEFAULT_CUTOFF_LOF, C.DEFAULT_LOF_K


def lof_k(n):
    """k des LOF im Vergleich: 20, höchstens n / 2 (k nahe n macht LOF unbrauchbar)."""
    return int(max(3, min(LOF_K, n // 2)))


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                 contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed, n_noise)


def fit_net(X, settings, n_epochs=None):
    """Deep SVDD mit den Reglern der Einstellungen (n_epochs überschreibbar, z. B. 0 für das untrainierte Netz)."""
    return dv.fit_deep_svdd(X, depth=settings.depth, width=settings.width, n_out=settings.n_out, activation=settings.activation, variant=settings.variant, nu=settings.nu, lr=C.DEFAULT_LR,
                            n_epochs=settings.epochs if n_epochs is None else n_epochs, seed=settings.net_seed, bias=settings.bias, weight_decay=C.DEFAULT_WEIGHT_DECAY, scaling=settings.scaling)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)
    p_total: int                  # Zahl der Merkmale einschließlich Rauschmerkmale
    net: dv.Model                 # das trainierte Netz
    ocsvm_fit: svm.Fit
    forest_if: isf.Forest
    values: dict                  # Detektor -> Anomalie-Wert je Tour (dazu die Bezugsdetektoren)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # Detektor -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold)
    flags: dict                   # Detektor -> markierte Touren bei der gewählten Schwelle
    oracle_f1: dict               # Detektor -> F1, wenn der wahre Anteil bekannt wäre (die k größten Werte)
    seconds: dict
    spread: float                 # Streuung der Ausgaben des trainierten Netzes (fällt beim Kollaps gegen 0)


_ROOT_CACHE = {}


def _root_fits(X, key, start):
    """Klassische und robuste Schätzung der Wurzel (unabhängig von den Reglern des Netzes: werden je Aufnahme nur einmal gerechnet)."""
    cache_key = (key, start)
    if key is not None and cache_key in _ROOT_CACHE:
        return _ROOT_CACHE[cache_key]
    t0 = time.perf_counter()
    classical = alg.fit_classical(X)
    t1 = time.perf_counter()
    robust = alg.fit_mcd(X, C.DEFAULT_SUPPORT, start, reweight=C.DEFAULT_REWEIGHT)
    out = (classical, robust, t1 - t0, time.perf_counter() - t1)
    if key is not None:
        if len(_ROOT_CACHE) > 600:
            _ROOT_CACHE.clear()
        _ROOT_CACHE[cache_key] = out
    return out


def _thresholds(values, settings, robust, net):
    """Markierung und Schwellenwert je Detektor: Standard = R^2 (Deep SVDD, Score über R^2 bei Soft-Boundary bzw. quadrierter Abstand über R^2), f < 0 (One-Class SVM), LOF 1.5, Score 0.5 (Isolation Forest),
    chi²-Quantil (robust), sonst die k größten Werte mit dem angenommenen Anteil."""
    flags, thr = {}, {}
    if settings.threshold_kind == "share":
        for d in DETECTORS:
            flags[d] = isf.flag_top(values[d], settings.share / 100.0)
            thr[d] = float(np.sort(values[d])[::-1][int(flags[d].sum()) - 1])
    else:
        thr["dsvdd"] = 0.0 if net.variant == "soft_boundary" else net.r2
        thr["ocsvm"] = C.FLAG_EPS
        thr["iforest"] = IF_CUTOFF
        thr["robust"] = alg.threshold(robust, settings.quantile)
        for d in DETECTORS:
            flags[d] = values[d] > thr[d]
    return flags, thr


def analyse(ds, settings=Settings(), params=None, root_key=None):
    secs = {}
    X = ds.X
    p_total = X.shape[1]
    t0 = time.perf_counter()
    net = fit_net(X, settings)
    secs["dsvdd"] = time.perf_counter() - t0
    Z = standardise(X)
    t0 = time.perf_counter()
    oc = svm.fit_ocsvm(Z, C.OCSVM_NU, C.OCSVM_GAMMA_FACTOR * svm.gamma_scale(Z))
    secs["ocsvm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lof_values = lof.fit_lof(Z, lof_k(len(X))).lof
    secs["lof"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    forest_if = isf.fit_forest(X, IF_TREES, min(IF_PSI, len(X)), settings.start)
    paths_if = isf.path_lengths(forest_if, X)
    secs["iforest"] = time.perf_counter() - t0
    classical, robust, secs["classical"], secs["robust"] = _root_fits(X, root_key, settings.start)
    d2_untrained = ((net.snapshots[0] - net.center) ** 2).sum(axis=1)                        # dasselbe Netz und Zentrum vor dem ersten Schritt
    oc_wide = svm.fit_ocsvm(Z, C.OCSVM_NU, 0.1 * svm.gamma_scale(Z))
    values = {"dsvdd": dv.score(net), "ocsvm": svm.score(oc), "iforest": isf.score_from_paths(paths_if, forest_if.psi), "robust": robust.d2,
              "untrained": d2_untrained, "ocsvm_wide": svm.score(oc_wide), "lof": lof_values, "ecod": ecod.score(X, "twosided")}
    flags, thr = _thresholds(values, settings, robust, net)
    scores = {}
    for d in DETECTORS:
        m = flag_metrics(flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=thr[d])
        scores[d] = m
    r2_untrained = dv.quantile_r2(d2_untrained, settings.nu)
    extra_flags = {"untrained": d2_untrained > r2_untrained, "ocsvm_wide": values["ocsvm_wide"] > C.FLAG_EPS, "lof": lof_values > LOF_CUTOFF, "ecod": np.zeros(len(X), bool)}
    for d in EXTRAS:
        m = flag_metrics(extra_flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=None)
        scores[d] = m
    scores["dsvdd"].update(loss=float(net.loss_history[-1]) if len(net.loss_history) else float("nan"), r2=net.r2)
    oracle = {d: flag_metrics(isf.flag_top(values[d], ds.anomaly.mean()), ds.anomaly)["f1"] for d in DETECTORS + EXTRAS}
    return Analysis(ds, settings, params, p_total, net, oc, forest_if, values, classical, robust, scores, flags, oracle, secs, dv.output_spread(net))


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)."""
    return analyse(make_dataset(*params), settings, params, root_key=params)


def score_maps(X2, settings=Settings(), grid=80, pad=0.5):
    """Abstands-Karte des Deep-SVDD-Netzes und Score des Isolation Forest auf einem Raster für zweidimensionale Kennzahlen (Darstellung der gelernten Kugel): (xs, ys, F [grid, grid], S [grid, grid], außerhalb [n]).
    F = R^2 - Abstand^2 (positiv im Innern der Kugel, Linie bei 0); das Netz wird dafür auf nur diesen zwei Merkmalen neu trainiert."""
    X2 = np.asarray(X2, dtype=float)
    lo, hi = X2.min(axis=0), X2.max(axis=0)
    span = hi - lo
    xs = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], grid)
    ys = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], grid)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    net = fit_net(X2, settings)
    F = (net.r2 - dv.distances2(net, pts)).reshape(grid, grid)
    forest = isf.fit_forest(X2, IF_TREES, min(IF_PSI, len(X2)), settings.start)
    S = isf.score_from_paths(isf.path_lengths(forest, pts), forest.psi).reshape(grid, grid)
    return xs, ys, F, S, dv.distances2(net) > net.r2


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_noise": (0, 5, 10, 20, 30, 40),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "epochs": (0, 10, 30, 100, 300, 1000),
    "width": C.WIDTH_OPTIONS,
    "depth": (0, 1, 2, 3),
    "n_out": (1, 2, 4, 8, 16),
    "nu": (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "share": (2, 5, 10, 20, 40),
}
SWEEP_LABELS = {"n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_noise": "Anzahl Rauschmerkmale", "n_modes": "Anzahl Betriebsarten", "curvature": "Krümmung des Normalbereichs", "noise": "Rauschen",
                "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "epochs": "Trainings-Epochen", "width": "Breite des Netzes", "depth": "Tiefe des Netzes (verdeckte Schichten)",
                "n_out": "Ausgabedimension", "nu": "ν (Schwelle R²)", "quantile": "chi²-Quantil der Schwelle (robust)", "share": "angenommener Anteil der Anomalien [%]"}
SETTING_PARAMETERS = ("epochs", "width", "depth", "n_out", "nu", "quantile", "share")


def _record(a):
    names = DETECTORS + EXTRAS
    out = {f"{d}_{k}": a.scores[d][k] for d in names for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    out["dsvdd_spread"] = a.spread
    for d in names:
        out[f"{d}_oracle_f1"] = a.oracle_f1[d]
    for d in DETECTORS + ("lof",):
        out[f"{d}_seconds"] = a.seconds[d]
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _analyse_seed(seed, settings, kw):
    data = {**DEFAULT_DATA, **kw}
    ds = make_dataset(seed=seed, **data)
    return analyse(ds, settings, None, root_key=(tuple(sorted(data.items())), seed))


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(_analyse_seed(s, settings, kw)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen der Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


SCENARIOS = (
    ("Standardfall", {}),
    ("gekrümmter Normalbereich", {"curvature": 1.0}),
    ("dichte Gruppe 30 %", {"kind": "cluster", "contamination": 30}),
    ("Lücke, 2 Betriebsarten", {"kind": "gap", "n_modes": 2}),
    ("Lücke, 3 Betriebsarten", {"kind": "gap", "n_modes": 3}),
    ("Korrelationsbruch", {"kind": "decorrelated"}),
    ("45 % verstreut", {"contamination": 45}),
    ("40 Rauschmerkmale", {"n_noise": 40}),
)


def scenario_table(settings=Settings(), scenarios=SCENARIOS, **base):
    """Die Detektoren und die Bezugsdetektoren in den Szenarien: AUC, F1 und F1 mit bekanntem Anteil (Mittel über die festen Sweep-Datensätze)."""
    rows = []
    for label, extra in scenarios:
        row = _mean_over_seeds(settings, **{**base, **extra})
        row["scenario"] = label
        rows.append(row)
    return rows


def epoch_curves(settings=Settings(), scenarios=SCENARIOS, net_seeds=(0, 1, 2), max_epochs=600, **base):
    """AUC des Netzes über die Trainings-Epochen (Momentaufnahmen inkl. Epoche 0 = untrainiert), gemittelt über die festen Sweep-Datensätze und `net_seeds` Netze, je Szenario; dazu die Streuung der Ausgaben.
    Rückgabe: [{scenario, epochs, auc, auc_std, spread}]."""
    out = []
    for label, extra in scenarios:
        per = []
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base, **extra})
            for ns in net_seeds:
                st = Settings(**{**settings.__dict__, "net_seed": ns, "epochs": max_epochs})
                net = fit_net(ds.X, st)
                row = {}
                for e, o in sorted(net.snapshots.items()):
                    d2 = ((o - net.center) ** 2).sum(axis=1)
                    row[e] = (roc_auc(d2, ds.anomaly), float(np.sqrt(o.var(axis=0).mean())))
                per.append(row)
        epochs = sorted(per[0])
        aucs = np.array([[r[e][0] for e in epochs] for r in per])
        spread = np.array([[r[e][1] for e in epochs] for r in per])
        out.append({"scenario": label, "epochs": epochs, "auc": aucs.mean(axis=0).tolist(), "auc_std": aucs.std(axis=0).tolist(), "spread": spread.mean(axis=0).tolist()})
    return out


def seed_table(settings=Settings(), epochs_list=(0, 30, 100, 300), net_seeds=tuple(range(10)), **base):
    """Streuung über die Netz-Seeds: je Epochenzahl AUC und F1 (an der Schwelle R^2), gemittelt über die fünf Datensätze; Streuung, Minimum und Maximum über alle Datensatz-Netz-Paare."""
    rows = []
    for ep in epochs_list:
        aucs, f1s = [], []
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base})
            for ns in net_seeds:
                net = fit_net(ds.X, Settings(**{**settings.__dict__, "net_seed": ns}), n_epochs=ep)
                aucs.append(roc_auc(dv.score(net), ds.anomaly))
                f1s.append(flag_metrics(dv.distances2(net) > net.r2 if net.variant == "one_class" else dv.score(net) > 0, ds.anomaly)["f1"])
        rows.append({"epochs": ep, "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs)), "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s))})
    return rows


def collapse_table(settings=Settings(), max_epochs=1000, net_seeds=(0, 1, 2), **base):
    """Bias und Eingabeskalierung: AUC und Ausgabestreuung über die Epochen für vier Konfigurationen (ohne / mit Bias) × (Min-Max / zentriert standardisiert); Mittel über Datensätze und Netz-Seeds."""
    rows = []
    for bias in (False, True):
        for scaling in ("minmax", "standard"):
            per = []
            for seed in C.SWEEP_SEEDS:
                ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base})
                for ns in net_seeds:
                    net = fit_net(ds.X, Settings(**{**settings.__dict__, "net_seed": ns, "bias": bias, "scaling": scaling, "epochs": max_epochs}))
                    per.append({e: (roc_auc(((o - net.center) ** 2).sum(axis=1), ds.anomaly), float(np.sqrt(o.var(axis=0).mean()))) for e, o in sorted(net.snapshots.items())})
            epochs = sorted(per[0])
            rows.append({"bias": bias, "scaling": scaling, "epochs": epochs, "auc": [float(np.mean([r[e][0] for r in per])) for e in epochs], "spread": [float(np.mean([r[e][1] for r in per])) for e in epochs]})
    return rows


def width_table(settings=Settings(), widths=C.WIDTH_OPTIONS, net_seeds=(0, 1, 2), max_epochs=600, **base):
    """AUC über Breite × Epochen (Momentaufnahmen), Mittel über Datensätze und Netz-Seeds."""
    rows = []
    for w in widths:
        per = []
        for seed in C.SWEEP_SEEDS:
            ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base})
            for ns in net_seeds:
                net = fit_net(ds.X, Settings(**{**settings.__dict__, "net_seed": ns, "width": w, "epochs": max_epochs}))
                per.append({e: roc_auc(((o - net.center) ** 2).sum(axis=1), ds.anomaly) for e, o in sorted(net.snapshots.items())})
        epochs = sorted(per[0])
        rows.append({"width": w, "epochs": epochs, "auc": [float(np.mean([r[e] for r in per])) for e in epochs]})
    return rows


def cost_table(settings=Settings(), **base):
    """Rechenzeit des Deep SVDD (Training mit den Epochen der Einstellung und mit 1000), der One-Class SVM, des LOF und des Isolation Forest über die Tourenzahl bei 12 und bei 52 Merkmalen."""
    times = []
    for n_noise in (0, 40):
        for n in (100, 300, 600):
            a_rows = [_analyse_seed(seed, settings, {**base, "n": n, "n_noise": n_noise}) for seed in C.SWEEP_SEEDS[:3]]
            long_secs = []
            for seed in C.SWEEP_SEEDS[:3]:
                ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **base, "n": n, "n_noise": n_noise})
                t0 = time.perf_counter()
                fit_net(ds.X, settings, n_epochs=1000)
                long_secs.append(time.perf_counter() - t0)
            times.append({"n": n, "p": 12 + n_noise, "dsvdd": float(np.mean([a.seconds["dsvdd"] for a in a_rows])), "dsvdd_1000": float(np.mean(long_secs)),
                          **{d: float(np.mean([a.seconds[d] for a in a_rows])) for d in ("ocsvm", "lof", "iforest")}})
    return {"times": times}


def threshold_table(settings=Settings(), **base):
    """Schwelle: (1) Kennzahlen über ν (Schwelle R^2); (2) F1 aller vier Detektoren bei ½-, 1- und 2-fach angenommenem Anteil."""
    cut = sweep("nu", settings=Settings(**{**settings.__dict__, "threshold_kind": "standard"}), **base)
    true_share = base.get("contamination", C.DEFAULT_CONTAMINATION)
    wrong = []
    for factor in (0.5, 1.0, 2.0):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "threshold_kind": "share", "share": int(round(true_share * factor))}), **base)
        row["x"], row["factor"] = int(round(true_share * factor)), factor
        wrong.append(row)
    return {"cutoff": cut, "wrong_share": wrong}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.05             # AUC-Abstand, ab dem ein Detektor als besser gilt
GAP_AUC = 0.8
RANK_AUC = 0.6
THRESHOLD_F1_DROP = 0.15
TRAINING_LOSS = 0.05          # AUC-Verlust gegenüber dem untrainierten Netz, ab dem das Training als schädlich gilt


def verdict(a):
    """(Art, Code, Kennzahlen): Lücke (keiner findet sie), Training schadet (das untrainierte Netz ist besser), ν kleiner als der Anteil, anderer Detektor besser, Deep SVDD besser, falsche Schwelle bei guter Rangfolge, sonst gleichauf."""
    ds = a.ds
    e = a.scores["dsvdd"]
    others = {d: a.scores[d]["auc"] for d in ("ocsvm", "iforest", "robust")}
    best_other = max(others.values())
    data = {"n": ds.n, "p": a.p_total, "n_noise": ds.n_noise, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": 100.0 * ds.anomaly.mean(), "n_anomalies": int(ds.anomaly.sum()), "nu": a.settings.nu,
            "epochs": a.settings.epochs, "best_other_auc": best_other, "untrained_auc": a.scores["untrained"]["auc"], "spread": a.spread, **{f"oracle_{d}": a.oracle_f1[d] for d in DETECTORS + EXTRAS},
            **{f"{d}_{k}": v for d in DETECTORS + EXTRAS for k, v in a.scores[d].items()}}
    if ds.kind == "gap" and max(best_other, e["auc"]) < GAP_AUC:
        return "warning", "gap", data
    if a.scores["untrained"]["auc"] - e["auc"] >= TRAINING_LOSS and a.settings.epochs > 0:
        return "warning", "training_hurts", data
    if 100.0 * ds.anomaly.mean() > 100.0 * a.settings.nu and best_other - e["auc"] >= WIN_MARGIN:
        return "warning", "nu_low", data
    if best_other - e["auc"] >= WIN_MARGIN:
        return "warning", "others_win", data
    if e["auc"] - best_other >= WIN_MARGIN:
        return "success", "dsvdd_wins", data
    if e["auc"] >= 0.95 and a.oracle_f1["dsvdd"] - e["f1"] > THRESHOLD_F1_DROP:
        return "warning", "threshold_off", data
    return "success", "comparable", data
