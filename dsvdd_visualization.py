"""Plotly-Visualisierungen der Deep-SVDD-Demo: Touren in der Ebene der größten Streuung, die Ausgaben des Netzes vor und nach dem Training (Kugel um c), Trainingsverlauf, die gelernte Kugel in der Ebene zweier Merkmale,
Wert-Histogramm mit Schwelle, ROC-Kurven, Kennzahlen-Balken, Sweeps und die Experimente (Szenarien, Epochenverlauf, Netz-Seeds, Kollaps, Breite, Schwelle, Kosten). Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dsvdd_ee_algorithm as alg

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, PINK = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f", "#c2185b"
DETECTOR_COLORS = {"dsvdd": PINK, "ocsvm": TEAL, "iforest": PURPLE, "robust": "#5f6b7a", "untrained": "#e58ab4", "ocsvm_wide": "#79c7cf", "lof": ORANGE, "ecod": GREEN}
DETECTOR_NAMES = {"dsvdd": "Deep SVDD", "ocsvm": "One-Class SVM", "iforest": "Isolation Forest", "robust": "robust (MCD)", "untrained": "Deep SVDD ohne Training", "ocsvm_wide": "One-Class SVM, γ = 0.1 · γ₀",
                  "lof": "LOF", "ecod": "ECOD (zweiseitig)"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke", "decorrelated": "Korrelationsbruch"}
DETECTORS = ("dsvdd", "ocsvm", "iforest", "robust")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def standardise(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s


def projection(X, robust):
    """Die Touren in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (standardisierte Kennzahlen): (Punkte n x 2, Achsen)."""
    s = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    axes = alg.projection_axes(robust.covariance / np.outer(s, s))
    return standardise(X) @ axes, axes


def _points(P, anomaly, flagged=None):
    normal = ~anomaly
    traces = [go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"),
              go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)")]
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip", name="als Anomalie markiert"))
    return traces


def build_scatter(P, anomaly, flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert)."""
    fig = go.Figure(_points(P, anomaly, flagged))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen)."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_points(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_score_hist(values, anomaly, thr, title="Deep-SVDD-Wert"):
    """Histogramm des Deep-SVDD-Werts (quadrierter Abstand zum Zentrum; Schwelle R²) (normale Touren blass, Sonderfahrten kräftig); senkrechte Linie = Schwelle."""
    hi = float(max(values.max(), thr * 1.05))
    lo = float(min(values.min(), thr * 0.95))
    bins = dict(start=lo, end=hi, size=(hi - lo) / 50.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.55, name="normale Touren", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=values[anomaly], xbins=bins, marker_color=RED, opacity=0.9, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(thr), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title=title), yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_rank_compare(dsvdd_values, robust_values, anomaly):
    """Rang von Deep SVDD gegen Rang der Wurzel (robuste Mahalanobis-Abstände) je Tour: Touren, die Deep SVDD und die Wurzel gleich einschätzen, liegen auf der Diagonalen; Sonderfahrten als Rauten."""
    n = len(dsvdd_values)
    re_ = np.argsort(np.argsort(dsvdd_values)) / (n - 1)
    rr = np.argsort(np.argsort(robust_values)) / (n - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr[~anomaly], y=re_[~anomaly], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.5), name="normale Touren", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rr[anomaly], y=re_[anomaly], mode="markers", marker=dict(size=9, color=RED, symbol="diamond"), name="Sonderfahrten", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Rang bei der Wurzel (robuste Mahalanobis-Abstände)", range=[-0.02, 1.02]),
                      yaxis=dict(title="Rang bei Deep SVDD", range=[-0.02, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) der vier Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen der vier Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in DETECTORS:
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def _band(fig, xs, rows, key, color, col):
    y, sd = np.array([r[key] for r in rows]), np.array([r[key + "_std"] for r in rows])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC der vier Detektoren (mit Streuung über die Sweep-Datensätze), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_wrong_share(rows):
    """F1 der vier Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist."""
    xs = [f"{r['x']} % ({r['factor']:g}×)" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_f1"] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="angenommener Anteil (Vielfaches des wahren)"), yaxis=dict(title="F1", range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


# --- Schritte: Abbildung und Training ----------------------------------------------------------------------------------------------------------


def output_projection(snapshots, center):
    """Feste Projektionsachsen für die Ausgabe-Karten: die zwei größten Streuungsrichtungen der Abweichungen (Ausgabe - c) des untrainierten Netzes (Epoche 0)."""
    dev = snapshots[0] - center
    _, _, vt = np.linalg.svd(dev - dev.mean(axis=0), full_matrices=False)
    axes = vt[:2].T if vt.shape[0] >= 2 else np.vstack([vt[:1], np.zeros((1, vt.shape[1]))]).T
    if axes.shape[1] < 2:
        axes = np.concatenate([axes, np.zeros((axes.shape[0], 1))], axis=1)
    return axes


def build_output_space(snapshots, center, anomaly, epochs, r2s, axes):
    """Die Ausgaben des Netzes (Abweichung von c, auf die zwei festen Achsen projiziert) nach den gewählten Epochen nebeneinander; der Kreis ist die Kugel mit dem Radius R (Schwelle) - bei mehr als zwei Ausgaben
    ist die Projektion der Kugel nur ein Teil des Kreises. Alle Karten haben dieselben Achsen, damit man das Zusammenziehen sieht."""
    fig = make_subplots(rows=1, cols=len(epochs), subplot_titles=[("untrainiert" if e == 0 else f"nach {e} Epochen") for e in epochs], horizontal_spacing=0.06)
    P0 = (snapshots[0] - center) @ axes
    lim = float(max(np.abs(P0).max(), np.sqrt(r2s[0])) * 1.08)
    normal = ~anomaly
    t = np.linspace(0, 2 * np.pi, 80)
    for k, e in enumerate(epochs, start=1):
        P = (snapshots[e] - center) @ axes
        fig.add_trace(go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=5, color=BLUE, opacity=0.55), hoverinfo="skip", showlegend=False), row=1, col=k)
        fig.add_trace(go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=7, color=RED, symbol="diamond"), hoverinfo="skip", showlegend=False), row=1, col=k)
        r = float(np.sqrt(r2s[e]))
        fig.add_trace(go.Scatter(x=r * np.cos(t), y=r * np.sin(t), mode="lines", line=dict(color="black", dash="dash", width=2), hoverinfo="skip", showlegend=False), row=1, col=k)
        fig.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers", marker=dict(size=9, color="black", symbol="x"), hoverinfo="skip", showlegend=False), row=1, col=k)
    fig.update_xaxes(range=[-lim, lim], zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-lim, lim], zeroline=False, showticklabels=False, scaleanchor=None)
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_training(loss_history, epochs, aucs, spreads):
    """Links der Wert der Zielfunktion über die Epochen (logarithmische Achse), rechts AUC (rosa) und Ausgabestreuung (grau, gestrichelt) an den Momentaufnahmen."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Zielfunktion", "AUC und Ausgabestreuung"), horizontal_spacing=0.12, specs=[[{}, {"secondary_y": True}]])
    if len(loss_history):
        fig.add_trace(go.Scatter(x=np.arange(1, len(loss_history) + 1), y=loss_history, mode="lines", line=dict(color=BLUE, width=3), hoverinfo="skip", showlegend=False), row=1, col=1)
    xs = [str(e) for e in epochs]
    fig.add_trace(go.Scatter(x=xs, y=aucs, mode="lines+markers", line=dict(color=PINK, width=3), name="AUC (Rangfolge)", hoverinfo="skip"), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=spreads, mode="lines+markers", line=dict(color=GRAY, width=2, dash="dash"), name="Streuung der Ausgaben", hoverinfo="skip"), row=1, col=2, secondary_y=True)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=2)
    fig.update_xaxes(title="Epoche", type="log", col=1)
    fig.update_yaxes(title="Ziel", type="log", col=1)
    fig.update_xaxes(title="Epoche", type="category", col=2)
    fig.update_yaxes(title="AUC", range=[0, 1.05], secondary_y=False, col=2)
    fig.update_yaxes(title="Streuung", rangemode="tozero", secondary_y=True, col=2)
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_boundary(xs, ys, F, S, X2, anomaly, outside, names):
    """Gelernte Kugel in der Ebene zweier Rohmerkmale (das Netz wird dafür auf nur diese zwei Merkmale neu trainiert): links R² - Abstand² (schwarze Linie: Rand der Kugel, innen positiv), rechts zum Vergleich
    der Score des Isolation Forest (Linie bei 0.5). Umkreiste Punkte liegen außerhalb der Kugel."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Deep SVDD: R² − Abstand²", "Isolation Forest: Score"), horizontal_spacing=0.10)
    lim = max(float(F.max()), 1e-9)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=F, colorscale="RdBu", zmid=0.0, zmin=-lim, zmax=lim, showscale=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Contour(x=xs, y=ys, z=F, contours=dict(start=0.0, end=0.0, size=1.0, coloring="none"), line=dict(color="black", width=3), showscale=False, showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=S, colorscale="YlOrRd", zmin=0.3, zmax=0.8, showscale=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Contour(x=xs, y=ys, z=S, contours=dict(start=0.5, end=0.5, size=1.0, coloring="none"), line=dict(color="black", width=3), showscale=False, showlegend=False, hoverinfo="skip"), row=1, col=2)
    normal = ~anomaly
    for col in (1, 2):
        fig.add_trace(go.Scatter(x=X2[normal, 0], y=X2[normal, 1], mode="markers", marker=dict(size=5, color=BLUE, line=dict(width=0.5, color="white")), hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=X2[anomaly, 0], y=X2[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond", line=dict(width=0.5, color="white")), hoverinfo="skip", showlegend=False), row=1, col=col)
    if outside.any():
        fig.add_trace(go.Scatter(x=X2[outside, 0], y=X2[outside, 1], mode="markers", marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=1.5, color="black")), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.update_xaxes(title=names[0])
    fig.update_yaxes(title=names[1], col=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------


def build_epoch_curves(rows):
    """AUC des Netzes über die Trainings-Epochen für mehrere Szenarien (Epoche 0 = untrainiertes Netz); Mittel über Datensätze und Netz-Seeds; die gepunktete Linie bei 0.5 ist Raten."""
    palette = [PINK, ORANGE, PURPLE, TEAL, GREEN, BLUE, RED, GRAY]
    fig = go.Figure()
    for k, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=[str(e) for e in r["epochs"]], y=r["auc"], mode="lines+markers", name=r["scenario"], line=dict(color=palette[k % len(palette)], width=3), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Trainings-Epochen (0 = untrainiert)", type="category"), yaxis=dict(title="AUC", range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_seeds(rows):
    """AUC über die Epochenzahl: Mittel (Balken), Streuung über alle Datensatz-Netz-Paare (Fehlerbalken), Minimum und Maximum (Rauten)."""
    xs = [str(r["epochs"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["auc"] for r in rows], error_y=dict(type="data", array=[r["auc_std"] for r in rows], visible=True), marker_color=PINK, name="Mittel ± Streuung", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_min"] for r in rows], mode="markers", marker=dict(symbol="diamond", size=9, color="black"), name="Minimum", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["auc_max"] for r in rows], mode="markers", marker=dict(symbol="diamond-open", size=9, color="black"), name="Maximum", hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Trainings-Epochen", type="category"), yaxis=dict(title="AUC", range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_collapse(rows):
    """Bias und Eingabeskalierung: AUC (links) und Streuung der Ausgaben (rechts, logarithmisch) über die Epochen für vier Konfigurationen."""
    styles = {(False, "minmax"): (PINK, "solid", "ohne Bias, Min-Max"), (False, "standard"): (BLUE, "solid", "ohne Bias, zentriert"), (True, "minmax"): (ORANGE, "dash", "mit Bias, Min-Max"),
              (True, "standard"): (TEAL, "dash", "mit Bias, zentriert")}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "Streuung der Ausgaben"), horizontal_spacing=0.12)
    for r in rows:
        color, dash, name = styles[(r["bias"], r["scaling"])]
        xs = [str(e) for e in r["epochs"]]
        fig.add_trace(go.Scatter(x=xs, y=r["auc"], mode="lines+markers", name=name, line=dict(color=color, width=3, dash=dash), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=r["spread"], mode="lines+markers", line=dict(color=color, width=3, dash=dash), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=1)
    fig.update_xaxes(title="Epochen", type="category")
    fig.update_yaxes(range=[0, 1.05], col=1)
    fig.update_yaxes(type="log", col=2)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_width(rows):
    """AUC über Breite (Zeilen) × Epochen (Spalten) des Netzes."""
    epochs = rows[0]["epochs"]
    z = [r["auc"] for r in rows]
    fig = go.Figure(go.Heatmap(x=[str(e) for e in epochs], y=[str(r["width"]) for r in rows], z=z, zmin=0.5, zmax=1.0, colorscale="Viridis", text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
                               textfont=dict(size=9), showscale=False, hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Trainings-Epochen", type="category"), yaxis=dict(title="Breite", type="category"))
    return lock_axes(fig)


def build_scenarios(rows):
    """AUC von Deep SVDD, dem untrainierten Netz, der One-Class SVM (γ₀ und 0.1 · γ₀), dem Isolation Forest und der Wurzel in den Szenarien (links) und F1 mit bekanntem Anteil (rechts); die Linie bei 0.5 ist Raten."""
    labels = [r["scenario"] for r in rows]
    names = ("dsvdd", "untrained", "ocsvm", "ocsvm_wide", "iforest", "robust")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC", "F1 mit bekanntem Anteil"), horizontal_spacing=0.10)
    for det in names:
        for col, key in ((1, f"{det}_auc"), (2, f"{det}_oracle_f1")):
            y = [r[key] for r in rows]
            fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], showlegend=col == 1, text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=7),
                                 hoverinfo="skip"), row=1, col=col)
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"), row=1, col=1)
    fig.update_yaxes(range=[0, 1.15])
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(height=560, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.5))
    return lock_axes(fig)


def build_cutoff(rows):
    """Deep SVDD: F1 (durchgezogen), Recall (gepunktet), Fehlalarmrate (gestrichelt) und AUC (grau) über ν bei der Schwelle R²."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["dsvdd_f1"] for r in rows], mode="lines+markers", name="F1", line=dict(color=PINK, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["dsvdd_recall"] for r in rows], mode="lines+markers", name="Recall", line=dict(color=PINK, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["dsvdd_false_alarm"] for r in rows], mode="lines+markers", name="Fehlalarmrate", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["dsvdd_auc"] for r in rows], mode="lines+markers", name="AUC", line=dict(color=GRAY, width=2), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="ν", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_costs(times):
    """Rechenzeit (Training bzw. Anpassen und Wert je Tour) von Deep SVDD (mit den Epochen der Einstellung und mit 1000), One-Class SVM, LOF und Isolation Forest über die Tourenzahl bei 12 und 52 Merkmalen."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("12 Merkmale", "52 Merkmale (40 Rauschmerkmale)"), horizontal_spacing=0.10)
    for col, p in ((1, 12), (2, 52)):
        rows = [t for t in times if t["p"] == p]
        for key, color, name in (("dsvdd", PINK, "Deep SVDD (Epochen der Einstellung)"), ("dsvdd_1000", "#7b1044", "Deep SVDD (1000 Epochen)"), ("ocsvm", TEAL, "One-Class SVM"), ("lof", ORANGE, "LOF"),
                                 ("iforest", PURPLE, "Isolation Forest")):
            y = [t[key] for t in rows]
            fig.add_trace(go.Bar(x=[f"n = {t['n']}" for t in rows], y=y, name=name, marker_color=color, showlegend=col == 1, text=[f"{v:.3f} s" for v in y], textposition="outside", textfont=dict(size=8), hoverinfo="skip"),
                          row=1, col=col)
    fig.update_yaxes(title="Sekunden", col=1)
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)
