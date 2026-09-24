"""Deep SVDD - eine gelernte Abbildung und eine Kugel um die Normalen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Deep SVDD - und lässt stattdessen das Beispiel wachsen.
Achtes Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: der Nachfolger der One-Class SVM. Statt eines festen Kernels lernt ein neuronales Netz die Abbildung, in der die Normalen in einer kleinen Kugel liegen.
Gemessen wird, was das bringt - und was nicht. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import dsvdd_algorithm as dv
import dsvdd_constants as C
from dsvdd_evaluation import (
    SWEEP_LABELS,
    SWEEP_VALUES,
    Settings,
    analyse_for,
    collapse_table,
    cost_table,
    epoch_curves,
    roc_auc,
    roc_curve,
    scenario_table,
    score_maps,
    seed_table,
    sweep,
    threshold_table,
    verdict,
    width_table,
)
from dsvdd_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    kind_options,
    load_permalink_settings,
    randomize_net_seed,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from dsvdd_visualization import (
    build_boundary,
    build_collapse,
    build_costs,
    build_cutoff,
    build_epoch_curves,
    build_features,
    build_method_bars,
    build_output_space,
    build_rank_compare,
    build_roc,
    build_scatter,
    build_scenarios,
    build_score_hist,
    build_seeds,
    build_sweep,
    build_training,
    build_width,
    build_wrong_share,
    output_projection,
    projection,
)

st.set_page_config(page_title="Deep SVDD – Sebastian Hanisch", layout="wide")


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _maps(data_params, settings, i, j):
    return score_maps(_analysis(data_params, settings).ds.X[:, [i, j]], settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings, values):
    return sweep(parameter, values=values, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _scenarios(settings):
    return scenario_table(settings)


@st.cache_data(show_spinner=False)
def _curves(settings):
    return epoch_curves(settings)


@st.cache_data(show_spinner=False)
def _seeds(base, settings):
    return seed_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _collapse(base, settings):
    return collapse_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _width(base, settings):
    return width_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _costs(base, settings):
    return cost_table(settings, **dict(base))


st.title("🕸️ Deep SVDD – eine gelernte Abbildung und eine Kugel um die Normalen")
st.markdown(
    """
Die One-Class SVM brauchte einen **festen Kernel** - und ihr Ergebnis hing an dessen Breite γ: dasselbe Verfahren hatte je nach γ F1 0.95 oder 0.00. **Deep SVDD** ersetzt den festen Kernel durch eine **gelernte Abbildung**: ein neuronales Netz bildet jede Tour auf wenige Zahlen ab,
und das Training zieht die Normalen in eine **möglichst kleine Kugel** um ein festes Zentrum $c$. Eine Tour ist um so auffälliger, je weiter ihr Bild vom Zentrum liegt. Damit fällt das γ weg - die Frage dieser Demo ist, ob die Abhängigkeit von einem Regler damit **verschwindet oder nur umzieht**
(auf Epochen, Breite, Skalierung, Aktivierung und den Zufall der Initialisierung).
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - achtes Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Szenario, One-Class SVM, LOF, Isolation Forest und die robuste Schätzung der Wurzel sind wortgleich aus den Vorgänger-Demos übernommen (die klassische Schätzung entfällt, damit die Balken lesbar bleiben); "
    "als Bezug laufen zusätzlich das **untrainierte** Netz, die One-Class SVM mit breitem Kernel, LOF und ECOD mit. Die Linie hat keinen Konvergenzpunkt; Deep SVDD ist die Fortsetzung des One-Class-SVM-Astes, der letzte Nachbar wäre ein Autoencoder (noch nicht gebaut)."
)

with st.expander("So funktioniert Deep SVDD", expanded=True):
    st.markdown(
        """
1. **Netz.** Die Kennzahlen jeder Tour laufen durch ein Netz $\\phi(x; W)$ aus verdeckten Schichten (Breite, Tiefe) und einer linearen Ausgabeschicht mit $d$ Ausgaben - **ohne Bias-Terme** (Standard; mit Bias könnte das Netz alle Touren auf einen Punkt abbilden).
2. **Zentrum.** $c$ ist der Mittelwert der Ausgaben des **untrainierten**, zufällig initialisierten Netzes und bleibt danach fest.
3. **Training.** One-Class-Variante: minimiere $\\tfrac1n \\sum_i \\lVert \\phi(x_i) - c\\rVert^2$ (Adam, die Demo rechnet die Gradienten selbst und prüft sie gegen Finite Differenzen). Soft-Boundary-Variante: zusätzlich ein Radius $R$ und ein Anteil ν: $R^2 + \\tfrac{1}{\\nu n}\\sum_i \\max(0, \\lVert \\phi(x_i) - c\\rVert^2 - R^2)$.
4. **Wert und Schwelle.** Der Wert ist der quadrierte Abstand $\\lVert \\phi(x) - c\\rVert^2$; als Schwelle dient das $(1-\\nu)$-Quantil der Trainingsabstände ($R^2$) - wie bei der One-Class SVM ist der **Anteil ν eingebaut**.

Was die Methode **voraussetzt**: passende Eingaben (das Netz ohne Bias bildet den Ursprung auf 0 ab), eine sinnvolle Zahl von Epochen und Glück bei der Initialisierung. Was **nicht** gelernt werden muss: ein Kernel.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren. AUC von Deep SVDD (Voreinstellung, 100 Epochen) bei 20 / 30 / 50 / 100 / 200 / 400 / 600 Touren: 0.72 / 0.82 / 0.91 / 0.94 / 0.94 / 0.92 / 0.93, das untrainierte Netz hat 0.99 / 0.97 / 0.99 / 1.00 / 0.99 / 1.00 / 1.00; "
             "F1 an der Schwelle 0.13 / 0.48 / 0.71 / 0.74 / 0.70 / 0.70 / 0.70.",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). AUC von Deep SVDD bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen: 0.91 / 0.92 / 0.94 / 0.92 / 0.82 / 0.92 - unruhig, weil das Ergebnis am Zufall der Initialisierung hängt; "
             "das untrainierte Netz hat 0.96 / 1.00 / 0.99 / 1.00 / 0.99 / 1.00.",
    )
    n_noise = st.slider(
        "Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", step=5,
        help="Zusätzliche unabhängige Spalten ohne Zusammenhang mit den Faktoren. AUC von Deep SVDD bei 0 / 5 / 10 / 20 / 30 / 40 Rauschmerkmalen: 0.92 / 0.86 / 0.87 / 0.73 / 0.72 / 0.71, das untrainierte Netz 0.99 / 0.97 / 0.90 / 0.77 / 0.83 / 0.73. "
             "Die One-Class SVM mit breitem Kernel hält 0.99, der Isolation Forest 0.99, die robuste Schätzung sinkt auf 0.88.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. AUC von Deep SVDD bei 1 / 2 / 3 Betriebsarten 0.92 / 0.94 / 0.87 (untrainiert 0.99 / 0.94 / 0.97), robust 0.99 / 0.95 / 0.85.",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Bei 0 / 0.25 / 0.5 / 0.75 / 1 steigt die AUC von Deep SVDD von 0.92 auf 0.96 / 0.97 / 0.98 / 0.98 und der F1 von 0.62 auf 0.77 / 0.78 / 0.80 / 0.83: "
             "je stärker die Krümmung, desto näher kommt das trainierte Netz dem untrainierten (0.99).",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen. AUC von Deep SVDD bei 0 / 0.25 / 0.5 / 0.75 / 1.0: 0.99 / 0.92 / 0.88 / 0.89 / 0.91 (F1 0.89 / 0.62 / 0.56 / 0.59 / 0.62).",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind. AUC von Deep SVDD bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % verstreuten Anomalien: 0.97 / 0.95 / 0.92 / 0.89 / 0.86 / 0.84 / 0.83 - das Training lernt die Anomalien mit; das untrainierte Netz bleibt bei 0.99-1.00. "
             "Zum Vergleich die One-Class SVM bei ν = 0.1: 1.00 bis 0.32.",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung. Dichte Gruppe (30 %): AUC von Deep SVDD **0.18** - unter Raten -, untrainiert 0.85, One-Class SVM 0.74 (mit breitem Kernel 0.84), Isolation Forest 0.70, robust 0.49, LOF 0.47, ECOD 0.84. "
             "In der Lücke: Deep SVDD 0.45 (2 Betriebsarten) und 0.53 (3), untrainiert nur 0.02 und 0.03 - hier hilft das Training - One-Class SVM 0.64 / 0.82, Isolation Forest 0.54 / 0.27, robust 0.40 / 0.38. "
             "Korrelationsbruch: Deep SVDD 0.92, untrainiert 0.70, One-Class SVM 0.88, Isolation Forest 0.80, ECOD 0.56, LOF 0.99, robust 1.00.",
    )
    if kind not in ("gap", "decorrelated"):
        seed_widget("strength_slider")
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind.",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))

    st.markdown("**Deep SVDD**")
    variant = st.selectbox(
        "Variante", C.VARIANTS, key="variant_select", format_func=lambda v: C.VARIANT_LABELS[v],
        help="One-Class: mittlerer quadrierter Abstand zum Zentrum. Soft-Boundary: Radius R und ν; R² wird alle fünf Epochen als (1 − ν)-Quantil der Abstände neu gesetzt. Bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen ist die AUC "
             "One-Class 0.99 / 0.99 / 0.95 / 0.92 / 0.91 / 0.89 und Soft-Boundary 0.99 / 0.99 / 0.97 / 0.89 / 0.83 / 0.83 - kein Vorteil.",
    )
    width = st.select_slider(
        "Breite der verdeckten Schichten", options=list(C.WIDTH_OPTIONS), key="width_select",
        help="Neuronen je verdeckter Schicht. Bei 100 Epochen ist die AUC von Deep SVDD bei Breite 4 / 8 / 16 / 32 / 64 gleich 0.98 / 0.98 / 0.95 / 0.92 / 0.88 - ein breiteres Netz kann sich stärker zusammenziehen und verliert mehr; "
             "das untrainierte Netz hat 0.99 bis 1.00.",
    )
    depth = st.slider(
        "Tiefe (verdeckte Schichten)", *bounds("depth_slider"), key="depth_slider",
        help="Anzahl verdeckter Schichten (0 = lineare Abbildung). Bei 0 / 1 / 2 / 3 Schichten ist die AUC von Deep SVDD nach 100 Epochen 0.98 / 0.92 / 0.92 / 0.95 (untrainiert 1.00 / 1.00 / 0.99 / 0.98).",
    )
    n_out = st.slider(
        "Ausgabedimension", *bounds("n_out_slider"), key="n_out_slider",
        help="Zahl der Ausgaben des Netzes. Bei 1 / 2 / 4 / 8 / 16 Ausgaben ist die AUC von Deep SVDD 0.70 / 0.79 / 0.82 / 0.92 / 0.97, die des untrainierten Netzes 0.66 / 0.93 / 0.96 / 0.99 / 0.98.",
    )
    epochs = st.slider(
        "Epochen", *bounds("epochs_slider"), key="epochs_slider", step=10,
        help="Trainingsschritte (Vollbatch, Adam, Lernrate 0.001). **0 = das untrainierte Netz.** Bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen: AUC 0.99 / 0.99 / 0.95 / 0.92 / 0.91 / 0.89, F1 an der Schwelle 0.90 / 0.86 / 0.77 / 0.62 / 0.61 / 0.63, "
             "Streuung der Ausgaben 0.081 / 0.048 / 0.037 / 0.019 / 0.010 / 0.003: mit dem Training verschlechtert sich die Rangfolge, während die Kugel kleiner wird.",
    )
    nu = st.slider(
        "ν (Anteil außerhalb der Kugel)", *bounds("nu_slider"), key="nu_slider", step=0.01, format="%.2f",
        help="Die Schwelle R² ist das (1 − ν)-Quantil der Trainingsabstände: es werden etwa ν der Touren markiert. Die Rangfolge (AUC 0.92) ändert sich nicht. Bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5: "
             "F1 0.13 / 0.29 / 0.59 / 0.62 / 0.55 / 0.44 / 0.32, Recall 0.07 / 0.17 / 0.43 / 0.61 / 0.82 / 0.88 / 0.94, Fehlalarmrate 0 % / 0 % / 0.4 % / 3.9 % / 12.7 % / 23 % / 45 %.",
    )
    activation = st.selectbox(
        "Aktivierung", C.ACTIVATIONS, key="activation_select",
        help="tanh oder ReLU. Bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen ist die AUC mit ReLU 0.98 / 0.99 / 0.99 / 0.92 / 0.73 / 0.49 (tanh: 0.99 / 0.99 / 0.95 / 0.92 / 0.91 / 0.89): ReLU trainiert anfangs besser und fällt später unter Raten.",
    )
    bias = st.checkbox(
        "Bias-Terme erlauben", key="bias_toggle",
        help="Im Paper ist das Netz bias-frei, damit es nicht alles auf c abbilden kann. Gemessen (Standardfall, Min-Max): mit Bias ist die AUC nach 0 / 10 / 30 / 100 / 300 / 1000 Epochen 0.99 / 0.99 / 0.95 / 0.85 / 0.87 / 0.90, bias-frei 0.99 / 0.99 / 0.95 / 0.92 / 0.91 / 0.89: "
             "die Ausgaben ziehen sich in beiden Fällen zusammen (Streuung nach 1000 Epochen 0.002 mit, 0.004 ohne Bias).",
    )
    scaling = st.selectbox(
        "Eingabeskalierung", C.SCALINGS, key="scaling_select", format_func=lambda s: C.SCALING_LABELS[s],
        help="Min-Max: Kennzahlen auf [0, 1] - ein bias-freies Netz kann so c erreichen. Standardisiert (zentriert): der Mittelwert der Normalen liegt im Ursprung, den ein bias-freies Netz auf 0 abbildet: "
             "die Normalen können dem Zentrum nie nahe kommen. AUC bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen: 0.95 / 0.89 / 0.65 / 0.58 / 0.48 / **0.32** - unter Raten.",
    )
    net_seed = st.number_input(
        "Netz-Seed (Initialisierung)", *bounds("net_seed_input"), key="net_seed_input", step=1,
        help="Zufall der Anfangsgewichte, getrennt vom Seed der Touren. Über 10 Netz-Seeds und 5 Datensätze ist die AUC nach 0 / 30 / 100 / 300 Epochen 0.99 ± 0.02 / 0.95 ± 0.02 / 0.91 ± 0.05 / 0.89 ± 0.05 (schlechtestes Netz 0.91 / 0.90 / 0.76 / 0.76).",
    )
    st.button("🎲 Neues Netz würfeln", width="stretch", on_click=randomize_net_seed, help="Würfelt einen neuen Netz-Seed: dieselben Touren, andere Anfangsgewichte.")
    threshold_kind = st.selectbox(
        "Schwelle", C.THRESHOLD_KINDS, key="threshold_kind_select", format_func=lambda k: C.THRESHOLD_LABELS[k],
        help="Standard: Deep SVDD über R², One-Class SVM bei f < 0, Isolation Forest über Score 0.5, robust über dem χ²-Quantil. Erwarteter Anteil: bei allen vieren werden die größten Werte markiert - "
             "dann entscheidet nur die Rangfolge, aber der Anteil muss bekannt sein.",
    )
    if threshold_kind == "standard":
        seed_widget("quantile_slider")
        quantile = st.slider(
            "Schwelle: χ²-Quantil (robust)", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
            help="Ab welchem Anteil der χ²-Verteilung eine Tour bei der robusten Schätzung als Anomalie gilt. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 der robusten Schätzung 0.67 / 0.77 / 0.84 / 0.89 / 0.90.",
        )
        st.session_state["_quantile_kept"] = quantile
        share = int(st.session_state.get("_share_kept", C.DEFAULT_SHARE))
    else:
        seed_widget("share_slider")
        share = st.slider(
            "Angenommener Anteil der Anomalien [%]", *bounds("share_slider"), key="share_slider",
            help="Wie viele Touren als Anomalie markiert werden (die größten Werte, für alle vier Detektoren). Beim wahren Anteil 10 % ist der F1 bei angenommenen 2 / 5 / 10 / 20 / 40 % bei Deep SVDD (Voreinstellung) "
                 "0.33 / 0.59 / 0.63 / 0.55 / 0.37, beim Isolation Forest 0.33 / 0.67 / 0.97 / 0.67 / 0.40, bei robust 0.33 / 0.67 / 0.90 / 0.66 / 0.40.",
        )
        st.session_state["_share_kept"] = share
        quantile = float(st.session_state.get("_quantile_kept", C.DEFAULT_QUANTILE))
    seed = st.number_input("Zufalls-Seed (Touren)", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_noise_slider": int(n_noise), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "variant_select": variant, "width_select": int(width), "depth_slider": int(depth), "n_out_slider": int(n_out),
    "epochs_slider": int(epochs), "nu_slider": float(nu), "activation_select": activation, "bias_toggle": "1" if bias else "0", "scaling_select": scaling, "net_seed_input": int(net_seed),
    "threshold_kind_select": threshold_kind, "quantile_slider": float(quantile), "share_slider": int(share), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_noise), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), int(seed))
settings = Settings(variant=variant, width=int(width), depth=int(depth), n_out=int(n_out), epochs=int(epochs), nu=float(round(nu, 2)), activation=activation, bias=bool(bias), scaling=scaling, net_seed=int(net_seed),
                    threshold_kind=threshold_kind, quantile=float(quantile), share=int(share))
with st.spinner("Trainiere das Netz..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
ds_s, oc_s, if_s, rb_s = (a.scores[d] for d in ("dsvdd", "ocsvm", "iforest", "robust"))
un_s = a.scores["untrained"]
n_anom = int(ds.anomaly.sum())
p_total = a.p_total
net = a.net
values = a.values["dsvdd"]
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)),
                          "contamination": int(contamination), "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- Deep SVDD in Aktion ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Deep SVDD in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Abbildung", 3: "3 · Training", 4: "4 · Wert", 5: "5 · Ergebnis"}
if "dsvdd_step" not in st.session_state or st.session_state.get("dsvdd_step_owner") != data_key:
    st.session_state["dsvdd_step"] = 1
    st.session_state["dsvdd_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="dsvdd_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes = projection(ds.X, a.robust)
if ds.kind == "decorrelated" and ds.p >= 2:
    corr = np.abs(np.corrcoef(ds.X[~ds.anomaly][:, : ds.p].T))
    np.fill_diagonal(corr, 0.0)
    pair_i, pair_j = (int(v) for v in np.unravel_index(np.argmax(corr), corr.shape))
else:
    pair_i, pair_j = 0, min(1, ds.p - 1)
names_all = ds.names
proj_axes = output_projection(net.snapshots, net.center)
shown = sorted({0, min(net.snapshots, key=lambda e: abs(e - 10)), min(net.snapshots, key=lambda e: abs(e - 100)), net.n_epochs})
snap_auc = {e: roc_auc(((o - net.center) ** 2).sum(axis=1), ds.anomaly) for e, o in sorted(net.snapshots.items())}
snap_spread = {e: float(np.sqrt(o.var(axis=0).mean())) for e, o in sorted(net.snapshots.items())}
d2_0 = ((net.snapshots[0] - net.center) ** 2).sum(axis=1)
r2_0 = net.r2_snapshots[0]


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {names_all[pair_i]} gegen {names_all[pair_j]}** (Einheiten wie gemessen)" + (" - das am stärksten korrelierte Paar der Normalen" if ds.kind == "decorrelated" else ""))
            c2.plotly_chart(build_features(ds.X, ds.anomaly, names_all, pair_i, pair_j), width="stretch", key="step_features")
        elif current_step == 2:
            c1, c2 = st.columns(2)
            c1.markdown("**Ausgaben des zufällig initialisierten Netzes** (Abweichung vom Zentrum c, auf zwei Achsen projiziert; Kreuz = c, gestrichelt = Schwelle R)")
            c1.plotly_chart(build_output_space({0: net.snapshots[0]}, net.center, ds.anomaly, [0], {0: r2_0}, proj_axes), width="stretch", key="step_out0")
            c2.markdown("**Abstand zum Zentrum vor dem Training** (Schwelle gestrichelt)")
            c2.plotly_chart(build_score_hist(d2_0, ds.anomaly, r2_0), width="stretch", key="step_hist0")
        elif current_step == 3:
            st.markdown("**Die Ausgaben nach den Epochen** (gleiche Achsen: die Punktwolke zieht sich zusammen)")
            st.plotly_chart(build_output_space(net.snapshots, net.center, ds.anomaly, shown, net.r2_snapshots, proj_axes), width="stretch", key="step_outputs")
            st.plotly_chart(build_training(net.loss_history, list(snap_auc), list(snap_auc.values()), list(snap_spread.values())), width="stretch", key="step_training")
            xs_, ys_, F_, S_, outside_ = _maps(data_params, settings, pair_i, pair_j)
            st.markdown(f"**Die gelernte Kugel in der Ebene {names_all[pair_i]} gegen {names_all[pair_j]}** (das Netz wird dafür auf nur diese zwei Merkmale neu trainiert; schwarze Linie: Rand der Kugel, umkreist: Touren außerhalb)")
            st.plotly_chart(build_boundary(xs_, ys_, F_, S_, ds.X[:, [pair_i, pair_j]], ds.anomaly, outside_, [names_all[pair_i], names_all[pair_j]]), width="stretch", key="step_boundary")
        elif current_step == 4:
            c1, c2 = st.columns(2)
            c1.markdown("**Deep-SVDD-Wert je Tour** (Schwelle gestrichelt)")
            thr_line = ds_s["threshold"] if net.variant == "one_class" else 0.0
            c1.plotly_chart(build_score_hist(values, ds.anomaly, thr_line), width="stretch", key="step_score_hist")
            c2.markdown("**Deep SVDD gegen die Wurzel**: Rang jeder Tour bei beiden")
            c2.plotly_chart(build_rank_compare(values, a.values["robust"], ds.anomaly), width="stretch", key="step_rank_compare")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(a.values[d], ds.anomaly), a.scores[d]["auc"]) for d in ("dsvdd", "ocsvm", "iforest", "robust")}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {p_total} Kennzahlen" + (f" ({ds.n_noise} davon reines Rauschen)" if ds.n_noise else "") + f", davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), "
               f"{ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. Das Netz bekommt alle {p_total} Kennzahlen, {C.SCALING_LABELS[settings.scaling]}-skaliert, und hat {net.n_parameters:,} Parameter ({net.depth} verdeckte Schichten der Breite {net.width}, {net.n_out} Ausgaben, "
               f"{'mit' if net.bias else 'ohne'} Bias).".replace(",", ".")
               + (" Beim Korrelationsbruch haben die Anomalien in jedem Merkmal normale Werte - nur die Kombination stimmt nicht." if ds.kind == "decorrelated" else ""))
elif step == 2:
    st.caption(f"Das Zentrum c ist der Mittelwert dieser Ausgaben (Betrag {np.linalg.norm(net.center):.2f}). Schon der Abstand zum Zentrum ist ein Detektor: AUC **{un_s['auc']:.2f}**, F1 {un_s['f1']:.2f} an der Schwelle R² = {r2_0:.3f} "
               f"({1 - net.nu:.0%}-Quantil der Abstände). Das Netz ist noch nicht trainiert; dass ein zufälliges, glattes Netz die Touren so gut trennt, ist der Bezugspunkt für alles Folgende.")
elif step == 3:
    st.caption(f"Nach {net.n_epochs} Epochen: AUC {snap_auc[net.n_epochs]:.2f} (untrainiert {snap_auc[0]:.2f}), Streuung der Ausgaben {snap_spread[net.n_epochs]:.3f} (untrainiert {snap_spread[0]:.3f}); "
               f"Zielfunktion {net.loss_history[-1]:.4f} (Start {net.loss_history[0]:.4f})." if net.n_epochs > 0 else "0 Epochen gewählt: es wird nicht trainiert, die Ausgaben sind die des zufälligen Netzes.")
elif step == 4:
    st.caption(f"Mittlerer Wert der normalen Touren {values[~ds.anomaly].mean():.3f}, der Sonderfahrten {values[ds.anomaly].mean():.3f}; die Schwelle R² = {net.r2:.3f} markiert {ds_s['n_flagged']} von {ds.n} Touren. "
               "Rechts: liegen die Punkte nahe der Diagonalen, urteilen Deep SVDD und Wurzel gleich.")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Detektoren gefunden haben – Deep SVDD gegen One-Class SVM gegen Isolation Forest gegen Wurzel")
st.caption(
    "Anomalie = Tour über der Schwelle (Standard: Deep SVDD über R², One-Class SVM bei f < 0, Isolation Forest über Score 0.5, robust über dem χ²-Quantil). **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Wert hat als eine "
    "zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). **Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (Deep SVDD)", f"{ds_s['auc']:.2f}", help="Rangfolge der Anomalie-Werte; darunter die anderen Detektoren.")
m1.caption(f"untrainiert {un_s['auc']:.2f} · One-Class SVM {oc_s['auc']:.2f} · Isolation Forest {if_s['auc']:.2f} · robust {rb_s['auc']:.2f}")
m2.metric("Recall (Deep SVDD)", _pct(ds_s["recall"]), help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m2.caption(f"untrainiert {_pct(un_s['recall'])} · One-Class SVM {_pct(oc_s['recall'])} · Isolation Forest {_pct(if_s['recall'])} · robust {_pct(rb_s['recall'])}")
m3.metric("Fehlalarmrate (Deep SVDD)", f"{ds_s['false_alarm']:.1%}", help="Anteil der normalen Touren, die als Anomalie markiert werden.")
m3.caption(f"untrainiert {un_s['false_alarm']:.1%} · One-Class SVM {oc_s['false_alarm']:.1%} · Isolation Forest {if_s['false_alarm']:.1%} · robust {rb_s['false_alarm']:.1%}")
m4.metric("F1 (Deep SVDD)", f"{ds_s['f1']:.2f}", help="Harmonisches Mittel aus Precision und Recall bei der Schwelle; darunter die F1 der anderen Detektoren und die von Deep SVDD, wenn der wahre Anteil bekannt wäre.")
m4.caption(f"untrainiert {un_s['f1']:.2f} · One-Class SVM {oc_s['f1']:.2f} · Isolation Forest {if_s['f1']:.2f} · robust {rb_s['f1']:.2f} · Deep SVDD mit bekanntem Anteil {a.oracle_f1['dsvdd']:.2f}")

_t = vd
if code == "dsvdd_wins":
    st.success(f"✅ Deep SVDD ist besser: AUC {_t['dsvdd_auc']:.2f} gegen {_t['best_other_auc']:.2f} beim besten anderen Detektor (One-Class SVM {_t['ocsvm_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}); "
               f"an der Schwelle Recall {_pct(_t['dsvdd_recall'])}, F1 {_t['dsvdd_f1']:.2f}.")
elif code == "comparable":
    st.success(f"✅ Deep SVDD ist ebenbürtig: AUC {_t['dsvdd_auc']:.2f} (untrainiert {_t['untrained_auc']:.2f}, One-Class SVM {_t['ocsvm_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}); bei der Schwelle F1 {_t['dsvdd_f1']:.2f} "
               f"(untrainiert {_t['untrained_f1']:.2f}, One-Class SVM {_t['ocsvm_f1']:.2f}, Isolation Forest {_t['iforest_f1']:.2f}, robust {_t['robust_f1']:.2f}), mit bekanntem Anteil {_t['oracle_dsvdd']:.2f}.")
elif code == "training_hurts":
    st.warning(f"⚠️ Das Training schadet: nach {_t['epochs']} Epochen ist die AUC {_t['dsvdd_auc']:.2f}, das **untrainierte** Netz hat {_t['untrained_auc']:.2f} (F1 {_t['dsvdd_f1']:.2f} gegen {_t['untrained_f1']:.2f}). "
               f"Die Ausgaben ziehen sich auf eine Streuung von {_t['spread']:.3f} zusammen, und die Rangfolge wird schlechter. Isolation Forest {_t['iforest_auc']:.2f}, One-Class SVM {_t['ocsvm_auc']:.2f}, robust {_t['robust_auc']:.2f}. "
               "Weniger Epochen (oder gar keine) wären besser gewesen - aber ohne Etiketten weiß man das nicht.")
elif code == "nu_low":
    st.warning(f"⚠️ ν ist kleiner als der Anteil der Anomalien: ν = {_t['nu']:.2f}, die Daten enthalten {_t['contamination']:.0f} % Sonderfahrten. AUC {_t['dsvdd_auc']:.2f} (Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}).")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['dsvdd_auc']:.2f} bei Deep SVDD (untrainiert {_t['untrained_auc']:.2f}), {_t['ocsvm_auc']:.2f} bei der One-Class SVM, {_t['iforest_auc']:.2f} beim Isolation Forest, "
               f"{_t['robust_auc']:.2f} robust: keiner findet sie zuverlässig. Das untrainierte Netz liegt weit unter Raten - die Lücke liegt im Innern der glatten Abbildung; das Training hebt die AUC, aber nicht über Raten.")
elif code == "others_win":
    extra = f" Immerhin hilft das Training hier: untrainiert ist die AUC nur {_t['untrained_auc']:.2f}." if _t["dsvdd_auc"] - _t["untrained_auc"] >= 0.1 else ""
    st.warning(f"⚠️ Ein anderer Detektor ist besser: AUC Deep SVDD {_t['dsvdd_auc']:.2f}, One-Class SVM {_t['ocsvm_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}; an der Schwelle F1 {_t['dsvdd_f1']:.2f} gegen "
               f"{max(_t['ocsvm_f1'], _t['iforest_f1'], _t['robust_f1']):.2f}.{extra}")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist gut (AUC {_t['dsvdd_auc']:.2f}), aber R² markiert nur ν = {_t['nu']:.2f} der Touren: F1 {_t['dsvdd_f1']:.2f} gegen {_t['oracle_dsvdd']:.2f} mit bekanntem Anteil.")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"),
            ("Fehlalarmrate", "false_alarm", "{:.3f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "Deep SVDD": [r[2].format(ds_s[r[1]]) for r in rows], "untrainiert": [r[2].format(un_s[r[1]]) for r in rows], "One-Class SVM": [r[2].format(oc_s[r[1]]) for r in rows],
              "Isolation Forest": [r[2].format(if_s[r[1]]) for r in rows], "robust (MCD)": [r[2].format(rb_s[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Rechenzeit", "Parameter", "Bewertung", "mittlerer Wert (normal / Anomalie)"],
              "Deep SVDD": [f"{a.seconds['dsvdd'] * 1000:.0f} ms", f"{net.n_parameters:,} Gewichte".replace(",", "."), f"quadrierter Abstand zum Zentrum ({net.n_epochs} Epochen)", f"{values[~ds.anomaly].mean():.3f} / {values[ds.anomaly].mean():.3f}"],
              "One-Class SVM": [f"{a.seconds['ocsvm'] * 1000:.1f} ms", "ν = 0.10, γ = γ₀", "Entscheidungsfunktion f", f"{a.values['ocsvm'][~ds.anomaly].mean():.3f} / {a.values['ocsvm'][ds.anomaly].mean():.3f}"],
              "Isolation Forest": [f"{a.seconds['iforest'] * 1000:.0f} ms", f"{len(a.forest_if.trees)} Bäume × ψ = {a.forest_if.psi}", "Pfadlänge in zufälligen Bäumen", f"{a.values['iforest'][~ds.anomaly].mean():.2f} / {a.values['iforest'][ds.anomaly].mean():.2f}"]})
    st.caption("Deep SVDD ist bei festem Seed deterministisch, der Zufall steckt in den Anfangsgewichten (Netz-Seed). Die klassische Schätzung der Wurzel entfällt in dieser Demo; das untrainierte Netz ist dasselbe Netz und Zentrum vor dem ersten Schritt.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Daten und Reglern ab?")
sweep_options = [k for k in SWEEP_LABELS if not ((kind in ("gap", "decorrelated") and k in ("strength",)) or (kind == "gap" and k == "n_modes") or (threshold_kind == "share" and k == "quantile") or (threshold_kind == "standard" and k == "share"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise), "contamination": int(contamination), "strength": float(strength),
           "epochs": int(epochs), "width": int(width), "depth": int(depth), "n_out": int(n_out), "nu": float(settings.nu), "quantile": float(quantile), "share": int(share)}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings, SWEEP_VALUES[sweep_param])
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben), Netz-Seed wie in der Seitenleiste; alle anderen Regler wie in der Seitenleiste. Links die Rangfolge (AUC), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle. "
               "Bei den Reglern des Netzes ändern sich nur die Linien von Deep SVDD.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Das Training schadet: AUC über die Epochen")
if st.button("AUC über die Epochen in acht Szenarien berechnen (dauert etwa 30 Sekunden)", key="curves_start"):
    st.session_state["curves_on"] = True
if st.session_state.get("curves_on"):
    with st.spinner("Trainiere 8 Szenarien × 5 Aufnahmen × 3 Netze..."):
        cv = _curves(settings)
    st.plotly_chart(build_epoch_curves(cv), width="stretch", key="curves_chart")
    st.caption("AUC der Abstände zum Zentrum an Momentaufnahmen des Trainings (600 Epochen); Epoche 0 = untrainiertes Netz. Mittel über 5 feste Datensätze und 3 Netz-Seeds. **Standardfall**: 1.00 → 0.94 (33 Epochen) → 0.88 (600); "
               "**gekrümmter Normalbereich**: 1.00 → 0.95; **45 % verstreut**: 0.99 → 0.95 (33) → 0.62 (600) - das Training lernt die Anomalien mit; **dichte Gruppe (30 %)**: 0.80 → 0.27 (3 Epochen) → 0.18 (59) - unter Raten; "
               "**40 Rauschmerkmale**: 0.73 → 0.65. Nur wo die Abbildung erst gelernt werden muss, hilft das Training: **Korrelationsbruch** 0.65 → 0.91 (105 Epochen) → 0.79 (600), **Lücke** (2 / 3 Betriebsarten) 0.05 / 0.02 → 0.47 / 0.59 (nach 188 Epochen) - nie über Raten hinaus. "
               "Die Zahl der Epochen, bei der es am besten ist, hängt vom Szenario ab und ist ohne Etiketten nicht zu erkennen.")

st.markdown("---")

st.subheader("🔬 Der Zufall der Initialisierung")
if st.button("10 Netz-Seeds vergleichen (dauert etwa 20 Sekunden)", key="seeds_start"):
    st.session_state["seeds_on"] = True
if st.session_state.get("seeds_on"):
    with st.spinner("Trainiere 4 Epochenzahlen × 5 Aufnahmen × 10 Netze..."):
        sd = _seeds(base_data, settings)
    st.plotly_chart(build_seeds(sd), width="stretch", key="seeds_chart")
    st.caption("AUC über alle 50 Paare aus Datensatz und Netz-Seed (Standardfall, Einstellungen der Seitenleiste bis auf die Epochen). Nach 0 / 30 / 100 / 300 Epochen: Mittel 0.99 / 0.95 / 0.91 / 0.89, Streuung 0.02 / 0.02 / 0.05 / 0.05, "
               "schlechtestes Netz 0.91 / 0.90 / 0.76 / 0.76, bestes 1.00 / 0.99 / 0.98 / 0.97; F1 an der Schwelle 0.90 ± 0.08 / 0.75 ± 0.06 / 0.65 ± 0.12 / 0.61 ± 0.12. "
               "Je länger trainiert wird, desto stärker hängt das Ergebnis von den Anfangsgewichten ab - das ist bei der deterministischen One-Class SVM nicht so.")

st.markdown("---")

st.subheader("🔬 Bias und Eingabeskalierung: kollabiert die Kugel?")
if st.button("Bias × Skalierung über 1000 Epochen berechnen (dauert etwa 20 Sekunden)", key="collapse_start"):
    st.session_state["collapse_on"] = True
if st.session_state.get("collapse_on"):
    with st.spinner("Trainiere 4 Konfigurationen × 5 Aufnahmen × 3 Netze über 1000 Epochen..."):
        cl = _collapse(base_data, settings)
    st.plotly_chart(build_collapse(cl), width="stretch", key="collapse_chart")
    st.caption("Standardfall, Mittel über 5 Datensätze und 3 Netz-Seeds. Die Ausgaben ziehen sich in **allen** vier Konfigurationen zusammen (Streuung nach 1000 Epochen 0.004 ohne und 0.002 mit Bias bei Min-Max; 0.023 und 0.009 zentriert): "
               "das bias-freie Netz verhindert den vollständigen Kollaps nicht, es macht ihn nur langsamer. Die **Rangfolge** überlebt ihn mit Min-Max-Eingaben (AUC nach 1000 Epochen 0.87 ohne, 0.90 mit Bias). "
               "Mit **zentrierten** Eingaben und ohne Bias kehrt sie sich um (0.94 → 0.31), weil der Ursprung nicht auf c abgebildet werden kann; mit Bias und zentrierten Eingaben fällt sie auf 0.76 (81 Epochen) und erholt sich auf 0.95.")

st.markdown("---")

st.subheader("🔬 Breite und Epochen")
if st.button("Breite × Epochen berechnen (dauert etwa 20 Sekunden)", key="width_start"):
    st.session_state["width_on"] = True
if st.session_state.get("width_on"):
    with st.spinner("Trainiere 5 Breiten × 5 Aufnahmen × 3 Netze..."):
        wt = _width(base_data, settings)
    st.plotly_chart(build_width(wt), width="stretch", key="width_chart")
    st.caption("AUC im Standardfall (Mittel über 5 Datensätze und 3 Netz-Seeds), Zeilen Breite, Spalten Epochen (Momentaufnahmen). Vor dem Training haben alle Breiten 0.98-1.00. Nach 105 Epochen: Breite 4 / 8 / 16 / 32 / 64 gleich 0.96 / 0.97 / 0.92 / 0.90 / 0.87, "
               "nach 600 Epochen 0.86 / 0.87 / 0.90 / 0.88 / 0.84: breitere Netze verlieren früher, und keine Breite verhindert den Abfall.")

st.markdown("---")

st.subheader("🔬 Wo die gelernte Abbildung trifft und wo nicht: acht Szenarien")
if st.button("Die Detektoren in acht Szenarien vergleichen (dauert etwa 40 Sekunden)", key="scenarios_start"):
    st.session_state["scenarios_on"] = True
if st.session_state.get("scenarios_on"):
    with st.spinner("Rechne 8 Szenarien × 5 Aufnahmen..."):
        sc_rows = _scenarios(settings)
    st.plotly_chart(build_scenarios(sc_rows), width="stretch", key="scenarios_chart")
    st.caption("Mittel über 5 feste Datensätze, Deep SVDD mit den Reglern der Seitenleiste (Standard: 100 Epochen, tanh, Breite 32, ν = 0.1), dazu das untrainierte Netz, die One-Class SVM (ν = 0.1, γ₀ und 0.1 · γ₀), Isolation Forest und Wurzel. "
               "**Standardfall**: 0.92 gegen 0.995 untrainiert und 1.00 beim Isolation Forest. **Gekrümmt**: 0.98 (untrainiert 0.99, One-Class SVM 0.98, mit breitem Kernel 0.79). **Dichte Gruppe (30 %)**: 0.18 - unter Raten (untrainiert 0.85, One-Class SVM 0.74, Isolation Forest 0.70). "
               "**Lücke**: 0.45 und 0.53 (untrainiert 0.02 und 0.03; One-Class SVM 0.64 und 0.82). **Korrelationsbruch**: 0.92 (untrainiert 0.70, One-Class SVM 0.88, robust 1.00). **45 % verstreut**: 0.83 (untrainiert 0.995, One-Class SVM 0.32, Isolation Forest 1.00). "
               "**40 Rauschmerkmale**: 0.71 (untrainiert 0.73, One-Class SVM 0.91, mit breitem Kernel 0.99). Rechts der F1 mit bekanntem Anteil: im Standardfall 0.63 (untrainiert 0.90, One-Class SVM mit breitem Kernel 0.95).")

st.markdown("---")

st.subheader("🔬 Die Schwelle")
if st.button("ν über die Schwelle vergleichen (dauert etwa 20 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 7 ν-Werte × 5 Datensätze und drei angenommene Anteile..."):
        tt = _threshold(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**Deep SVDD: F1, Recall, Fehlalarmrate und AUC je ν (Schwelle R²)**")
    c1.plotly_chart(build_cutoff(tt["cutoff"]), width="stretch", key="cutoff_chart")
    c2.markdown("**F1 bei falsch angenommenem Anteil** (½×, 1×, 2× des wahren)")
    c2.plotly_chart(build_wrong_share(tt["wrong_share"]), width="stretch", key="wrong_share_chart")
    st.caption("Links (Standardfall, Voreinstellung): F1 bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 gleich 0.13 / 0.29 / 0.59 / 0.62 / 0.55 / 0.44 / 0.32, die AUC bleibt bei 0.92 - die Schwelle R² markiert **so viele Touren, wie ν vorgibt**, unabhängig von den Daten. "
               "Rechts: ein falsch angenommener Anteil kostet alle Detektoren viel (Deep SVDD 0.59 / 0.63 / 0.55, Isolation Forest 0.67 / 0.97 / 0.67); beim wahren Anteil verliert Deep SVDD schon gegen Isolation Forest und robust, weil die Rangfolge schlechter ist.")

st.markdown("---")

st.subheader("🔬 Kosten")
if st.button("Rechenzeit berechnen (dauert etwa 30 Sekunden)", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    with st.spinner("Messe die Rechenzeit..."):
        ct = _costs(tuple(kv for kv in base_data if kv[0] not in ("n", "n_noise")), settings)
    st.plotly_chart(build_costs(ct["times"]), width="stretch", key="cost_chart")
    st.caption("Mittel über 3 feste Datensätze (Zeiten rechnerabhängig, nur die Größenordnungen zählen). Das Training mit 100 Epochen braucht bei 100 / 300 / 600 Touren und 12 Merkmalen 13 / 28 / 51 ms, mit 1000 Epochen 0.12 / 0.24 / 0.44 s - "
               "das Vierfache bis Achtfache der One-Class SVM (1.7 / 5.9 / 14 ms) und weniger als der Isolation Forest (61 / 126 / 133 ms). Das Netz ist klein (Vollbatch, wenige Tausend Gewichte); bei größeren Netzen und Datensätzen wüchse der Aufwand linear in Epochen, Gewichten und Touren.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Training verbessert die Abbildung** | Im Standardfall verschlechtert es sie: AUC **0.995 vor dem Training**, 0.92 nach 100 und 0.89 nach 1000 Epochen (F1 0.90 → 0.62); bei 45 % Anomalien 0.99 → 0.83, bei einer dichten Gruppe (30 %) 0.80 → **0.18**. Ein zufälliges glattes Netz ist schon ein guter Detektor; das Ziel „Kugel verkleinern“ ist nicht das Ziel „Anomalien finden“. | Isolation Forest, One-Class SVM mit breitem Kernel |
| **Man weiß, wann man aufhören muss** | Das beste Epochen-Zahl hängt vom Szenario ab (Standardfall 0, Korrelationsbruch etwa 100, Lücke etwa 190) und ist ohne Etiketten nicht zu erkennen. Über 10 Netz-Seeds streut die AUC nach 100 Epochen mit 0.05 (schlechtestes Netz 0.76). | (keiner) |
| **Die Eingaben passen zum bias-freien Netz** | Mit zentriert standardisierten Eingaben liegen die Normalen im Ursprung, den das Netz auf 0 abbildet: die AUC sinkt auf 0.48 (300 Epochen) bzw. 0.32 (1000). | Min-Max-Skalierung, Bias |
| **Das bias-freie Netz verhindert den Kollaps** | Es verlangsamt ihn: die Streuung der Ausgaben fällt in allen Konfigurationen (0.082 → 0.004 nach 1000 Epochen); die Rangfolge überlebt es bei Min-Max-Eingaben (0.87). Mit ReLU fällt die AUC nach 1000 Epochen auf 0.49. | Aktivierung, Epochen |
| **ν ist der Anteil, den man kennt** | Die Schwelle R² markiert ν der Touren, unabhängig von den Daten (F1 0.13 bei ν = 0.01, 0.62 bei 0.1, 0.32 bei 0.5). Das Training lernt die Anomalien mit: AUC 0.97 bei 2 %, 0.83 bei 45 % Anomalien. | ECOD, Isolation Forest |
| **Die Struktur der Anomalien ist glatt** | Die Lücke zwischen den Betriebsarten sieht das untrainierte Netz gar nicht (AUC 0.02 und 0.03, unter Raten), das Training hebt sie auf 0.45 und 0.53; die One-Class SVM hat dort 0.64 und 0.82. | One-Class SVM |
| **Viele irrelevante Merkmale** | AUC 0.92 → 0.71 bei 40 Rauschmerkmalen (untrainiert 0.73); die One-Class SVM mit breitem Kernel hält 0.99, der Isolation Forest 0.99. | Feature Bagging, Isolation Forest |
| **Kleine Stichproben** | Bei 20 / 30 Touren AUC 0.72 / 0.82 (untrainiert 0.99 / 0.97). | LOF, Isolation Forest |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie: die Wurzel Elliptic Envelope, LOF, Feature Bagging, Isolation Forest, Extended IF, ECOD und die One-Class SVM (gebaut), ein Autoencoder (noch nicht gebaut). "
    "Keiner ist überlegen: Deep SVDD ersetzt den Kernel durch eine gelernte Abbildung - und ersetzt die Abhängigkeit von γ durch eine von Epochen, Breite, Skalierung, Aktivierung und dem Zufall der Initialisierung."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Netz.** $\phi(x; W) = W_L\,\sigma(W_{L-1}\cdots\sigma(W_1 x))$ mit den Gewichtsmatrizen $W_1, \dots, W_L$ (ohne Bias; $\sigma$ = tanh oder ReLU, die letzte Schicht ist linear). Die Eingaben $x$ sind auf $[0, 1]$ skaliert.

**Zentrum.** $c = \tfrac1n\sum_i \phi(x_i; W^{(0)})$ für die Anfangsgewichte $W^{(0)}$, danach fest; Komponenten mit $|c_k| < \varepsilon = 0.1$ werden auf $\pm\varepsilon$ geschoben (sonst wäre 0 eine triviale Lösung).

**One-Class-Ziel.** $\min_W\ \tfrac1n\sum_i \lVert \phi(x_i; W) - c\rVert^2 + \tfrac{\lambda}{2}\sum_l \lVert W_l\rVert_F^2$ mit $\lambda = 10^{-6}$.
**Soft-Boundary-Ziel.** $\min_{W, R}\ R^2 + \tfrac{1}{\nu n}\sum_i \max\{0, \lVert \phi(x_i; W) - c\rVert^2 - R^2\} + \tfrac{\lambda}{2}\sum_l \lVert W_l\rVert_F^2$; $R^2$ wird nach 10 Epochen alle 5 Epochen als $(1-\nu)$-Quantil der Abstände gesetzt.

**Gradienten.** Mit $\delta_i = 2(\phi(x_i) - c)/n$ am Ausgang (Soft-Boundary: $2(\phi(x_i) - c)\,\mathbb 1[\lVert\phi(x_i) - c\rVert^2 > R^2]/(\nu n)$) läuft die Rückwärtsrechnung von Hand; die Demo prüft sie gegen Finite Differenzen. Optimiert wird mit Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$, Lernrate $10^{-3}$, Vollbatch).

**Wert und Schwelle.** $s(x) = \lVert \phi(x) - c\rVert^2$; Anomalie, wenn $s(x) > R^2$ mit $R^2$ = $(1-\nu)$-Quantil der Trainingsabstände: höchstens ein Anteil ν der Touren liegt strikt darüber.

**Grenzen.** (1) Das Ziel verkleinert die Kugel, es maximiert nicht die Trennung von Anomalien. (2) Ohne Bias bildet das Netz den Ursprung auf 0 ab. (3) Die Schwelle markiert ν der Touren. (4) Das Ergebnis hängt an Epochen, Breite, Aktivierung und Seed.

Implementiert in `dsvdd_algorithm.py` (Netz, Zentrum, Ziele, Backpropagation, Adam), `dsvdd_ocsvm.py` (One-Class SVM des Vorgängers), `dsvdd_lof.py`, `dsvdd_isolation_forest.py` und `dsvdd_ee_algorithm.py` (LOF, Isolation Forest, χ², MCD; wortgleich aus den Vorgängern), `dsvdd_ecod.py` (ECOD als Kontrast),
`dsvdd_scenario.py` (Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen, Korrelationsbruch), `dsvdd_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
