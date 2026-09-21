"""Defaults, Regler-Grenzen und Presets für die Deep-SVDD-Demo (Anomalie-Erkennung an Lieferrouten-Kennzahlen; Szenario, One-Class SVM, LOF, Isolation Forest und Vergleichsschätzer aus den Vorgänger-Demos)."""

# --- Merkmale: die 12 Kennzahlen der PCA-Demo (Name, Einheit, Mittelwert, typische Streuung), dazu Zusatzmerkmale für den Fall n < p --------------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_BASE_FEATURES = len(FEATURES)
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_BASE_FEATURES))
# Reihenfolge, in der die ersten p Merkmale gewählt werden: reihum durch die vier Gruppen, damit schon p = 2 beide latenten Faktoren sieht (Distanz und Zeitfenster-Enge)
FEATURE_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
EXTRA_MEAN, EXTRA_SCALE = 50.0, 10.0                   # Zusatzmerkmale 13 ... p (zufällige Mischungen der latenten Faktoren plus eigenes Rauschen)

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 20, 600
DEFAULT_P = 12
DEFAULT_N_NOISE = 0
N_NOISE_MIN, N_NOISE_MAX = 0, 40
P_MIN, P_MAX = 2, 30
N_MODES_MIN, N_MODES_MAX = 1, 3
DEFAULT_N_MODES = 1
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_CONTAMINATION = 10                             # Prozent
CONTAMINATION_MIN, CONTAMINATION_MAX = 1, 45
KINDS = ("scattered", "cluster", "gap", "decorrelated")
KIND_LABELS = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke zwischen den Betriebsarten", "decorrelated": "Korrelationsbruch (Randverteilungen normal)"}
DEFAULT_KIND = "scattered"
DEFAULT_STRENGTH = 6.0
STRENGTH_MIN, STRENGTH_MAX = 3.0, 12.0
DEFAULT_SUPPORT = 0.5                                   # Stützanteil h/n (0.5 = größter Bruchpunkt); 1.0 = alle Punkte = klassische Schätzung
SUPPORT_MIN, SUPPORT_MAX = 0.5, 1.0
DEFAULT_QUANTILE = 0.975                                # chi^2-Quantil der Schwelle
QUANTILE_MIN, QUANTILE_MAX = 0.90, 0.999
DEFAULT_REWEIGHT = True
DEFAULT_SEED = 7

# --- Deep SVDD und Vergleichsdetektoren -----------------------------------------------------------------------------------------
DEFAULT_VARIANT = "one_class"
VARIANTS = ("one_class", "soft_boundary")
VARIANT_LABELS = {"one_class": "One-Class (Kugel um c)", "soft_boundary": "Soft-Boundary (Radius R, ν)"}
DEFAULT_WIDTH, WIDTH_OPTIONS = 32, (4, 8, 16, 32, 64)
DEFAULT_DEPTH, DEPTH_MIN, DEPTH_MAX = 2, 0, 3
DEFAULT_N_OUT, N_OUT_MIN, N_OUT_MAX = 8, 1, 16
DEFAULT_EPOCHS, EPOCHS_MIN, EPOCHS_MAX = 100, 0, 1000
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_ACTIVATION = "tanh"
ACTIVATIONS = ("tanh", "relu")
DEFAULT_BIAS = False
DEFAULT_SCALING = "minmax"                              # Eingabeskalierung: Min-Max [0, 1] (bias-freies Netz kann so c erreichen) oder standardisiert (zentriert: der Ursprung liegt mitten in den Daten)
SCALINGS = ("minmax", "standard")
SCALING_LABELS = {"minmax": "Min-Max [0, 1]", "standard": "standardisiert (zentriert)"}
DEFAULT_NU = 0.10                                       # Anteil der Trainingstouren außerhalb der Kugel (Schwelle R^2 = (1 - nu)-Quantil der Abstände)
NU_MIN, NU_MAX = 0.01, 0.50
DEFAULT_NET_SEED = 0
# One-Class SVM des Vorgängers im Vergleich
OCSVM_NU = 0.10
OCSVM_GAMMA_FACTOR = 1.0
DEFAULT_TREES = 100                                     # Isolation Forest im Vergleich
DEFAULT_PSI = 256
DEFAULT_CUTOFF_IF = 0.5                                 # nominelle Score-Schwelle des Isolation Forest
DEFAULT_CUTOFF_LOF = 1.5                                # LOF-Schwelle im Vergleich
DEFAULT_LOF_K = 20                                      # LOF im Vergleich: k = 20 (höchstens n / 2)
THRESHOLD_KINDS = ("standard", "share")
THRESHOLD_LABELS = {"standard": "Standard (R² bzw. f < 0 bzw. Score 0.5 bzw. χ²-Quantil)", "share": "erwarteter Anteil (für alle vier)"}
DEFAULT_THRESHOLD_KIND = "standard"
DEFAULT_SHARE = 10                                      # angenommener Anteil der Anomalien [%] (= der wahre im Standardfall)
SHARE_MIN, SHARE_MAX = 1, 45
SMO_TOL = 1e-6
FLAG_EPS = 1e-6
SMO_MAX_ITER = 100_000

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
Q = 2                                                   # latente Faktoren (fest; die PCA-Demo variiert sie, hier geht es um Anomalien)
CROSS_LOADING = 0.15
WITHIN_LOADINGS = (0.95, 0.9, 0.85)
CURVATURE_FREQUENCY = 1.6
CURVATURE_AMPLITUDE = 2.0
LAYOUT_SEED = 20240915                                  # dieselben festen Matrizen wie in der PCA-Demo
EXTRA_LAYOUT_SEED = LAYOUT_SEED + 2
MODE_RADIUS = 2.2                                       # Betriebsarten liegen auf einem Kreis dieses Radius im Faktorraum
MODE_SD = 0.6                                           # Streuung innerhalb einer Betriebsart (bei nur einer Betriebsart 1, wie in der PCA-Demo)
CLUSTER_SD = 0.3                                        # Streuung der dichten Anomalie-Gruppe
GAP_SD = 0.3                                            # Streuung der Anomalien in der Lücke
CLUSTER_ANGLE = 0.6                                     # Richtung der dichten Gruppe im Faktorraum (Bogenmaß)

# --- Auswertung --------------------------------------------------------------------------------------------------------
MCD_STARTS = 500                                        # zufällige Startmengen (je zwei C-Schritte)
MCD_KEEP = 10                                           # die besten davon laufen bis zur Konvergenz
MCD_INITIAL_STEPS = 2
MCD_MAX_STEPS = 50
RIDGE = 1e-9                                            # relative Regularisierung der Kovarianz (n < p)
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(n=DEFAULT_N_TOURS, p=DEFAULT_P, n_noise=DEFAULT_N_NOISE, n_modes=DEFAULT_N_MODES, curvature=DEFAULT_CURVATURE, noise=DEFAULT_NOISE, contamination=DEFAULT_CONTAMINATION,
                kind=DEFAULT_KIND, strength=DEFAULT_STRENGTH, variant=DEFAULT_VARIANT, width=DEFAULT_WIDTH, depth=DEFAULT_DEPTH, n_out=DEFAULT_N_OUT, epochs=DEFAULT_EPOCHS, nu=DEFAULT_NU,
                activation=DEFAULT_ACTIVATION, bias=DEFAULT_BIAS, scaling=DEFAULT_SCALING, net_seed=DEFAULT_NET_SEED, threshold_kind=DEFAULT_THRESHOLD_KIND, quantile=DEFAULT_QUANTILE, share=DEFAULT_SHARE,
                seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Ohne Training (0 Epochen)": _preset(epochs=0),
    "ReLU, 1000 Epochen": _preset(activation="relu", epochs=1000),
    "Zentrierte Eingaben": _preset(scaling="standard", epochs=300),
    "Gekrümmter Normalbereich": _preset(curvature=1.0),
    "Korrelationsbruch": _preset(kind="decorrelated"),
    "Viele Anomalien (45 %)": _preset(contamination=45),
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Im Mittel über fünf Aufnahmen: 300 Touren, 12 Merkmale, 10 % verstreute Anomalien, ein Netz mit zwei verdeckten Schichten der Breite 32 (tanh, 8 Ausgaben, ohne Bias, Min-Max-Skalierung), 100 Epochen. AUC 0.92, F1 0.62 - "
                                    "das **untrainierte** Netz derselben Größe hat AUC 0.995 und F1 0.90: das Training verschlechtert die Rangfolge. Isolation Forest 1.00 / 0.96, One-Class SVM (γ₀) 0.95 / 0.67, mit breitem Kernel 1.00 / 0.95.",
    "Ohne Training (0 Epochen)": "Im Mittel über fünf Aufnahmen: dasselbe Netz, aber 0 Epochen - das zufällig initialisierte Netz mit dem Zentrum c aus seinen eigenen Ausgaben. AUC 0.995 und F1 0.90 (Recall 0.89, Fehlalarmrate 0.9 %): "
                                "so gut wie der Isolation Forest (1.00 / 0.96). Der Abstand zum Zentrum in einem zufälligen glatten Merkmalsraum ist schon ein guter Detektor - das Training fügt hier nichts hinzu.",
    "ReLU, 1000 Epochen": "Im Mittel über fünf Aufnahmen: ReLU statt tanh, 1000 Epochen. AUC 0.49 (Raten) und F1 0.13, das untrainierte Netz hat 0.98: die Ausgaben ziehen sich auf eine Streuung von 0.016 zusammen, und die Rangfolge geht verloren. "
                          "Mit ReLU ist das Training nach 30 Epochen noch gut (AUC 0.99) und danach nicht mehr.",
    "Zentrierte Eingaben": "Im Mittel über fünf Aufnahmen: die Kennzahlen zentriert standardisiert (Mittelwert 0) statt auf [0, 1] skaliert, 300 Epochen. AUC 0.48, F1 0.19, das untrainierte Netz hat 0.95. Ein Netz ohne Bias bildet den Ursprung immer auf 0 ab: "
                           "die Normalen liegen genau dort und können dem Zentrum nie näher kommen - nur die Anomalien können auf c gezogen werden. Die Rangfolge kehrt sich um (nach 1000 Epochen AUC 0.32).",
    "Gekrümmter Normalbereich": "Im Mittel über fünf Aufnahmen: die normale Fläche ist gebogen (Krümmung 1). AUC 0.98, F1 0.83 (Fehlalarmrate 1.7 %) - hier schadet das Training kaum (untrainiert 0.99 / 0.90); "
                                "One-Class SVM (γ₀) 0.98 / 0.85, mit breitem Kernel nur 0.79 / 0.73.",
    "Korrelationsbruch": "Im Mittel über fünf Aufnahmen: die Merkmale der Anomalien sind je Spalte aus den Normalen neu gezogen. Deep SVDD hat AUC 0.92 (F1 0.54) - und hier **hilft** das Training: untrainiert nur 0.70. "
                         "LOF 0.99 und robust 1.00 sind besser, One-Class SVM 0.88, Isolation Forest 0.80, ECOD 0.56.",
    "Viele Anomalien (45 %)": "Im Mittel über fünf Aufnahmen: 45 % Anomalien, ν = 0.1. AUC 0.83, F1 0.34 - das untrainierte Netz hat 0.995: das Training lernt die Anomalien als normal (bei 2 % Anomalien AUC 0.97, bei 45 % 0.83). "
                              "Die One-Class SVM bricht ganz ein (0.32), Isolation Forest und ECOD haben 1.00.",
}
# Bänder (Seed des Presets; mit dem ausgelieferten Code kalibriert, bewusst weit): Kennzahlen der Detektoren (dsvdd_*, ocsvm_*, iforest_*, robust_*, untrained_*) und erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"dsvdd_auc": (0.8, 0.99), "untrained_auc": (0.95, 1.0), "iforest_auc": (0.98, 1.0), "verdict": ("training_hurts", "comparable")},
    "Ohne Training (0 Epochen)": {"dsvdd_auc": (0.97, 1.0), "dsvdd_f1": (0.75, 1.0), "verdict": ("comparable",)},
    "ReLU, 1000 Epochen": {"dsvdd_auc": (0.2, 0.75), "untrained_auc": (0.9, 1.0), "verdict": ("training_hurts",)},
    "Zentrierte Eingaben": {"dsvdd_auc": (0.2, 0.7), "untrained_auc": (0.85, 1.0), "verdict": ("training_hurts",)},
    "Gekrümmter Normalbereich": {"dsvdd_auc": (0.93, 1.0), "dsvdd_f1": (0.65, 1.0), "verdict": ("comparable", "training_hurts")},
    "Korrelationsbruch": {"dsvdd_auc": (0.8, 1.0), "untrained_auc": (0.4, 0.85), "robust_auc": (0.95, 1.0), "verdict": ("others_win",)},
    "Viele Anomalien (45 %)": {"dsvdd_auc": (0.6, 0.95), "untrained_auc": (0.95, 1.0), "ocsvm_auc": (0.0, 0.6), "verdict": ("training_hurts",)},
}
