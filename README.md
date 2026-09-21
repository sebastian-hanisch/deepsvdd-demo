# Deep SVDD – eine gelernte Abbildung und eine Kugel um die Normalen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-deepsvdd-demo.streamlit.app/)**

Achtes Stück der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Deep SVDD** (Ruff, Vandermeulen, Görnitz, Deecke, Siddiqui, Binder, Müller und Kloft) – an einem wachsenden Beispiel, gegen die [One-Class SVM](../ocsvm-demo),
den [Isolation Forest](../isolation-forest-demo) und die robuste Schätzung der Wurzel ([elliptic-envelope-demo](../elliptic-envelope-demo)); als Bezug laufen das **untrainierte** Netz, die One-Class SVM mit breitem Kernel, der [LOF](../lof-demo) und [ECOD](../ecod-demo) mit.
Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in den Vorgängern und der [pca-demo](../pca-demo) (Szenario, One-Class SVM, LOF, Isolation Forest, Wurzel-Schätzer und ECOD-Kern wortgleich übernommen, per Test gegen eingefrorene Werte geprüft; die klassische Schätzung entfällt, damit die Balken lesbar bleiben).

**Einordnung in die Reihe (die Kanten des Graphen):** Deep SVDD ist die **Fortsetzung des One-Class-SVM-Astes**. Die One-Class SVM hatte eine gemessene Schwäche: ihr Ergebnis hing an der Kernel-Breite γ (F1 0.95 oder 0.00 je nach γ) und an ν.
Deep SVDD ersetzt den festen Kernel durch eine **gelernte Abbildung** (neuronales Netz ohne Bias, Zentrum c aus dem initialen Netz, Ziel: die Normalen in eine möglichst kleine Kugel um c ziehen). Die Frage der Demo: **verschwindet die Abhängigkeit von einem Regler – oder zieht sie nur um?**
Ergebnis in Kürze: **sie zieht um – und das Training selbst schadet.** Das **untrainierte** Zufallsnetz hat AUC 0.995 und F1 0.90 – so gut wie der Isolation Forest –, nach 100 Epochen Training sind es 0.92 und 0.62, nach 1000 Epochen 0.89. Das Ziel „Kugel verkleinern“ ist nicht das Ziel „Anomalien finden“.
Nur wo die Abbildung erst gelernt werden muss (Korrelationsbruch, Lücke), hilft das Training. Die Linie hat **keinen Konvergenzpunkt**; der letzte Nachbar, der [autoencoder-anomalie-demo](../autoencoder-anomalie-demo), ist gebaut (Rekonstruktionsfehler; die Linie ist damit komplett).
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ecod-demo                  (Kontrast: verteilungsfrei)                              [gebaut]
  ├─ lof-demo → feature-bagging-demo (lokale Dichte; Ensembles gegen viele Merkmale)     [beide gebaut]
  ├─ ocsvm-demo → deepsvdd-demo (gelernte Grenze; gelernte Abbildung)                    [beide gebaut; dieses Stück: Deep SVDD]
  ├─ isolation-forest-demo → extended-isolation-forest-demo (Zufallsbäume)               [beide gebaut]
  └─ autoencoder-anomalie-demo  (Rekonstruktionsfehler)                                  [gebaut]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25; Netz mit 2 verdeckten Schichten der Breite 32, tanh, 8 Ausgaben, ohne Bias, Min-Max-Skalierung, 100 Epochen (Adam, Lernrate 0.001, Vollbatch), ν = 0.1 (Schwelle R² = 0.9-Quantil der Trainingsabstände), Netz-Seed 0; One-Class SVM ν = 0.1 und γ = 1 / p; Isolation Forest 100 Bäume × ψ = 256 und Schwelle 0.5, robust χ²-Quantil 0.975; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall, Voreinstellung | ❌ **Das Training schadet.** Deep SVDD AUC **0.92**, F1 **0.62** (Recall 0.61, Fehlalarmrate 3.9 %) – das **untrainierte** Netz (dasselbe Netz und Zentrum vor dem ersten Schritt): AUC **0.995**, F1 **0.90**. Isolation Forest 1.00 / 0.96, One-Class SVM (γ₀) 0.95 / 0.67, mit breitem Kernel (0.1 · γ₀) 1.00 / 0.95, robust 1.00 / 0.84. Mit bekanntem Anteil 0.63 (untrainiert 0.90) |
| **Epochen** | ❌ AUC bei 0 / 10 / 30 / 100 / 300 / 1000 Epochen: **0.995 / 0.99 / 0.95 / 0.92 / 0.91 / 0.89**, F1 0.90 / 0.86 / 0.77 / 0.62 / 0.61 / 0.63, Streuung der Ausgaben 0.081 / 0.048 / 0.037 / 0.019 / 0.010 / 0.003: die Kugel wird kleiner, die Rangfolge schlechter. Die γ-Abhängigkeit der One-Class SVM ist zu einer Abhängigkeit von der **Trainingsdauer** geworden |
| Breite, Tiefe, Ausgaben | ➖ AUC nach 100 Epochen bei Breite 4 / 8 / 16 / 32 / 64: 0.98 / 0.98 / 0.95 / 0.92 / 0.88 (untrainiert 0.99–1.00: breitere Netze verlieren mehr); Tiefe 0 / 1 / 2 / 3: 0.98 / 0.92 / 0.92 / 0.95; Ausgaben 1 / 2 / 4 / 8 / 16: 0.70 / 0.79 / 0.82 / 0.92 / 0.97 (untrainiert 0.66 / 0.93 / 0.96 / 0.99 / 0.98). Soft-Boundary hat keinen Vorteil (AUC nach 100 / 300 / 1000 Epochen 0.89 / 0.83 / 0.83 gegen 0.92 / 0.91 / 0.89) |
| **Zufall der Initialisierung** | ⚠️ Über 10 Netz-Seeds und 5 Datensätze (50 Paare): AUC nach 0 / 30 / 100 / 300 Epochen **0.99 ± 0.02 / 0.95 ± 0.02 / 0.91 ± 0.05 / 0.89 ± 0.05**, schlechtestes Netz 0.91 / 0.90 / 0.76 / 0.76, bestes 1.00 / 0.99 / 0.98 / 0.97; F1 0.90 ± 0.08 / 0.75 ± 0.06 / 0.65 ± 0.12 / 0.61 ± 0.12. Je länger trainiert wird, desto stärker hängt das Ergebnis an den Anfangsgewichten – die One-Class SVM ist deterministisch |
| **Bias und Skalierung** | ⚠️ **Die Ausgaben ziehen sich in allen vier Konfigurationen zusammen** (Streuung nach 1000 Epochen 0.004 ohne, 0.002 mit Bias bei Min-Max; 0.023 und 0.009 zentriert): das bias-freie Netz verhindert den Kollaps nicht, es verlangsamt ihn. Die **Rangfolge** überlebt ihn mit Min-Max-Eingaben (AUC nach 1000 Epochen 0.87 ohne, 0.90 mit Bias). Mit **zentriert standardisierten Eingaben** und ohne Bias **kehrt sie sich um**: 0.94 → 0.58 (100 Epochen) → 0.48 (300) → **0.32** (1000) – der Mittelpunkt der Daten wird immer auf 0 abgebildet, die Normalen können dem Zentrum nie nahe kommen |
| Aktivierung | ❌ ReLU: AUC nach 0 / 10 / 30 / 100 / 300 / 1000 Epochen 0.98 / 0.99 / 0.99 / 0.92 / 0.73 / **0.49** (tanh 0.995 / 0.99 / 0.95 / 0.92 / 0.91 / 0.89): anfangs besser, später unter Raten |
| Viele Anomalien / dichte Gruppe | ❌ **Das Training lernt die Anomalien mit:** AUC bei 2 / 5 / 10 / 20 / 30 / 40 / 45 % verstreuten Anomalien 0.97 / 0.95 / 0.92 / 0.89 / 0.86 / 0.84 / 0.83, das untrainierte Netz 0.99–1.00 überall. **Dichte Gruppe (30 %): 0.18 – unter Raten** (untrainiert 0.85, One-Class SVM 0.74, mit breitem Kernel 0.84, Isolation Forest 0.70, robust 0.49, LOF 0.47, ECOD 0.84) |
| Korrelationsbruch | ✅ **Hier hilft das Training:** AUC 0.92 (F1 0.54), untrainiert nur 0.70; Verlauf 0.65 → 0.91 (105 Epochen) → 0.79 (600). One-Class SVM 0.88, Isolation Forest 0.80, ECOD 0.56 – aber LOF 0.99 und robust 1.00 sind besser |
| Lücke zwischen Betriebsarten | ⚠️ AUC 0.45 (2 Betriebsarten) und 0.53 (3), untrainiert 0.02 und 0.03 (unter Raten: die Lücke liegt im Innern der glatten Abbildung); das Training hebt sie auf höchstens 0.47 / 0.59 (nach 188 Epochen), nie über Raten hinaus. One-Class SVM 0.64 / 0.82, Isolation Forest 0.54 / 0.27, robust 0.40 / 0.38 |
| Gekrümmter Normalbereich | ➖ AUC bei Krümmung 0 / 0.25 / 0.5 / 0.75 / 1: 0.92 / 0.96 / 0.97 / 0.98 / 0.98, F1 0.62 → 0.83 (untrainiert 0.99 / 0.90): je stärker die Krümmung, desto weniger schadet das Training; One-Class SVM 0.98 / 0.85, mit breitem Kernel nur 0.79 / 0.73 |
| Rauschmerkmale | ❌ AUC bei 0 / 5 / 10 / 20 / 30 / 40 Rauschmerkmalen 0.92 / 0.86 / 0.87 / 0.73 / 0.72 / 0.71 (untrainiert 0.995 / 0.97 / 0.90 / 0.77 / 0.83 / 0.73); One-Class SVM mit breitem Kernel 0.99 (F1 0.92), Isolation Forest 0.99 |
| Schwelle | ➖ R² markiert **so viele Touren, wie ν vorgibt**, unabhängig von den Daten: F1 bei ν = 0.01 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 gleich 0.13 / 0.29 / 0.59 / 0.62 / 0.55 / 0.44 / 0.32 (AUC bleibt 0.92; Recall 0.07 … 0.94, Fehlalarmrate 0 % … 45 %). Falsch angenommener Anteil (½× / 1× / 2×): Deep SVDD 0.59 / 0.63 / 0.55, Isolation Forest 0.67 / 0.97 / 0.67 |
| Kleine Stichproben | ❌ AUC bei 20 / 30 / 50 / 100 Touren 0.72 / 0.82 / 0.91 / 0.94 (untrainiert 0.99 / 0.97 / 0.99 / 1.00) |
| Rechenzeit | ➖ Training mit 100 Epochen bei 100 / 300 / 600 Touren und 12 Merkmalen 13 / 28 / 51 ms, mit 1000 Epochen 0.12 / 0.24 / 0.44 s – das Vier- bis Achtfache der One-Class SVM (1.7 / 5.9 / 14 ms), weniger als der Isolation Forest (61 / 126 / 133 ms) – bei einem kleinen Netz mit Vollbatch |

## Was die Demo zeigt

1. **Deep SVDD in Aktion** (Schritt-Slider + Abspielen): **Touren** (Ebene der größten Streuung und zwei Rohmerkmale) → **Abbildung** (Ausgaben des zufällig initialisierten Netzes, Abweichung vom Zentrum c auf zwei feste Achsen projiziert, mit Schwelle R; Abstandsverteilung – schon hier AUC 0.995) →
   **Training** (die Ausgaben nach 0, ~10, ~100 und den letzten Epochen auf denselben Achsen: die Punktwolke zieht sich zusammen; Verlauf der Zielfunktion, AUC und Streuung; die gelernte Kugel in der Ebene zweier Rohmerkmale, dafür neu trainiert) →
   **Wert** (Verteilung des Abstands mit Schwelle R²; Rang bei Deep SVDD gegen Rang bei der Wurzel) → **Ergebnis** (Kennzahlen und ROC-Kurven der vier Detektoren).
2. **Was die Detektoren gefunden haben:** AUC, Recall, Fehlalarmrate, F1 (dazu der F1 mit bekanntem Anteil), Urteil (Codes: Lücke → **Training schadet** (das untrainierte Netz ist um mindestens 0.05 AUC besser) → ν kleiner als der Anteil → anderer Detektor besser → Deep SVDD besser → falsche Schwelle → ebenbürtig), Detailtabelle mit Rechenzeiten.
3. **📐 Sweeps** über Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, **Epochen, Breite, Tiefe, Ausgabedimension, ν**, χ²-Quantil und angenommenen Anteil (feste Datensätze ab 100000, Streuung).
4. **🔬 Experimente auf Abruf:** AUC über die Epochen in acht Szenarien, 10 Netz-Seeds, Bias × Eingabeskalierung über 1000 Epochen, Breite × Epochen, acht Szenarien × Detektoren (mit untrainiertem Netz, One-Class SVM mit breitem Kernel), Schwelle (ν und falsch angenommener Anteil), Kosten.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (das Training verbessert die Abbildung, man weiß, wann man aufhören muss, Eingaben passen zum bias-freien Netz, das bias-freie Netz verhindert den Kollaps, ν, glatte Struktur, irrelevante Merkmale, kleine Stichproben).

Regler: Touren (20–600), Merkmale (2–30), Rauschmerkmale (0–40), Betriebsarten (1–3), Krümmung, Rauschen, Anteil der Anomalien (1–45 %), Art (verstreut / dichte Gruppe / in der Lücke – ab zwei Betriebsarten / Korrelationsbruch), Abstand (bei "Lücke" und "Korrelationsbruch" ausgeblendet, Wert bleibt erhalten),
**Variante** (One-Class / Soft-Boundary), **Breite** (4–64), **Tiefe** (0–3), **Ausgaben** (1–16), **Epochen** (0–1000; 0 = das untrainierte Netz), **ν**, **Aktivierung** (tanh / ReLU), **Bias** (an / aus), **Eingabeskalierung** (Min-Max / zentriert standardisiert), **Netz-Seed** (+ 🎲, getrennt vom Seed der Touren),
**Schwelle** (Standard: R² und χ²-Quantil der Wurzel, ausgeblendet beim erwarteten Anteil; sonst der angenommene Anteil für alle vier).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC DS | F1 DS | AUC unt. | F1 unt. | AUC OCSVM | F1 OCSVM | AUC IF | F1 IF | AUC robust | F1 robust | Urteil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Standardfall (Voreinstellung) | 0.94 | 0.71 | 1.00 | 0.98 | 0.97 | 0.69 | 1.00 | 0.98 | 1.00 | 0.85 | Training schadet |
| Ohne Training (0 Epochen) | 1.00 | 0.98 | 1.00 | 0.98 | 0.97 | 0.69 | 1.00 | 0.98 | 1.00 | 0.85 | ebenbürtig |
| ReLU, 1000 Epochen | 0.51 | 0.24 | 0.98 | 0.88 | 0.97 | 0.69 | 1.00 | 0.98 | 1.00 | 0.85 | Training schadet |
| Zentrierte Eingaben | 0.47 | 0.20 | 0.99 | 0.85 | 0.97 | 0.69 | 1.00 | 0.98 | 1.00 | 0.85 | Training schadet |
| Gekrümmter Normalbereich | 0.98 | 0.85 | 0.99 | 0.95 | 1.00 | 0.85 | 1.00 | 0.97 | 1.00 | 0.39 | ebenbürtig |
| Korrelationsbruch | 0.93 | 0.64 | 0.71 | 0.14 | 0.86 | 0.32 | 0.78 | 0.23 | 1.00 | 0.84 | anderer Detektor besser |
| Viele Anomalien (45 %) | 0.89 | 0.34 | 0.99 | 0.35 | 0.37 | 0.26 | 1.00 | 0.83 | 0.96 | 0.85 | Training schadet |

## Modell und Verfahren

- **Szenario** (`dsvdd_scenario.py`): wortgleich aus den Vorgängern (zwei versteckte Faktoren, 12 Kennzahlen der PCA-Demo, Betriebsarten, Krümmung, Anomalien verstreut / dichte Gruppe / in der Lücke / Korrelationsbruch mit exaktem Anteil, Zusatzmerkmale bis p = 30, Rauschmerkmale).
- **Deep SVDD** (`dsvdd_algorithm.py`, numpy von Grund auf): bias-freies MLP (Xavier bzw. He, linearer Ausgang), Zentrum c = Mittel der Ausgaben des initialen Netzes (Komponenten unter 0.1 auf ±0.1), One-Class-Ziel (mittlerer quadrierter Abstand + Gewichtsabfall 1e-6) und Soft-Boundary-Ziel (R² + Hinge, R² alle 5 Epochen als (1 − ν)-Quantil),
  **Backpropagation von Hand** (gegen Finite Differenzen geprüft), Adam, Vollbatch, Momentaufnahmen der Ausgaben. Eingaben Min-Max [0, 1] (oder zentriert standardisiert, um die Falle zu zeigen). Schwelle R² = (1 − ν)-Quantil der Trainingsabstände.
- **One-Class SVM** (`dsvdd_ocsvm.py`, SMO-Löser, gegen scikit-learn geprüft), **Isolation Forest**, **LOF**, **ECOD** und **Wurzel** (`dsvdd_isolation_forest.py`, `dsvdd_lof.py`, `dsvdd_ecod.py`, `dsvdd_ee_algorithm.py`): wortgleich aus den Vorgängern.
- **Auswertung** (`dsvdd_evaluation.py`): AUC, mittlere Präzision, Precision, Recall, F1, Fehlalarmrate für vier Detektoren und vier Bezugsdetektoren; **F1 mit bekanntem Anteil** als Referenz für die Schwelle; Sweeps, Experiment-Tabellen (Epochenverlauf, Netz-Seeds, Kollaps, Breite, Kosten), Urteil.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen (vor dem Bau gemessen):** (1) "Die gelernte Abbildung beseitigt die γ-Abhängigkeit nicht, sondern verschiebt sie" – **bestätigt, und schärfer als gedacht**: sie verschiebt sie auf Epochen, Breite, Aktivierung, Skalierung und den Netz-Seed, und der Regler „Epochen“ hat sein Optimum bei **null**.
  (2) "Bias führt zum Kollaps, AUC ≈ 0.5" – **widerlegt für die Rangfolge**: mit Bias überlebt sie (AUC 0.90 nach 1000 Epochen), und die Ausgabestreuung fällt auch **ohne** Bias auf 0.004; nur die Kombination zentrierter Eingaben und bias-freies Netz zerstört sie (0.32).
  (3) "Bei guten Reglern so gut wie die One-Class SVM mit breitem Kernel" – **widerlegt**: das beste Deep SVDD ist das untrainierte (0.995 / 0.90), die One-Class SVM mit breitem Kernel hat 1.00 / 0.95. (4) "Gekrümmt / Betriebsarten / Korrelationsbruch: die Abbildung folgt der Struktur besser als der feste Kernel" – **nur beim Korrelationsbruch** (0.92 gegen 0.88 bei γ₀, nach 105 Epochen 0.91).
  (5) "Lücke: die glatte Abbildung sieht sie nicht" – **bestätigt** (untrainiert 0.02 / 0.03), das Training hebt sie auf 0.47 / 0.59, nie über Raten. (6) "Rauschmerkmale schaden weniger als bei der Kernel-Distanz" – **widerlegt**: 0.71 bei 40 Rauschmerkmalen, die One-Class SVM mit breitem Kernel hat 0.99.
  (7) "ν wie bei der One-Class SVM: ν < Anteil zieht die Anomalien mit" – **anders**: die Rangfolge hängt nicht an ν (die Schwelle nur), aber das **Training** lernt die Anomalien mit (0.83 bei 45 %, dichte Gruppe 0.18). (8) "Kleine Stichproben: Überanpassung, riesige Streuung" – bestätigt (0.72 bei 20 Touren). (9) "Training kostet den Faktor 100–1000" – **widerlegt**: das Vier- bis Achtfache der One-Class SVM bei kleinem Netz und Vollbatch. (10) "Anderer Netz-Seed → merklich andere AUC" – bestätigt (± 0.05 nach 100 Epochen).
- **Eine erste Fassung mit zentriert standardisierten Eingaben täuschte:** AUC 0.48 nach 300 Epochen, unter Raten. Ursache war nicht das Training, sondern der Ursprung: ein bias-freies Netz bildet den Mittelpunkt der Daten immer auf 0 ab, das Zentrum liegt mindestens 0.1 je Ausgabe daneben, und nur die Anomalien am Rand können auf c gezogen werden.
  Mit Min-Max-Eingaben (wie im Paper für Bilder) verschwindet das; die Demo zeigt die Falle als Regler.
- **Das untrainierte Netz ist der ehrliche Bezugspunkt:** der Abstand zum Zentrum in einem zufälligen glatten Merkmalsraum ist schon ein guter Detektor (auf diesem Szenario mit niedrigdimensionaler latenter Struktur; bei 40 Rauschmerkmalen nur 0.73). Ohne Etiketten weiß man nicht, wann das Training aufhören müsste (bester Punkt: Standardfall 0 Epochen, Korrelationsbruch etwa 100, Lücke etwa 190).
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie; die Rauschmerkmale sind unabhängige Spalten. Kleines Netz, Vollbatch, feste Lernrate 0.001 und Gewichtsabfall 1e-6 (mit 1e-3 keine Änderung); das Ergebnis gilt für dieses Szenario und diese Netzgrößen. Literatur nur mit Namen: Ruff, Vandermeulen, Görnitz, Deecke, Siddiqui, Binder, Müller und Kloft (Deep SVDD).

## Verifikation

- Deep SVDD: **Gradienten gegen Finite Differenzen** (One-Class und Soft-Boundary × tanh und ReLU × Tiefe 0–2 × mit und ohne Bias: 24 Konfigurationen), Aktivierungsableitungen, das Netz hat keine Bias-Terme und bildet 0 auf 0 ab, Zentrum = Mittel des initialen Netzes (fest, eps-Regel), Schwelle = (1 − ν)-Quantil (markiert höchstens ν),
  beide Werte-Varianten, Zielfunktion und Streuung fallen beim Training, Soft-Boundary-Hinge (großer Radius: kein Gradient), 0 Epochen = untrainiertes Netz, Determinismus und Netz-Seed, die Falle zentrierter Eingaben (der Datenmittelpunkt landet exakt auf 0), Bias verkleinert die Streuung weiter, ungültige Argumente abgelehnt.
- Übernommene Bausteine: One-Class SVM (gegen `sklearn.svm.OneClassSVM`), LOF, Isolation Forest, Wurzel-Schätzer (χ² gegen scipy), ECOD-Schwanzwahrscheinlichkeiten. Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, `n_noise` ändert nur angehängte Spalten, exakter Anomalie-Anteil, Geometrie der Arten, Korrelationsbruch.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Epochen-, Netz-Seed-, Kollaps-, Breite-, Szenarien-, Schwellen- und Kostentabellen; jeweils Mittel über die festen Sweep-Datensätze, positive **und** negative Aussagen; **weite Bänder für trainierte Netze** wegen numpy-/BLAS-Rundung, enge für das untrainierte; Rechenzeiten nur als Größenordnung);
  alle 7 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, beide Varianten × jede Art der Anomalien, Extremwerte aller Netzregler, Abspielen ohne doppelte Schlüssel, Netz-Seed-Knopf und Bias-Kästchen, ausgeblendete Regler behalten ihre Werte, Sweep-Optionen, Experimente auf Abruf),
  Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Epochenverlauf, Netz-Seeds, Bias × Skalierung, Breite × Epochen, Szenarien, Schwelle, Kosten), 🚧 Grenzen, Mathe |
| `dsvdd_algorithm.py` | Bias-freies Netz, Zentrum, One-Class- und Soft-Boundary-Ziel, Backpropagation von Hand, Adam, Schwelle |
| `dsvdd_ocsvm.py`, `dsvdd_lof.py`, `dsvdd_isolation_forest.py`, `dsvdd_ee_algorithm.py`, `dsvdd_ecod.py` | One-Class SVM, LOF, Isolation Forest, χ²-Verteilung, klassische Schätzung, FastMCD, ECOD (wortgleich aus den Vorgängern) |
| `dsvdd_scenario.py`, `dsvdd_constants.py` | Touren mit Betriebsarten, Krümmung, Anomalien (auch Korrelationsbruch) und Rauschmerkmalen; Konstanten, Presets |
| `dsvdd_evaluation.py` | Kennzahlen, Analyse, Schwellen, Sweeps, Experimente, Urteil |
| `dsvdd_presets.py`, `dsvdd_visualization.py` | Permalink/Presets (ausgeblendete Regler), Plotly-Figuren (achsengesperrt) |
| `tests/` | Deep SVDD (Finite-Differenzen-Gradienten, Zentrum, Falle, Kollaps), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
