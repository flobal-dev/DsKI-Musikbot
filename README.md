# DsKI-Musikbot
Datensatz zum download gibt es hier: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification - Projekt zu finden auf master
# 🎵 Musikgenre-Klassifikation

Ein Machine-Learning-Projekt zur automatischen Klassifikation von Musikgenres auf Basis von Audiofeatures. Entwickelt in drei iterativen Versionen – diese Version (V3) erreicht eine deutlich verbesserte Genauigkeit gegenüber der Baseline durch sauberes Daten-Handling und ein optimiertes Ensemble-Modell.

---

## Inhaltsverzeichnis

- [Überblick](#überblick)
- [Features](#features)
- [Modelle](#modelle)
- [Ergebnisse](#ergebnisse)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Projektstruktur](#projektstruktur)
- [Versionshistorie](#versionshistorie)

---

## Überblick

Das Projekt klassifiziert Musikstücke anhand von akustischen Merkmalen in 10 Genres (z. B. Blues, Classical, Jazz, Metal, Rock …). Als Datenbasis dient der **GTZAN Genre Collection Dataset** (~1000 WAV-Dateien, 30 Sekunden je Track).

Der komplette Pipeline-Ablauf:

```
WAV-Dateien → Feature Extraction → Train/Test-Split → Augmentation → Training → Evaluation
```

---

## Features

Aus jedem Track werden **85 Features** extrahiert:

| Feature | Dimensionen | Beschreibung |
|---|---|---|
| MFCC (Mean + Std) | 20 × 2 | Klangfarbe / Timbre |
| Chroma (Mean + Std) | 12 × 2 | Tonhöhenverteilung |
| Spectral Centroid (Mean + Std) | 1 × 2 | „Helligkeit" des Klangs |
| Spectral Bandwidth (Mean + Std) | 1 × 2 | Frequenzbreite |
| Spectral Rolloff (Mean + Std) | 1 × 2 | Hochfrequenzanteil |
| ZCR (Mean + Std) | 1 × 2 | Rauigkeit / Noisiness |
| RMS Energy (Mean + Std) | 1 × 2 | Lautstärke-Dynamik |
| Tempo | 1 | Geschätzte BPM |

---

## Modelle

Trainiert und verglichen werden:

- **KNN** – K-Nearest Neighbors (k=5, euclidean)
- **SVM** – Support Vector Machine (RBF-Kernel, Hyperparameter-Tuning via GridSearchCV)
- **Random Forest** – 300 Bäume
- **Gradient Boosting** – 200 Estimators, lr=0.1
- **Ensemble (optimiert)** – Soft Voting: SVM × 3, RF × 2, GB × 1 (KNN entfernt)

---

## Ergebnisse

| Modell | Accuracy |
|---|---|
| KNN | 65,5% |
| **SVM (tuned)** | **74 %** |
| Random Forest | 68,5 % |
| Gradient Boosting | 71,5 % |
| Ensemble (opt)  | 73,5% |

Zum Vergleich: Baseline V1 ~58 %, V2 Best ~93,5 %. (V2 Falsche Ergebnisse wegen Data Leakage)

---

## Voraussetzungen

- Python 3.9+
- Jupyter Notebook / JupyterLab
- [GTZAN Genre Collection Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Abhängigkeiten:

```
numpy
pandas
librosa
scikit-learn
seaborn
matplotlib
```

---

## Installation

```bash
git clone https://github.com/dein-nutzername/musikgenre-klassifikation.git
cd musikgenre-klassifikation

pip install numpy pandas librosa scikit-learn seaborn matplotlib
```

---

## Verwendung

1. GTZAN-Dataset herunterladen und entpacken.
2. Den Pfad in der Notebook-Zelle anpassen:

```python
DATASET_PATH = r"pfad/zum/dataset/genres_original"
```

3. Notebook ausführen:

```bash
jupyter notebook Musikbot_v3.ipynb
```

Alle Zellen der Reihe nach ausführen. Das Notebook führt automatisch durch:
- Feature Extraction aller WAV-Dateien
- Train/Test-Split (80/20, stratifiziert)
- Data Augmentation (×6 auf Trainingsdaten)
- Training aller Modelle inkl. GridSearchCV für SVM
- Ausgabe von Accuracy, Confusion Matrices, Classification Reports und Feature Importance

---

## Projektstruktur

```
musikgenre-klassifikation/
├── Musikbot_v3.ipynb   # Hauptnotebook
└── README.md
```

---

## Versionshistorie

### V3 (aktuell)
- ✅ Train/Test-Split **vor** der Augmentation → kein Data Leakage
- ✅ 6× Augmentation (Pitch +2, Pitch −2, Stretch ×1.1, Stretch ×0.9, Noise) → ~4800 Trainingssamples
- ✅ Optimiertes Ensemble: KNN entfernt, SVM stärker gewichtet (×3)
- ✅ `np.atleast_1d`-Fix für librosa.beat.beat_track Versionskompatibilität

### V2
- Ensemble aus KNN, SVM, RF, GB
- Augmentation, aber noch nach dem Split (Data Leakage)

### V1
- Einfaches KNN-Modell als Baseline (~58 % Accuracy)

---

## Lizenz

Dieses Projekt steht unter der [MIT License](LICENSE).
