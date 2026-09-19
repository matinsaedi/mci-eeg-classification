# MCI Classification from Raw EEG Using 1D CNNs

Deep learning-based classification of **Mild Cognitive Impairment (MCI)** from raw resting-state EEG using one-dimensional convolutional neural networks.

This repository contains the code and supporting material for my **2022 bachelor's thesis in Electrical Engineering at K. N. Toosi University of Technology**.

The project investigates whether MCI can be distinguished from healthy aging **directly from raw EEG**, without handcrafted feature extraction, signal denoising, or time-frequency representations. It also studies how EEG window length affects classification performance and how much discriminative information is retained by individual electrodes.

> **Research use only:** this project is an academic classification study, not a clinically validated diagnostic system.

## Project Highlights

* Built an end-to-end 1D CNN pipeline operating directly on **raw 19-channel EEG**.
* Evaluated four temporal resolutions: **2.5, 5, 7.5, and 10 seconds**.
* Used **subject-wise leave-one-subject-out cross-validation (LOOCV)** across 60 participants.
* Trained **19 separate single-channel models** to study the discriminative contribution of individual EEG electrodes.
* Historically reported a best all-channel accuracy of **95.58%** using 7.5-second windows.
* Historically reported **85.97% accuracy using only the C3 electrode** in the single-channel study.

```mermaid
flowchart LR
    A[30-minute raw EEG] --> B[Non-overlapping windows]
    B --> C[1D CNN]
    C --> D[Window-level predictions]
    D --> E[Held-out subject evaluation]
```

## Dataset

The project uses EEG recordings from **60 participants**:

| Group                     | Participants |
| ------------------------- | -----------: |
| Mild Cognitive Impairment |           29 |
| Healthy Control           |           31 |
| **Total**                 |       **60** |

Recording characteristics:

* Resting-state, eyes-closed EEG
* Participants older than 55
* 30 minutes per participant
* Sampling frequency: **256 Hz**
* **19 scalp electrodes** placed according to the international 10-20 system

The electrodes are:

`Fp1`, `Fp2`, `F7`, `F3`, `Fz`, `F4`, `F8`, `T3`, `C3`, `Cz`, `C4`, `T4`, `T5`, `P3`, `Pz`, `P4`, `T6`, `O1`, `O2`

The dataset originates from:

> M. Kashefpoor, H. Rabbani, and M. Barekatain, “Supervised dictionary learning of EEG signals for mild cognitive impairment diagnosis,” *Biomedical Signal Processing and Control*, vol. 53, 101559, 2019.
> https://doi.org/10.1016/j.bspc.2019.101559

The EEG recordings are **not included in this repository**. See [`data/README.md`](data/README.md) for the expected archive names, extracted files, and array shapes.

## Method

### 1. Raw EEG Segmentation

No handcrafted features, frequency transforms, time-frequency representations, or explicit denoising stages are used in the classification pipeline.

Each 30-minute recording is divided into non-overlapping windows of:

* 2.5 seconds
* 5 seconds
* 7.5 seconds
* 10 seconds

A separate CNN architecture was evaluated for each temporal resolution.

### 2. 1D CNN Classification

Each convolutional block follows:

```text
Conv1D -> ReLU -> Dropout(0.2) -> BatchNormalization -> MaxPool1D
```

The convolutional stack is followed by:

```text
Flatten -> Dense(100, ReLU) -> Dense(1, Sigmoid)
```

The all-channel models begin with layer normalization. The original single-channel architecture does not.

| Window      | Convolution Kernels | Filters            |
| ----------- | ------------------- | ------------------ |
| 2.5 seconds | 7, 13, 19           | 16, 32, 64         |
| 5 seconds   | 9, 13, 17, 21, 23   | 16, 32, 64, 32, 32 |
| 7.5 seconds | 11, 15, 19, 23, 27  | 16, 32, 64, 32, 32 |
| 10 seconds  | 11, 15, 19, 23, 27  | 16, 32, 64, 32, 32 |

The notebook's default active experiment is the 2.5-second all-channel configuration. The other original architectures remain in the architecture cell for manual selection.

### 3. Subject-wise Evaluation

Performance is evaluated using **leave-one-subject-out cross-validation**.

For each of the 60 folds:

1. All EEG windows belonging to one participant are held out for testing.
2. The model is trained using data from the remaining 59 participants.
3. The held-out participant is evaluated only after training.
4. The process is repeated until every participant has served as the test subject.

This keeps the test subject independent from the training set instead of randomly mixing windows from the same participant across train and test data.

### 4. Single-Channel Analysis

To investigate whether individual electrodes retain useful information for MCI classification, the 10-second architecture was trained separately for each of the **19 EEG channels**.

The 10-second configuration was selected for this experiment because it produced the highest sensitivity in the all-channel study.

## Historical Results

The results below come from the original project experiments and a later unpublished manuscript draft based on the thesis. They are provided as **historical research results**, not as newly reproduced benchmarks from the cleaned repository.

### Multi-channel Classification

| Window    |          Accuracy |  Precision | Sensitivity | Specificity |
| --------- | ----------------: | ---------: | ----------: | ----------: |
| 2.5 s     |     92.48 ± 8.73% |     92.01% |      92.46% |      92.49% |
| 5 s       |     91.80 ± 9.80% |     89.22% |      94.45% |      89.32% |
| **7.5 s** | **95.58 ± 8.68%** | **92.59%** |      98.76% |  **92.61%** |
| 10 s      |     95.24 ± 8.44% |     91.72% |  **99.10%** |      91.63% |

The **7.5-second model** achieved the highest reported accuracy, precision, and specificity, while the **10-second model** achieved the highest sensitivity.

### Single-Channel Classification

The strongest reported individual electrodes were:

| Channel |           Accuracy |
| ------- | -----------------: |
| **C3**  | **85.97 ± 26.71%** |
| **T3**  | **84.24 ± 26.19%** |
| **T5**  | **81.87 ± 24.87%** |
| Cz      |     81.80 ± 28.38% |
| Fz      |     81.48 ± 28.87% |

Across all 19 channels, reported accuracy ranged from **61.77% for T4** to **85.97% for C3**.

These single-channel results are exploratory. They suggest that some electrodes may retain useful discriminative information, but they should not be interpreted as evidence of clinically validated EEG biomarkers.

## Repository Structure

```text
.
├── MCI_EEG_Conv1D.ipynb     # Preprocessing, models, training, and evaluation
├── data/
│   └── README.md             # Expected dataset organization
├── docs/
│   ├── thesis-fa.pdf         # Bachelor's thesis (Persian)
│   └── presentation-fa.pdf   # Thesis presentation (Persian)
├── requirements.txt
└── README.md
```

Datasets, trained models, editable manuscript drafts, and clinical source files are intentionally excluded.

## Running the Notebook

The project was developed in Google Colab.

1. Place the two dataset archives in `My Drive/Bachelor's Project/` using the exact filenames documented in [`data/README.md`](data/README.md).
2. Open [`MCI_EEG_Conv1D.ipynb`](MCI_EEG_Conv1D.ipynb) in Google Colab.
3. Enable a GPU runtime if available.
4. Select the desired channel configuration and EEG window length near the beginning of the notebook:

```python
c = "All"
t_batch = 2.5
```

5. Make sure the corresponding architecture and model output directory match the selected experiment.
6. Run the notebook from top to bottom.

A complete LOOCV experiment requires training **60 separate models**, so execution time depends on the selected architecture and hardware.

## Reproducibility and Evaluation Notes

This repository contains a cleaned version of code originally developed for the 2022 thesis.

The historical results above **have not been fully reproduced after the repository cleanup**. The current notebook derives early-stopping validation data from the training portion of each fold rather than from the held-out test participant. This removes test-subject leakage during model selection, so a complete rerun may produce results different from the historical thesis/manuscript values.

Repository preparation also included:

* Notebook structure and syntax validation
* Preprocessing checks against the original EEG arrays
* Verification of subject counts, labels, and generated window shapes
* Preservation of the original CNN architectures
* A current TensorFlow/Keras runtime smoke test covering model construction, training, prediction, evaluation, and `.keras` save/reload consistency

The complete 60-fold experiment has not yet been rerun under the cleaned evaluation pipeline.

## Supporting Documents

* [`docs/thesis-fa.pdf`](docs/thesis-fa.pdf) — bachelor's thesis in Persian
* [`docs/presentation-fa.pdf`](docs/presentation-fa.pdf) — thesis presentation in Persian
* [`MCI_EEG_Conv1D.ipynb`](MCI_EEG_Conv1D.ipynb) — implementation and experiments

## Tech Stack

**Python · TensorFlow/Keras · NumPy · scikit-learn · Jupyter / Google Colab**

## Responsible Use

This repository is intended for **research and educational purposes only**.

The models have not undergone clinical validation and should not be used as medical diagnostic systems. Clinical deployment would require independent validation on larger and more diverse populations, prospective evaluation, and appropriate medical and regulatory review.
