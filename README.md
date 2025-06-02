# FairSpeech: Gender-Balanced Speech Emotion Recognition

FairSpeech is a deep learning framework for **Speech Emotion Recognition (SER)** that mitigates gender bias through adversarial learning and transformer-based architectures. It supports 5-fold-cross-validation and offers modular components for extensibility.

## 🗂 Project Structure

```
FairSpeech-Code/
├── analysis.ipynb                      # EDA and performance analysis
├── data/RAVDESS_audio                 # Dataset utilities
│   ├── download_RAVDESS_audio.py
│   └── reclassify.py
├── env/                               # Environment specifications
│   ├── conda-spec.txt
│   └── requirements.txt
├── logs/                              # Saved models and metrics
│   └── pointwise_conv_baseline*/...
├── src/                               # Main source code
│   ├── config.yaml                    # Global configuration
│   ├── train.py                       # Train single model
│   ├── train_5_fold.py                # Train with 5-fold CV
│   ├── evaluation.py                  # Evaluate single model
│   ├── evaluation_5_fold.py           # Evaluation for multiple folds
│   ├── data/
│   │   └── RAVDESS_dataset.py         # PyTorch dataset for RAVDESS
│   └── models/
│       ├── feature_extractor.py      # wav2vec2.0 Feature extractor
│       ├── transformer_encoder.py    # Transformer-based encoder
│       ├── pointwise_conv_classifier.py  # Pointwise Conv1D baseline
│       ├── heads.py                  # Classification heads
│       └── grl.py                    # Gradient Reversal Layer for adversarial learning
```

## 🚀 How to Run

### 1. Setup Environment

```bash
conda env create -f environment.yml -n fairspeech
conda activate fairspeech
```

### 2. Download and Preprocess Data

```bash
python data/RAVDESS_audio/download_RAVDESS_audio.py
python data/RAVDESS_audio/reclassify.py
```

### 3. Train Model (5-fold cross-validation)

```bash
python src/train_5_fold.py --config src/config.yaml
```

Or train a single model:

```bash
python src/train.py --config src/config.yaml
```

### 4. Evaluate Model

```bash
python src/evaluation_5_fold.py --checkpoint_dir logs/[folder_name]
```

## 📊 Features

* Support for Pointwise Conv1D and Transformer-based classifiers
* 5-fold cross-validation
* Adversarial debiasing using Gradient Reversal Layer
* Gender-aware performance tracking

## 📄 License

This project is for academic use only.

---
