# Multilabel Sentiment and Emotion Analysis for Bangla YouTube Comments

## 📌 Project Overview

This repository contains an end-to-end **sentiment and emotion analysis system** for **Bangla YouTube comments**, implemented and evaluated using both **classical machine learning** and **deep learning** models.

The project focuses on **real-world, noisy, and code-mixed text**, including:

* Bangla (BN)
* English (EN)
* Romanized Bangla (RN)

It supports:

* **Sentiment classification**

  * 3-label: Negative, Neutral, Positive
  * 5-label: Highly Negative, Negative, Neutral, Positive, Highly Positive
* **Emotion detection**

  * Anger, Joy, Disgust, Fear, Surprise, Sad, None

All experiments, preprocessing, and evaluations are fully reproducible using the provided datasets and notebooks.

---

## 📁 Repository Structure

```
├── Sentiment.csv                 # Multilabel sentiment dataset
├── Emotion.csv                   # Emotion detection dataset
├── sentiment_analysis.ipynb      # Sentiment modeling notebook
├── emotion_analysis.ipynb        # Emotion modeling notebook
├── Sentiment_and_Emotion_Analysis_Report.pdf
└── README.md
```

---

## 📊 Dataset Description

The datasets are derived from Bangla YouTube comments collected between **2013 and early 2018**, originally introduced by Tripto & Ali (2018).

### Key Characteristics

* Comments collected from popular Bangla YouTube videos
* Top-level comments only (replies excluded)
* Contains slang, sarcasm, abusive language, and code-mixing
* Annotated manually by adult annotators

### Files

* **Sentiment.csv**

  * Supports both 3-label and 5-label sentiment schemes
* **Emotion.csv**

  * Supports 7 discrete emotion classes

Each file includes raw text, language tag (BN/EN/RN), and domain metadata .

---

## 🔧 Text Preprocessing

A shared, language-aware preprocessing pipeline is applied across both tasks:

* Lowercasing and normalization
* Removal of URLs, mentions, hashtags
* Language-specific stopword removal:

  * English (NLTK)
  * Bangla
  * Romanized Bangla (custom list)
* English-only stemming (Porter Stemmer)
* Token length filtering
* Dropping empty or invalid rows

This pipeline ensures robustness against noisy, real-world social media text.

---

## 🤖 Models Implemented

### Classical Machine Learning

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM
* Passive Aggressive Classifier

### Deep Learning

* Simple RNN
* LSTM / Multi-layer LSTM
* Bidirectional LSTM (BiLSTM)

### Feature Representation

* **TF–IDF** for classical models
* **Tokenized & padded sequences** for deep learning models

---

## 📈 Evaluation Strategy

* Stratified **80/20 train–test split**
* **5-fold cross-validation**
* Metrics reported:

  * Accuracy
  * Precision
  * Recall
  * F1-score

For 5-label sentiment classification, **class weighting** is applied to mitigate class imbalance, especially for extreme sentiment categories.

---

## 🏆 Results Summary

### 3-Label Sentiment Classification (Test Accuracy)

| Model               | Accuracy   |
| ------------------- | ---------- |
| **BiLSTM**          | **0.6340** |
| Logistic Regression | 0.6274     |
| Naive Bayes         | 0.6251     |
| SVM                 | 0.6207     |

### 5-Label Sentiment Classification (Test Accuracy)

| Model               | Accuracy   |
| ------------------- | ---------- |
| **BiLSTM**          | **0.5000** |
| Logistic Regression | 0.4644     |
| Naive Bayes         | 0.4548     |
| SVM                 | 0.4521     |

### Emotion Detection (Test Accuracy)

| Model               | Accuracy   |
| ------------------- | ---------- |
| **BiLSTM**          | **0.4758** |
| Logistic Regression | 0.4610     |
| Naive Bayes         | 0.4498     |
| SVM                 | 0.4387     |

**Key takeaway:**
BiLSTM consistently achieves the best generalization across all tasks, while classical models remain strong, efficient baselines .

---

## ⚙️ How to Run

1. Clone the repository

```bash
git clone https://github.com/Anu213007/Sentiment-Emotion-Analysis
cd Sentiment-Emotion-Analysis
```

2. Open the notebooks

```bash
jupyter notebook
```

3. Run:

* `sentiment_analysis.ipynb` for sentiment classification
* `emotion_analysis.ipynb` for emotion detection

Ensure required Python libraries (e.g., `scikit-learn`, `nltk`, `tensorflow`, `numpy`, `pandas`) are installed.

---

## 🚧 Limitations

* Romanized Bangla text remains highly inconsistent
* Rare sentiment and emotion classes are difficult to model
* Deep learning performance is limited without pre-trained multilingual embeddings

---

## 🔮 Future Improvements

* Integrate transformer-based models (mBERT, XLM-R)
* Improve Romanized Bangla normalization
* Add macro-averaged and per-class F1 metrics
* Extend to deployment (API or web interface)

---

## 📄 Report

A detailed system description, experimental setup, and analysis are available in:

**`Sentiment_and_Emotion_Analysis_Report.pdf`**

---

## 📌 Acknowledgements

Dataset and task formulation are based on:

> Tripto, N. I., & Ali, M. E. (2018). *Detecting Multilabel Sentiment and Emotions from Bangla YouTube Comments.*
