# Sentiment-Analysis-Minibatching

This repository contains the implementation for **Assignment 1 of NLP 202**, focusing on **minibatching** in PyTorch for **sentiment analysis** on the IMDB dataset. It features two models: a **Logistic Regression (LR)** baseline and a more expressive **LSTM classifier**, both trained and evaluated across various batch sizes and learning rates.

## Project Overview

### Models:
- `LR.py`: Logistic Regression with embedding and mean-pooling
- `LSTM.py`: LSTM with dropout and final hidden state classification

### Core Concepts:
- Sentiment analysis on IMDB reviews (binary classification)
- Efficient training with **mini-batches** of size {1, 8, 16, 32, 64}
- Learning rate tuning across `{1e-4, 5e-4, 1e-3, 5e-3, 1e-2}`
- Performance visualization (accuracy vs. batch size / learning rate)
- Saving predictions and evaluation logs

## Environment Setup

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/Sentiment-Analysis-Minibatching.git
cd Sentiment-Analysis-Minibatching
```

### 2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Sample `requirements.txt`
```txt
torch
spacy
tqdm
scikit-learn
matplotlib
pandas
numpy
```

> Note: The scripts automatically download the IMDB dataset and `spaCy` language model (`en_core_web_sm`) if not already present.

## How to Run

### Train Logistic Regression
```bash
python LR.py
```

### Train LSTM
```bash
python LSTM.py
```

Both scripts:
- Shuffle and split IMDB data
- Perform batch-based training + validation
- Save prediction results to `.csv` files
- Output training time, validation/test accuracy, and learning rate plots

## Results Summary

| Model | Best Batch Size | Best Learning Rate | Val Accuracy | Test Accuracy | Training Time |
|-------|------------------|--------------------|--------------|----------------|----------------|
| LR    | 16               | 0.001              | 90.40%       | 88.86%         | 49.19 sec      |
| LSTM  | 16               | 0.001              | 90.10%       | 87.82%         | 3116.40 sec    |

> Logistic Regression had faster training and generalization, while LSTM demonstrated robust sequential modeling capabilities.

## File Structure

```
Sentiment-Analysis-Minibatching/
├── LR.py                        # Logistic Regression model training
├── LSTM.py                      # LSTM model training
├── requirements.txt             # Dependencies
├── dev_predictions_LR.csv       # Sample saved predictions (auto-generated)
├── test_predictions_LR.csv      # Sample saved predictions (auto-generated)
├── *.png                        # Accuracy and training time plots (auto-generated)
└── README.md                    # You're here
```
