# Resume Classification Using Machine Learning

## Overview 🔍
This repository contains a machine learning pipeline designed to automate the classification of resumes into specific job categories. By leveraging Natural Language Processing (NLP) and machine learning models, the project provides an efficient solution to streamline recruitment processes, replacing manual classification with automated predictions.

## Features ✨

### Text Preprocessing
- Lowercasing text data.
- Removing punctuation and special characters.
- Tokenization.
- Stopword removal.
- Lemmatization.

### Feature Extraction
- Count Vectorization.
- TF-IDF Vectorization.
- Word Embeddings (Word2Vec).

### Machine Learning Models
- Logistic Regression with L1 Regularization.
- Decision Trees, Random Forests.
- Support Vector Machines (SVM).
- Neural Networks (Feedforward, RNN, LSTM).

### Evaluation Metrics
- Accuracy.
- Confusion Matrices.
- Regularization techniques to reduce overfitting.

## Dataset 📂
The dataset used for this project is sourced from Kaggle: **Resume Dataset**. It contains over 2,400 resumes across more than 24 job categories. Each resume includes textual data and associated metadata.

## Results 📊

### Best Performing Model 🚀
- **75% Test Accuracy** using Logistic Regression with L1 regularization and TF-IDF Vectorization.

### Observations
- TF-IDF provided the most effective feature extraction method for imbalanced datasets.
- L1 regularization helped reduce overfitting and improve generalization.

## Getting Started

### Prerequisites ⚙️
- **Python** 3.6+

#### Libraries 🔧
- `nltk`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- `wordcloud`, `sklearn`, `tensorflow`, `gensim`

### Usage
Run the Jupyter Notebook to preprocess the data, train models, and evaluate results.

## File Structure
- **code.ipynb**: Main Jupyter Notebook with all steps.
- **archive/data/**: Directory to store datasets.
