# Deep Learning Course Project 2026

## Semantic Paraphrase Detection with Deep Learning

This repository contains an educational project created as part of the Deep Learning course at SoftUni.

The goal is to compare different approaches for determining whether two sentences have the same meaning.

## Dataset

The project uses the **PAWS-Wiki Labeled (Final)** dataset from the PAWS (Paraphrase Adversaries from Word Scrambling) collection.

PAWS-Wiki contains sentence pairs based on Wikipedia text.

Labels:

* `0` – non-paraphrase
* `1` – paraphrase

Original dataset size:

* **49,401** training examples
* **8,000** validation examples
* **8,000** test examples

The **PAWS-Wiki dataset** was released by Google Research and is used in this project for educational purposes. **Google LLC (Google) is acknowledged as the data source.**

### Sources

* [PAWS-Wiki Dataset on Hugging Face](https://huggingface.co/datasets/google-research-datasets/paws)
* [Official PAWS Repository by Google Research](https://github.com/google-research-datasets/paws)

## Project Structure

* `01_Data_and_EDA.ipynb` – data loading, cleaning and exploratory data analysis
* `02_TFIDF_Logistic_Regression.ipynb` – TF-IDF + Logistic Regression baseline
* `03_Siamese_BiLSTM.ipynb` – Siamese BiLSTM model

## Tools

Python, Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras, Hugging Face Datasets and Google Colab.

## Reference

Yuan Zhang, Jason Baldridge, and Luheng He.
**PAWS: Paraphrase Adversaries from Word Scrambling.** 2019.

[Paper on arXiv](https://arxiv.org/abs/1904.01130)
