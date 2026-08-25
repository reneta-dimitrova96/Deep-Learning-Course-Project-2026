# Deep Learning Course Project 2026

## Semantic Paraphrase Detection with Deep Learning

This repository contains an educational project created as part of the Deep Learning course at SoftUni.

The project investigates how different machine learning and deep learning approaches perform on the task of semantic paraphrase detection.

## Problem Statement

Semantic paraphrase detection is the task of determining whether two sentences express the same meaning.

This is a challenging NLP problem because two sentences can contain many of the same words while expressing different meanings, or they can express the same meaning using different sentence structures.

Paraphrase detection can be useful in applications such as duplicate question detection, information retrieval, search and text matching.

The goal of this project is to compare three approaches with different levels of model complexity:

- TF-IDF with Logistic Regression as a traditional machine learning baseline
- A Siamese BiLSTM neural network with a shared encoder
- Fine-tuning a pretrained DistilBERT language model

The models are evaluated on the same test data using accuracy, precision, recall and F1-score.

## Dataset

The project uses the **PAWS-Wiki Labeled (Final)** dataset from the PAWS (Paraphrase Adversaries from Word Scrambling) collection.

PAWS-Wiki contains sentence pairs based on Wikipedia text, including challenging paraphrase and non-paraphrase examples with high lexical overlap.

Labels:

- `0` – non-paraphrase
- `1` – paraphrase

Original dataset size:

- **49,401** training examples
- **8,000** validation examples
- **8,000** test examples

### Data Preparation

The same cleaning procedure is applied consistently across the project.

Sentence pairs with conflicting labels are removed, and only one occurrence of the remaining exact duplicate sentence pairs is kept.

After cleaning, the datasets contain:

- **49,334** training examples
- **7,994** validation examples
- **7,977** test examples

The training, validation and test sets are kept separate throughout model development and evaluation.

The PAWS-Wiki dataset was released by Google Research. Google LLC (Google) is acknowledged as the data source.

### Dataset Sources

- [PAWS-Wiki Dataset on Hugging Face](https://huggingface.co/datasets/google-research-datasets/paws)
- [Official PAWS Repository by Google Research](https://github.com/google-research-datasets/paws)
- [PAWS: Paraphrase Adversaries from Word Scrambling](https://arxiv.org/abs/1904.01130)

## Models

### 1. TF-IDF + Logistic Regression

A traditional machine learning baseline is created using TF-IDF representations of the sentence pairs and Logistic Regression.

Different settings are evaluated on the validation set, and a Logistic Regression model with balanced class weights is selected as the final baseline.

### 2. Siamese BiLSTM

A Siamese neural network is built using a shared encoder containing an Embedding layer and a Bidirectional LSTM.

Each sentence is processed through the same encoder, and the resulting representations are compared for binary paraphrase classification.

Several model versions are explored, including changes in model capacity, dropout, learning rate, Optuna hyperparameter tuning and L2 regularization.

The original Siamese BiLSTM configuration achieves the best validation F1-score among the tested versions and is selected for final evaluation.

### 3. Pretrained DistilBERT

The third approach fine-tunes the pretrained **DistilBERT Base Uncased** model for binary sentence-pair classification.

DistilBERT was selected because it provides contextual language representations while being smaller and faster than BERT, making it a practical choice for fine-tuning with limited computational resources.

Learning rate and weight decay experiments are performed using the validation set.

The selected configuration uses:

- learning rate: `2e-5`
- weight decay: `0.01`
- training epochs: `3`
- batch size: `16`

The model is loaded and fine-tuned using Hugging Face Transformers and PyTorch.

## Evaluation

All final models are evaluated on the same cleaned test set containing **7,977 sentence pairs**.

For precision, recall and F1-score, the paraphrase class (`label = 1`) is treated as the positive class.

The main evaluation metrics are:

- **Accuracy** – proportion of all predictions that are correct
- **Precision** – proportion of predicted paraphrases that are actually paraphrases
- **Recall** – proportion of true paraphrases correctly identified by the model
- **F1-score** – harmonic mean of precision and recall

## Final Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.5295 | 0.4718 | 0.5521 | 0.5088 |
| Siamese BiLSTM | 0.7415 | 0.6856 | 0.7654 | 0.7233 |
| DistilBERT | **0.8844** | **0.8453** | **0.9034** | **0.8734** |

The results show a clear improvement from the traditional machine learning baseline to the deep learning approaches.

The Siamese BiLSTM improves considerably over TF-IDF with Logistic Regression, while DistilBERT achieves the strongest performance across all four evaluation metrics.

The final DistilBERT test F1-score is **0.8734**.

## Error Analysis

The final notebook compares the prediction errors produced by the three approaches.

The analysis includes:

- examples correctly classified by DistilBERT but missed by both other models
- examples misclassified by all three models
- DistilBERT false positives and false negatives
- common patterns found in difficult sentence pairs

The results show that high lexical similarity can be challenging for all three approaches when small changes in names, relationships, word order or sentence structure affect the meaning.

DistilBERT correctly classifies many examples that are misclassified by the other two models, although some challenging cases remain. 

## Project Structure

- `notebooks/01_data_preparation_and_eda.ipynb` – data loading, cleaning and exploratory data analysis
- `notebooks/02_traditional_ml_baseline.ipynb` – TF-IDF + Logistic Regression baseline
- `notebooks/03_siamese_neural_network.ipynb` – Siamese BiLSTM experiments
- `notebooks/04_pretrained_language_model_fine_tuning.ipynb` – pretrained DistilBERT fine-tuning, tuning and evaluation
- `notebooks/05_model_comparison_and_error_analysis.ipynb` – final model comparison and error analysis
- `requirements.txt` – Python dependencies
- `.gitignore` – files and generated model artifacts excluded from version control

## Running the Project

The notebooks were developed and executed in Google Colab.

The notebooks follow the project workflow in numerical order:

`01 → 02 → 03 → 04 → 05`

The dataset is downloaded directly from Hugging Face and is not stored in this repository.

Model checkpoints, trained model weights and training output directories are also excluded from the repository.

Notebooks 02–04 generate local prediction files used for the final model comparison. These intermediate files are not committed to the repository, while the saved outputs in Notebook 05 document the final comparison and error analysis.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- Optuna
- Google Colab

## References

### PAWS

Yuan Zhang, Jason Baldridge, and Luheng He.  
**PAWS: Paraphrase Adversaries from Word Scrambling.** 2019.  
[Paper on arXiv](https://arxiv.org/abs/1904.01130)

### DistilBERT

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.  
**DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.** 2019.  
[Paper on arXiv](https://arxiv.org/abs/1910.01108)

### Hugging Face Transformers

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, et al.  
**Transformers: State-of-the-Art Natural Language Processing.** 2020.  
[Paper on ACL Anthology](https://aclanthology.org/2020.emnlp-demos.6/)

### Hugging Face Datasets

Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, et al.  
**Datasets: A Community Library for Natural Language Processing.** 2021.  
[Paper on arXiv](https://arxiv.org/abs/2109.02846)

### Optuna

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama.  
**Optuna: A Next-generation Hyperparameter Optimization Framework.** 2019.  
[Paper on ACM Digital Library](https://doi.org/10.1145/3292500.3330701)

### Software and Model Sources

- [DistilBERT Base Uncased on Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased)

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Transformers GitHub Repository](https://github.com/huggingface/transformers)

- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Hugging Face Datasets GitHub Repository](https://github.com/huggingface/datasets)

- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [Optuna GitHub Repository](https://github.com/optuna/optuna)

