# Deep Learning Course Project 2026

## Semantic Paraphrase Detection with Deep Learning

This repository contains an educational project created as part of the Deep Learning course at SoftUni.

The goal of the project is to compare different approaches for determining whether two sentences have the same meaning.

Three approaches are explored:

- TF-IDF with Logistic Regression as a traditional machine learning baseline
- A Siamese BiLSTM neural network with a shared encoder
- Fine-tuning a pretrained DistilBERT language model

## Dataset

The project uses the **PAWS-Wiki Labeled (Final)** dataset from the PAWS (Paraphrase Adversaries from Word Scrambling) collection.

PAWS-Wiki contains sentence pairs based on Wikipedia text.

Labels:

- `0` – non-paraphrase
- `1` – paraphrase

Original dataset size:

- **49,401** training examples
- **8,000** validation examples
- **8,000** test examples

During data preparation, exact duplicate sentence pairs and pairs with conflicting labels are removed. The cleaned datasets contain:

- **49,334** training examples
- **7,994** validation examples
- **7,977** test examples

The PAWS-Wiki dataset was released by Google Research. **Google LLC (Google) is acknowledged as the data source.**

### Dataset Sources

- [PAWS-Wiki Dataset on Hugging Face](https://huggingface.co/datasets/google-research-datasets/paws)
- [Official PAWS Repository by Google Research](https://github.com/google-research-datasets/paws)
- [PAWS Paper](https://arxiv.org/abs/1904.01130)

## Models

### TF-IDF + Logistic Regression

A traditional machine learning baseline is created using TF-IDF text features and Logistic Regression.

A balanced version of Logistic Regression is selected based on the validation results.

### Siamese BiLSTM

A Siamese neural network is built using a shared encoder consisting of an Embedding layer and a Bidirectional LSTM.

The two sentences are processed separately through the same encoder and their learned representations are compared for paraphrase classification.

Several versions of the model are explored, including different model sizes, dropout settings, hyperparameter tuning and L2 regularization.

### Pretrained Language Model

The final approach fine-tunes the pretrained **DistilBERT Base Uncased** model for binary sentence-pair classification.

DistilBERT was selected because it provides contextual representations while being smaller and faster than BERT, making it suitable for fine-tuning with limited computational resources.

The model is loaded and fine-tuned using the **Hugging Face Transformers** library and PyTorch.

## Model Comparison

The three approaches are evaluated on the same cleaned test set using accuracy, precision, recall and F1-score.

The traditional TF-IDF baseline provides a reference point for the project. The Siamese BiLSTM improves the classification performance considerably, while the pretrained DistilBERT model achieves the strongest overall results.

A separate model comparison and error analysis notebook examines the differences between the approaches in more detail.

## Project Structure

- `notebooks/01_data_preparation_and_eda.ipynb` – data loading, cleaning and exploratory data analysis
- `notebooks/02_traditional_ml_baseline.ipynb` – TF-IDF and Logistic Regression baseline
- `notebooks/03_siamese_neural_network.ipynb` – Siamese BiLSTM experiments
- `notebooks/04_pretrained_language_model_fine_tuning.ipynb` – pretrained DistilBERT fine-tuning and evaluation
- `notebooks/05_model_comparison_and_error_analysis.ipynb` – model comparison and error analysis

## Tools and Libraries

Python, Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras, PyTorch, Hugging Face Transformers, Hugging Face Datasets, Accelerate, Optuna and Google Colab.

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

### Software and Model Sources

- [DistilBERT Base Uncased on Hugging Face](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Transformers Paper](https://aclanthology.org/2020.emnlp-demos.6/)
