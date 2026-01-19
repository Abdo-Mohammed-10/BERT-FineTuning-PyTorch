# BERT Fine-Tuning for News Classification 📰🚀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UZ3W0CfN7UYK4pWfCpvLeJ4bFG57dClr?usp=sharing)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)

> **Fine-tuning a pre-trained BERT model to classify news articles into 4 categories (World, Sports, Business, Sci/Tech) using the AG News dataset.**

---

## 📌 Overview

This project demonstrates how to leverage **Transfer Learning** effectively. Instead of training a model from scratch, we fine-tune a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model to achieve state-of-the-art results on text classification tasks.

Beyond simple classification, this project visualizes how the model "learns" to separate topics in the high-dimensional vector space using **t-SNE**.

### Key Features
* **Transfer Learning:** Fine-tuning `bert-base-uncased` for a downstream task.
* **PyTorch Implementation:** Custom training loop with evaluation and checkpointing.
* **Latent Space Visualization:** Using **t-SNE** to project BERT embeddings into 2D space, visualizing cluster separation between different news topics.
* **Drive Integration:** Automated saving of the best model to Google Drive.

---

## 📊 Dataset: AG News

The dataset consists of news articles from the AG's corpus of news pages.
* **Classes:** 4 (World, Sports, Business, Sci/Tech).
* **Input:** Title + Description of the news article.

---

## 🛠️ Tech Stack

* **Core Framework:** `PyTorch`
* **Model:** `Hugging Face Transformers` (BERT)
* **Dimensionality Reduction:** `t-SNE` (from `sklearn`)
* **Data Handling:** `Pandas`, `NumPy`

---

## 📈 Results & Visualization

*(Place your t-SNE plot here)*

### Latent Space Analysis (t-SNE)
The visualization demonstrates that after fine-tuning, BERT successfully maps semantically similar news articles closer together in the embedding space, creating distinct clusters for each category (Sports vs. Politics, etc.).

---

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/BERT-FineTuning-PyTorch.git](https://github.com/YourUsername/BERT-FineTuning-PyTorch.git)
    cd BERT-FineTuning-PyTorch
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch transformers pandas scikit-learn matplotlib
    ```

3.  **Run the Notebook:**
    Open `Bert_Topic_Classification.ipynb` in Jupyter or Google Colab to start training.

---

## 🤝 Future Improvements

* [ ] Experiment with other transformer architectures (RoBERTa, DistilBERT).
* [ ] Implement a Gradio/Streamlit app for real-time inference.
* [ ] Add Mixed Precision Training (AMP) for faster training.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
