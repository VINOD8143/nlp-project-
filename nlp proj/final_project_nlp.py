# -*- coding: utf-8 -*-
"""final_project_nlp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NIwYLZgmIsRvYy_QpM-NgaC65plsZmAZ
"""

pip install pandas sumy transformers torch datasets

!pip install nltk scikit-learn
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # Download the 'punkt_tab' resource

import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extractive_summary_tfidf(text, num_sentences=3):
    """
    Extractive summarization using TF-IDF and cosine similarity.
    """
    # Sentence tokenization
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # Not enough sentences to summarize

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Sentence scores: sum of similarities
    scores = sim_matrix.sum(axis=1)

    # Top N sentence indices
    ranked_indices = np.argsort(scores)[-num_sentences:]
    ranked_indices.sort()  # To maintain original order in text

    # Build summary
    summary = ' '.join([sentences[i] for i in ranked_indices])
    return summary

# Load dataset
df = pd.read_csv('/content/dataset_nlp new.csv', encoding='latin-1')

# Apply Extractive Summarization
df["extractive_summary"] = df["abstract"].apply(lambda x: extractive_summary_tfidf(str(x)))

# Save Extractive Summaries
os.makedirs("/mnt/data", exist_ok=True)
df.to_csv("/mnt/data/extractive_summaries.csv", index=False)

print("✅ Extractive summaries (TF-IDF + Cosine Similarity) saved as extractive_summaries.csv")

from google.colab import files
files.download("/mnt/data/extractive_summaries.csv")

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import pandas as pd

# Load T5 Model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def abstractive_summary_batch(texts, max_length=100, batch_size=8):
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(["summarize: " + text for text in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        summary_ids = model.generate(inputs.input_ids, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summaries.extend(tokenizer.batch_decode(summary_ids, skip_special_tokens=True))
    return summaries

df = pd.read_csv('/content/dataset_nlp new.csv', encoding='latin-1')
# Apply Abstractive Summarization in Batches
df["abstractive_summary"] = abstractive_summary_batch(df["abstract"].astype(str).tolist())
os.makedirs("/mnt/data", exist_ok=True)

# Save Faster Abstractive Summaries
df.to_csv("/mnt/data/abstractive_summaries.csv", index=False)
print("✅ Faster abstractive summaries saved as abstractive_summaries.csv")
print(os.path.abspath("/mnt/data/abstractive_summaries.csv"))

from google.colab import files
files.download("/mnt/data/abstractive_summaries.csv")

!pip install datasets

from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load and clean dataset
df = pd.read_csv('/content/dataset_nlp new.csv', encoding='latin-1')
df = df.dropna(subset=["abstract", "title"])
df["abstract"] = df["abstract"].fillna("")
df["title"] = df["title"].fillna("")

# Rename and convert to dataset
df = df[["abstract", "title"]].rename(columns={"abstract": "text", "title": "summary"})
dataset = Dataset.from_pandas(df.reset_index(drop=True))

# Preprocessing function
def preprocess_function(examples):
    inputs = ["summarize: " + str(text) for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")

    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Training args (no evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./t5_finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train and evaluate manually
trainer.train()
trainer.evaluate()

print("✅ Model fine-tuned successfully!")

import os

model_path_fine_tuned = "./t5_finetuned"
if os.path.exists(model_path_fine_tuned):
    print("✅ Fine-tuned model directory exists.")
    print("📂 Files inside:", os.listdir(model_path_fine_tuned))
else:
    print("❌ Model directory not found. Fine-tuning might have failed.")

pip install rouge-score pandas torch transformers

import pandas as pd
from rouge_score import rouge_scorer

# Load the summarization datasets
extractive_df = pd.read_csv("/mnt/data/extractive_summaries.csv")
abstractive_df = pd.read_csv("/mnt/data/abstractive_summaries.csv")

# Define ROUGE Scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_summaries(df, summary_col, reference_col="abstract"):
    """
    Computes ROUGE Precision, Recall, and F1-score for the summaries.
    """
    results = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ref, summary in zip(df[reference_col], df[summary_col]):
        if pd.isna(ref) or pd.isna(summary):  # Skip missing values
            continue
        scores = scorer.score(ref, summary)

        for rouge_type in results.keys():
            results[rouge_type].append({
                "precision": scores[rouge_type].precision,
                "recall": scores[rouge_type].recall,
                "f1": scores[rouge_type].fmeasure
            })

    # Compute average scores
    avg_scores = {}
    for rouge_type in results.keys():
        avg_scores[rouge_type] = {
            "precision": sum(d["precision"] for d in results[rouge_type]) / len(results[rouge_type]),
            "recall": sum(d["recall"] for d in results[rouge_type]) / len(results[rouge_type]),
            "f1": sum(d["f1"] for d in results[rouge_type]) / len(results[rouge_type])
        }

    return avg_scores

# Evaluate Extractive Summarization
extractive_scores = evaluate_summaries(extractive_df, "extractive_summary")

# Evaluate Abstractive Summarization
abstractive_scores = evaluate_summaries(abstractive_df, "abstractive_summary")

# Print Scores
print("\n🔹 **Extractive Summarization Scores:**")
print(pd.DataFrame(extractive_scores))

print("\n🔹 **Abstractive Summarization Scores:**")
print(pd.DataFrame(abstractive_scores))