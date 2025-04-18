# Academic Paper Summarization using NLP

This project applies **Extractive and Abstractive Summarization** techniques on academic paper abstracts by Indian authors. It combines TF-IDF, HuggingFace Transformers, and pre-trained models from `sumy` and `transformers` to generate summaries.

## Requirements

Install the following dependencies:

```bash
pip install pandas sumy transformers torch datasets nltk scikit-learn spacy
```

And run these downloads for NLTK and SpaCy:

```python
import nltk
nltk.download('punkt')
!python -m spacy download en_core_web_sm
```

## File Structure

- `final_project.ipynb`: Main notebook with all summarization models implemented.
- `data/`: Folder where the CSV file with abstracts should be placed (create manually if needed).
- `output/`: Summarization outputs will be stored here (optional).

## Input Format

The notebook expects a **CSV file** with a column named `abstract`. Each row contains an academic abstract.

| title               | abstract                                        |
|---------------------|------------------------------------------------|
| ML in Healthcare    | Machine learning has transformed healthcare...|
| NLP Applications    | Natural language processing has evolved...    |

## How to Run

1. **Download or clone the project**, and place your data file in the appropriate location.
2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook final_project.ipynb
   ```
3. **Run each cell sequentially** using `Shift + Enter`.

## What It Does

### 1. Extractive Summarization using TF-IDF & Cosine Similarity
- Tokenizes the text into sentences.
- Uses `TfidfVectorizer` and `cosine_similarity` to score and select the most relevant sentences.

### 2. Summarization using `sumy` Library
- Implements **LexRank**, **Luhn**, and **LSA** summarizers via `sumy`.

### 3. Abstractive Summarization using HuggingFace Transformers
- Utilizes the `facebook/bart-large-cnn` model from HuggingFace to generate fluent, human-like summaries.

## Sample Output

Example output from BART:

```
Original: Machine learning models have shown promise in predicting patient outcomes...
Summary: ML models are effective in forecasting outcomes in medical research.
```

## Notes

- SpaCy is used for sentence preprocessing.
- Summary length and model can be configured in the code.
- Large models like BART require GPU for fast inference.
