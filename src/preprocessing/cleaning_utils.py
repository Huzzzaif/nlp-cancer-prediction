#We will clean the text and preapre if for the model, how can we do thi?
#     1) Convert all to lower case 
#     2) Remove special characters
#     3) Remove numbers
#     4) Remove stopwords

import re
import nltk
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
general_stopwords = set(stopwords.words('english'))

# SciSpaCy is only needed for lemmatize_text(), which the SVM/BioBERT
# pipelines do not call. Load it lazily so the baseline runs without it.
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_sci_sm")
    return _nlp

#certain symbols mean somehting so retain it " - , / , % , .  : ; ( )"
# Retain important medical terms
medical_terms_to_keep = {
    "no", "not", "without", "denies", "denied", "negative", "positive",
    "possible", "likely", "suggests", "suggested", "concern", "risk", "ruled",
    "benign", "malignant", "metastatic", "recurrent",
    "acute", "chronic", "recent", "history",
    "increase", "decrease", "elevated", "normal", "abnormal"
}

custom_stopwords = general_stopwords - medical_terms_to_keep

def clean_text(text):
    text = str(text).lower()
    # Retain alphanumeric characters and specified punctuation
    text = re.sub(r'[^a-z0-9\s\-\./%:;,()]+', '', text) #allow lowercase, allow digits, retain given punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in custom_stopwords]
    return " ".join(tokens)

def lemmatize_text(text):
    doc = _get_nlp()(text)
    return " ".join([token.lemma_ for token in doc])

def main():
    examples = [
        "45-year-old patient with Stage III malignant tumor.",
        "Patient denies mass in liver; no metastatic disease found.",
        "Elevated bilirubin levels suggest concern for obstruction.",
        "History of recurrent malignant glioma detected.",
        "No abnormal lymph nodes seen on recent scan."
    ]

    for i, text in enumerate(examples, 1):
        print(f"\nExample {i}: Original Text")
        print(text)
        
        cleaned = clean_text(text)
        print("\nAfter Cleaning:")
        print(cleaned)
        
        lemmatized = lemmatize_text(cleaned)
        print("\nAfter Lemmatization:")
        print(lemmatized)

if __name__ == "__main__":
    main()