from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
# Extract TF-IDF word-level features (1-2 grams)
# Important for capturing important medical terms and short phrases.
def extract_tfidf_word_ngrams(texts, max_features):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer='word',
        max_features=max_features,
        lowercase=False,
        token_pattern=r'\b\w+\b'
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

# Extract TF-IDF character-level features (3-5 characters)
# Useful for capturing subword patterns, spelling variations in medical terminology.
def extract_tfidf_char_ngrams(texts, max_features):
    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer='char',
        max_features=max_features,
        lowercase=False
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

#- Test Example Texts
texts = [
    "Patient has glioblastoma multiforme.",
    "No evidence of melanoma found.",
    "Colon carcinoma resection completed."
]

# --- Test Word n-grams
word_tfidf, word_vectorizer = extract_tfidf_word_ngrams(texts, max_features=20)
print("\nWord-level TF-IDF feature names:")
print(word_vectorizer.get_feature_names_out())
word_df = pd.DataFrame(word_tfidf.toarray(), columns=word_vectorizer.get_feature_names_out())
print("\nWord-level TF-IDF matrix:")
print(word_df)

# --- Test Character n-grams
char_tfidf, char_vectorizer = extract_tfidf_char_ngrams(texts, max_features=20)
print("\nCharacter-level TF-IDF feature names:")
print(char_vectorizer.get_feature_names_out())
char_df = pd.DataFrame(char_tfidf.toarray(), columns=char_vectorizer.get_feature_names_out())
print("\nCharacter-level TF-IDF matrix:")
print(char_df)
