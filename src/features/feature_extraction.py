"""TF-IDF feature extraction.

Vectorizers are built unfitted and must be fit on the TRAINING split only,
then used to transform the test split. Fitting on the full corpus leaks test
vocabulary, document frequencies, and max_features selection into training.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_word_vectorizer(max_features):
    """Word-level TF-IDF (1-2 grams): medical terms and short phrases."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer='word',
        max_features=max_features,
        lowercase=False,
        token_pattern=r'\b\w+\b'
    )


def build_tfidf_char_vectorizer(max_features):
    """Character-level TF-IDF (3-5 grams): subword patterns, spelling variants."""
    return TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer='char',
        max_features=max_features,
        lowercase=False
    )


def fit_transform_train_test(train_texts, test_texts, max_features=4000):
    """Fit both vectorizers on train_texts only, transform both splits.

    Returns (X_train, X_test, word_vectorizer, char_vectorizer).
    """
    from scipy.sparse import hstack

    word_vec = build_tfidf_word_vectorizer(max_features)
    char_vec = build_tfidf_char_vectorizer(max_features)

    train_word = word_vec.fit_transform(train_texts)
    train_char = char_vec.fit_transform(train_texts)

    test_word = word_vec.transform(test_texts)
    test_char = char_vec.transform(test_texts)

    return (
        hstack([train_word, train_char]).tocsr(),
        hstack([test_word, test_char]).tocsr(),
        word_vec,
        char_vec,
    )


if __name__ == "__main__":
    import pandas as pd

    texts = [
        "Patient has glioblastoma multiforme.",
        "No evidence of melanoma found.",
        "Colon carcinoma resection completed."
    ]

    word_vec = build_tfidf_word_vectorizer(max_features=20)
    word_tfidf = word_vec.fit_transform(texts)
    print("\nWord-level TF-IDF feature names:")
    print(word_vec.get_feature_names_out())
    print(pd.DataFrame(word_tfidf.toarray(),
                       columns=word_vec.get_feature_names_out()))

    char_vec = build_tfidf_char_vectorizer(max_features=20)
    char_tfidf = char_vec.fit_transform(texts)
    print("\nCharacter-level TF-IDF feature names:")
    print(char_vec.get_feature_names_out())
    print(pd.DataFrame(char_tfidf.toarray(),
                       columns=char_vec.get_feature_names_out()))
