# analyze_top_terms.py

import numpy as np
import pandas as pd
from joblib import load

def get_top_tfidf_features(classifier, vectorizer, label_names, top_n=5):
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features_per_class = {}

    for i, class_label in enumerate(label_names):
        top_n_idx = np.argsort(classifier.coef_[i])[-top_n:]
        top_n_terms = feature_names[top_n_idx]
        top_features_per_class[class_label] = list(reversed(top_n_terms))  # most important first

    return top_features_per_class

if __name__ == "__main__":
    # Load model components (adjust paths as needed)
    clf = load("models/svm_model.joblib")              # Your trained LinearSVC model
    vectorizer = load("models/tfidf_vectorizer.joblib") # Your fitted TfidfVectorizer
    label_encoder = load("models/label_encoder.joblib") # Your fitted LabelEncoder

    # Get class labels
    label_names = label_encoder.classes_

    # Extract top terms
    top_terms = get_top_tfidf_features(clf, vectorizer, label_names, top_n=5)

    # Display or save
    df_terms = pd.DataFrame.from_dict(top_terms, orient='index', columns=[f"Top_{i+1}" for i in range(5)])
    df_terms.to_csv("results/top_tfidf_terms_per_cancer.csv")
    print(df_terms)
