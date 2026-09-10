"""TF-IDF + Linear SVM baseline for cancer-type classification.

Evaluation protocol
-------------------
The train/test split happens on RAW TEXT, before any vectorizer is fit.
Vectorizers and SMOTE are fit on the training split only; the test split is
transformed with the already-fitted vectorizers and is never resampled.
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.analysis.check_leak import apply_clinical_masking
from src.features.feature_extraction import fit_transform_train_test
from src.preprocessing.cleaning_utils import clean_text

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 4000

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Step 1: Load and merge ---
reports_df = pd.read_csv("data/TCGA_Reports.csv")
labels_df = pd.read_csv("data/tcga_patient_to_cancer_type.csv")

reports_df['patient_id'] = reports_df['patient_filename'].apply(lambda x: x.split('.')[0])
merged_df = pd.merge(reports_df, labels_df, on='patient_id', how='inner')
print(f"Merged reports: {len(merged_df)}")

# --- Step 2: Clean ---
merged_df["clean_text"] = merged_df["text"].apply(clean_text)

# --- Step 3: Encode labels ---
label_encoder = LabelEncoder()
merged_df["label"] = label_encoder.fit_transform(merged_df["cancer_type"])
print(f"Classes: {len(label_encoder.classes_)}")

# --- Step 4: Clinical masking ---
merged_df = apply_clinical_masking(merged_df, text_column="clean_text")

# --- Step 5: Split FIRST, on raw text (prevents TF-IDF leakage) ---
train_text, test_text, y_train, y_test = train_test_split(
    merged_df["clean_text"],
    merged_df["label"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=merged_df["label"],
)
print(f"Train: {len(train_text)}  Test: {len(test_text)}")

# --- Step 6: Fit vectorizers on TRAIN ONLY, transform both ---
X_train, X_test, word_vectorizer, char_vectorizer = fit_transform_train_test(
    train_text, test_text, max_features=MAX_FEATURES
)
print(f"Feature matrix: train {X_train.shape}, test {X_test.shape}")

# --- Step 7: SMOTE on TRAIN ONLY ---
print("Class distribution BEFORE SMOTE:", dict(sorted(Counter(y_train).items())))
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("Class distribution AFTER SMOTE:", dict(sorted(Counter(y_train_resampled).items())))

# --- Step 8: Train ---
svm_model = LinearSVC(class_weight='balanced')
svm_model.fit(X_train_resampled, y_train_resampled)

# --- Step 9: Evaluate on untouched test split ---
y_pred = svm_model.predict(X_test)
report_txt = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, digits=4
)
print("\n=== Classification report (held-out test set) ===")
print(report_txt)

report_dict = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
)
metrics = {
    "model": "TF-IDF (word 1-2gram + char 3-5gram) + LinearSVC",
    "n_reports": int(len(merged_df)),
    "n_classes": int(len(label_encoder.classes_)),
    "n_train": int(len(train_text)),
    "n_test": int(len(test_text)),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "max_features_per_analyzer": MAX_FEATURES,
    "smote": "training split only",
    "vectorizers_fit_on": "training split only",
    "accuracy": report_dict["accuracy"],
    "macro_f1": report_dict["macro avg"]["f1-score"],
    "weighted_f1": report_dict["weighted avg"]["f1-score"],
    "macro_precision": report_dict["macro avg"]["precision"],
    "macro_recall": report_dict["macro avg"]["recall"],
    "per_class": {
        c: report_dict[c] for c in label_encoder.classes_ if c in report_dict
    },
}
with open("results/svm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("results/svm_classification_report.txt", "w") as f:
    f.write(report_txt)

np.savetxt(
    "results/svm_confusion_matrix.csv",
    confusion_matrix(y_test, y_pred),
    delimiter=",",
    fmt="%d",
)

print(f"accuracy    {metrics['accuracy']:.4f}")
print(f"macro F1    {metrics['macro_f1']:.4f}")
print(f"weighted F1 {metrics['weighted_f1']:.4f}")

# --- Step 10: Persist ---
dump(svm_model, "models/svm_model.joblib")
dump(word_vectorizer, "models/word_vectorizer.joblib")
dump(char_vectorizer, "models/char_vectorizer.joblib")
dump(label_encoder, "models/label_encoder.joblib")
print("\nSaved model, vectorizers, label encoder, and results/svm_metrics.json")
