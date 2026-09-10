# --- Imports (organized cleanly) ---
import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from joblib import dump
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack

from src.preprocessing.cleaning_utils import clean_text
from src.analysis.check_leak import apply_clinical_masking
from src.features.feature_extraction import (
    extract_tfidf_word_ngrams,
    extract_tfidf_char_ngrams
)
from collections import Counter
import transformers  # Just checking version

# --- Step 1: Load and Merge Data ---
reports_df = pd.read_csv("data/TCGA_Reports.csv")
labels_df = pd.read_csv("data/tcga_patient_to_cancer_type.csv")

reports_df['patient_id'] = reports_df['patient_filename'].apply(lambda x: x.split('.')[0])
merged_df = pd.merge(reports_df, labels_df, on='patient_id', how='inner')

# --- Step 2: Clean and Preprocess Text ---
merged_df["clean_text"] = merged_df["text"].apply(clean_text)

# --- Step 3: Encode Labels ---
label_encoder = LabelEncoder()
merged_df["label"] = label_encoder.fit_transform(merged_df["cancer_type"])

# --- Step 4: Apply Clinical Masking ---
merged_df = apply_clinical_masking(merged_df, text_column="clean_text")

# --- Step 4.5: Random Checks for Cleaning and Masking ---
print("\n🔍 Random Sample Checks for Masking:")
for _ in range(3):
    random_idx = random.randint(0, len(merged_df) - 1)
    print(f"\nSample Index: {random_idx}")
    print("\nOriginal Text (Truncated):")
    print(merged_df.loc[random_idx, "text"][:800])
    print("\nCleaned and Masked Text (Truncated):")
    print(merged_df.loc[random_idx, "clean_text"][:800])
    mask_count = merged_df.loc[random_idx, "clean_text"].count("[CLINICAL_MASK]")
    print(f"\nNumber of [CLINICAL_MASK] tokens: {mask_count}")
print("\nRandom check done. Proceeding to feature extraction...\n")

# --- Step 5: Feature Extraction ---

# 5.1 TF-IDF word features
word_tfidf, word_vectorizer = extract_tfidf_word_ngrams(merged_df['clean_text'], max_features=4000)

# 5.2 TF-IDF char features
char_tfidf, char_vectorizer = extract_tfidf_char_ngrams(merged_df['clean_text'], max_features=4000)

# 5.4 Combine all features
from scipy.sparse import csr_matrix
X = hstack([word_tfidf, char_tfidf])
y = merged_df["label"]

print("Final feature matrix shape:", X.shape)
print("Number of classes:", len(label_encoder.classes_))

# --- Step 6: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 6.5: Apply SMOTE ---
print("\nClass distribution BEFORE SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("\nClass distribution AFTER SMOTE:", Counter(y_train_resampled))

# --- Step 7: Train SVM Model ---
svm_model = LinearSVC(class_weight='balanced')
svm_model.fit(X_train_resampled, y_train_resampled)

# --- Step 8: Evaluate Model ---
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- Step 9: Save Model and Vectorizer ---
dump(svm_model, "models/svm_model.joblib")
dump(word_vectorizer, "models/word_vectorizer.joblib")
dump(char_vectorizer, "models/char_vectorizer.joblib")
dump(label_encoder, "models/label_encoder.joblib")
print("\nModel, vectorizer, and label encoder saved successfully.")

# --- Step 10: Visualization Functions ---

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_per_class_metrics(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names))
    )
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)

    metrics_df.plot(
        x='Class',
        y=['Precision', 'Recall', 'F1-Score'],
        kind='bar',
        figsize=(16, 8)
    )
    plt.title('Per-Class Precision, Recall, F1-Score')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# --- Step 11: Run Visualizations ---
print("\nPlotting Confusion Matrix...")
plot_confusion_matrix(y_test, y_pred, class_names=label_encoder.classes_, normalize=True)

print("\nPlotting Per-Class Metrics...")
plot_per_class_metrics(y_test, y_pred, class_names=label_encoder.classes_)

# --- Step 12: Info ---
print("\nTransformers version in use:", transformers.__version__)
