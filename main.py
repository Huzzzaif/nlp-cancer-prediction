#------imports for part 1
import pandas as pd
from src.preprocessing import clean_text
from src.check_leak import apply_clinical_masking

#-----imports for part 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

#------imports from part 3
from datasets import Dataset
from src.model import load_biobert_model
from joblib import dump

#-----Part 1 : Data merging and cleaning------ 

import sys

MODE = "eval"  # options: "train" or "eval"
if len(sys.argv) > 1:
    MODE = sys.argv[1]

# Load the reports
reports_df = pd.read_csv("data/TCGA_Reports.csv")
# print("Reports:")
# print(reports_df.head())

# Load the labels
labels_df = pd.read_csv("data/tcga_patient_to_cancer_type.csv")
# print("\nLabels:")
# print(labels_df.head())

reports_df['patient_id'] = reports_df['patient_filename'].apply(lambda x: x.split('.')[0])

merged_df = pd.merge(reports_df, labels_df, on='patient_id', how='inner')
# print("Merged shape:", merged_df.shape)
# print(merged_df[['patient_id', 'cancer_type', 'text']].head())
# 2. Clean the text column
merged_df["clean_text"] = merged_df["text"].apply(clean_text)

# 3. Label encode cancer_type
label_encoder = LabelEncoder()
merged_df["label"] = label_encoder.fit_transform(merged_df["cancer_type"])

# # 4. Count label mentions BEFORE masking
# print("🔍 Label Mentions BEFORE Masking:")
# pre_mask_counts = count_label_mentions(merged_df.copy(), label_encoder.classes_)
# for label, count in pre_mask_counts.items():
#     print(f"{label}: {count}")

# 5. Mask labels to avoid leakage
merged_df = apply_clinical_masking(merged_df, text_column="clean_text")

# # 6. Count label mentions AFTER masking
# print("\nLabel Mentions AFTER Masking:")
# post_mask_counts = count_label_mentions(merged_df.copy(), label_encoder.classes_)
# for label, count in post_mask_counts.items():
#     print(f"{label}: {count}")

# # 7. Deep check: are any labels still showing up after masking?
# unmasked_issues = []
# for label in label_encoder.classes_:
#     found = merged_df[merged_df["clean_text"].str.contains(fr"\b{label.lower()}\b", case=False, regex=True)]
#     if not found.empty:
#         print(f"Label '{label}' still appears in {len(found)} sample(s) after masking.")
#         unmasked_issues.append((label, len(found)))

# if not unmasked_issues:
#     print("\n🎉 All class labels were successfully masked from the clean_text!")


# ---------------------- Sample Comparison ----------------------

# import random

# label_to_check = "READ"

# # Find indices where the label appears in the raw text
# matching_indices = merged_df[merged_df['text'].str.contains(fr"\b{label_to_check}\b", case=False, regex=True)].index

# if matching_indices.any():
#     # Pick a random index from matches
#     sample_index = random.choice(matching_indices.tolist())

#     # Step 1: Original raw text
#     original_text = merged_df.loc[sample_index, 'text']

#     # Step 2: Cleaned version before masking
#     cleaned_before_masking = clean_text(original_text)

#     # Step 3: Cleaned & masked version
#     masked_text = merged_df.loc[sample_index, 'clean_text']

#     # Step 4: Display all three
#     print("\n📍 Sample Index:", sample_index)
#     print("\n🔹 Original Raw Text:\n", original_text[:2000])  # truncate if needed
#     print("\n🔹 Cleaned Before Masking:\n", cleaned_before_masking[:2000])
#     print("\n🔹 After Masking:\n", masked_text[:2000])
# else:
#     print(f"No sample found with label '{label_to_check}' in the original text.")


#---part 2-----

#not we will convert cleaned text to TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(merged_df['clean_text'])

print("Shape of the TF-IDF vector:",X.shape)


y = label_encoder.fit_transform(merged_df['cancer_type'])

print("Classes:", label_encoder.classes_)

# Stratify to ensure each cancer type is evenly split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#train model
model= LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#evaluate
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_ ))

# Save trained components to 'models/' directory (make sure it exists)
dump(model, "models/svm_model.joblib")
dump(vectorizer, "models/tfidf_vectorizer.joblib")
dump(label_encoder, "models/label_encoder.joblib")

print("Model, vectorizer, and label encoder saved successfully!")


import transformers
print("Transformers version in use:", transformers.__version__)



#-----BioBert Model_______
label_encoder = LabelEncoder()
#this basically converts class names into numbers
merged_df["label"] = label_encoder.fit_transform(merged_df["cancer_type"])

#now we create hugging face dataset
hf_dataset = Dataset.from_pandas(merged_df[["clean_text","label"]])
hf_dataset = hf_dataset.rename_column("clean_text","text")

# Load BioBERT tokenizer and model
tokenizer, model = load_biobert_model(num_labels=len(label_encoder.classes_))

#tokenizing
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True,max_length=512)

# Apply the tokenizer to the entire dataset
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

from datasets import ClassLabel

# Convert label column to ClassLabel type
num_classes = len(label_encoder.classes_)
features = tokenized_dataset.features.copy()
features["label"] = ClassLabel(num_classes=num_classes)

tokenized_dataset = tokenized_dataset.cast(features)
# Split into train/test sets with stratification by label
dataset_split = tokenized_dataset.train_test_split(test_size=0.2, stratify_by_column="label")

train_dataset = dataset_split["train"]
test_dataset = dataset_split["test"]

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

from transformers import DataCollatorWithPadding

# Initialize the data collator with your tokenizer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Pass the data collator to your Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if MODE == "train":
    trainer.train()
    trainer.save_model("./results")  # Save the fine-tuned model


if MODE == "eval":
    from transformers import AutoModelForSequenceClassification

    # Load the trained model from results folder
    model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-2859")

    # Re-initialize the Trainer with the loaded model
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Evaluate on the test set
    eval_results = trainer.evaluate()
    print("📊 Evaluation Results:")
    for k, v in eval_results.items():
        print(f"{k}: {v:.4f}")
