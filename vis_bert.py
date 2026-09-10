import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from datasets import Dataset, Features, Sequence, Value, ClassLabel
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer

from src.preprocessing.BioBert_Preprocessing import bio_bert_preprocessing
from src.preprocessing.tokenizer_utils import tokenizer, chunk_text
from bioBert_model import BioBERTForWeightedClassification

# 1) ---------- Load & preprocess data -----------------
reports_df = pd.read_csv("data/TCGA_Reports.csv")
labels_df  = pd.read_csv("data/tcga_patient_to_cancer_type.csv")
reports_df["patient_id"] = reports_df["patient_filename"].str.split(".").str[0]
df = pd.merge(reports_df, labels_df, on="patient_id")
df = bio_bert_preprocessing(df)

# Label encoding
label_enc = LabelEncoder()
df["label"] = label_enc.fit_transform(df["cancer_type"])

# Compute class weights
cls_counts  = np.bincount(df["label"])
cls_weights = 1.0 / cls_counts
cls_weights = torch.tensor(cls_weights / cls_weights.sum(), dtype=torch.float)

# 2) ---------- Encode dataset --------------------------
hf_ds = Dataset.from_pandas(df[["clean_text", "label"]])

def encode(ex):
    payload = chunk_text(ex["clean_text"])[0]
    return tokenizer.prepare_for_model(
        payload, max_length=512, padding="max_length", truncation=True
    )

hf_ds = hf_ds.map(encode, remove_columns=["clean_text"])

features = Features({
    "input_ids":      Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int32")),
    "token_type_ids": Sequence(Value("int32")),
    "label": ClassLabel(num_classes=len(label_enc.classes_), names=label_enc.classes_.tolist())
})
hf_ds = hf_ds.cast(features)

# Split
hf_ds = hf_ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
test_ds = hf_ds["test"]

# 3) ---------- Load trained model -----------------------
ckpt_dir = "ckpts/test_model"
model = BioBERTForWeightedClassification.from_pretrained_with_weights(
    pretrained_model_name_or_path=ckpt_dir,
    class_weights=cls_weights
)
model.eval()

viz_trainer = Trainer(model=model, tokenizer=tokenizer)
preds = viz_trainer.predict(test_ds).predictions.argmax(1)
y_true = test_ds["label"]

# 4) ---------- Print classification report --------------
print("\n=== Classification Report ===")
print(classification_report(
    y_true,
    preds,
    target_names=test_ds.features["label"].names,
    digits=4
))

# 5) ---------- Confusion Matrix -------------------------
cm = confusion_matrix(y_true, preds)
fig, ax = plt.subplots(figsize=(12, 10))
norm = mcolors.LogNorm(vmin=0.1, vmax=cm.max())
cax = ax.imshow(cm, interpolation="nearest", cmap="Blues", norm=norm)

ax.set_title("Confusion Matrix – BioBERT", fontsize=16)
ax.set_xlabel("Predicted label", fontsize=14)
ax.set_ylabel("True label", fontsize=14)

labels = test_ds.features["label"].names
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        if val > 0:
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=6)

fig.colorbar(cax)
plt.tight_layout()
plt.show()

# 6) ---------- Bar Chart for Class Metrics --------------
precisions, recalls, f1s, _ = precision_recall_fscore_support(
    y_true,
    preds,
    labels=list(range(len(labels)))
)

metrics_df = pd.DataFrame({
    "Cancer Type": labels,
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1s
})

fig, ax = plt.subplots(figsize=(15, 6))
width = 0.25
x = np.arange(len(metrics_df))

ax.bar(x - width, metrics_df["Precision"], width=width, label='Precision')
ax.bar(x, metrics_df["Recall"], width=width, label='Recall')
ax.bar(x + width, metrics_df["F1-Score"], width=width, label='F1-Score')

ax.set_xlabel('Cancer Type', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
ax.set_title('BioBERT Performance by Cancer Type', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df["Cancer Type"], rotation=90, fontsize=8)
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()
