# biobert_main.py  ──────────────────────────────────────────────────────────────
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
from datasets import Dataset, ClassLabel, Features, Value, Sequence
from sklearn.preprocessing import LabelEncoder
from src.preprocessing.masking_utils import apply_clinical_masking
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, AutoConfig)
from transformers.utils import logging as hf_logging
from sklearn.utils import resample

# show tqdm progress bars
hf_logging.set_verbosity_info()


def main() -> None:
# ────────── CLI ───────────────────────────────────────────────────────────────
    MODE = "train"


    
    from src.preprocessing.BioBert_Preprocessing import bio_bert_preprocessing
    from src.preprocessing.tokenizer_utils       import tokenizer, chunk_text   

    reports_df = pd.read_csv("data/TCGA_Reports.csv")
    labels_df  = pd.read_csv("data/tcga_patient_to_cancer_type.csv")
    reports_df["patient_id"] = reports_df["patient_filename"].str.split(".").str[0]
    df = pd.merge(reports_df, labels_df, on="patient_id")

    df = bio_bert_preprocessing(df)                      # full masking pipeline

    df = apply_clinical_masking(df, text_column="clean_text")

    
    label_enc = LabelEncoder()
    df["label"] = label_enc.fit_transform(df["cancer_type"])
    
    max_count = df["label"].value_counts().max()

    df_oversampled = pd.concat([
        resample(
            df[df["label"] == label],
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        for label in df["label"].unique()
    ])

    # Shuffle the new oversampled dataframe
    df = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print("✅ Oversampled label distribution:")
    print(df["label"].value_counts())


    # 1) Build a Hugging Face dataset from the text + label columns
    hf_ds = Dataset.from_pandas(df[["clean_text", "label"]])

    # 2) Encode each report (take the first 510-token payload, CLS/SEP added later)
    def encode(example):
        payload = chunk_text(example["clean_text"])[0]          # first slice (≤510 ids)
        return tokenizer.prepare_for_model(
            payload,
            max_length=512,
            padding="max_length",
            truncation=True
        )

    hf_ds = hf_ds.map(encode, remove_columns=["clean_text"])

    # 3) Declare the final feature schema (ClassLabel enables stratified split)
    num_classes = len(label_enc.classes_)
    features = Features({
        "input_ids":      Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int32")),
        "token_type_ids": Sequence(Value("int32")),
        "label":          ClassLabel(num_classes=num_classes,
                                    names=label_enc.classes_.tolist())
    })

    hf_ds = hf_ds.cast(features)

    # 4) 80 / 20 stratified split
    hf_ds = hf_ds.train_test_split(test_size=0.2,
                                stratify_by_column="label",
                                seed=42)
    train_ds, test_ds = hf_ds["train"], hf_ds["test"]
    
    MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.1"
    num_labels = len(label_enc.classes_)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    
    training_args = TrainingArguments(
    output_dir="ckpts",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,                      # more frequent logging
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    num_train_epochs=3,                    # ✅ same as best found
    per_device_train_batch_size=16,         # ✅ updated
    per_device_eval_batch_size=16,          # ✅ updated
    warmup_ratio=0.077,                     # ✅ updated
    weight_decay=0.018,                     # ✅ updated
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
    logging_first_step=True,
    logging_dir="logs",
    dataloader_num_workers=4,
    disable_tqdm=False,
    )


    
    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy":  accuracy_score(p.label_ids, preds),
            "precision": precision_score(p.label_ids, preds, average="macro", zero_division=0),
            "recall":    recall_score  (p.label_ids, preds, average="macro", zero_division=0),
            "f1":        f1_score      (p.label_ids, preds, average="macro"),
        }

    trainer = Trainer(
        model_init   = model_init,
        args         = training_args,
        train_dataset= train_ds,
        eval_dataset = test_ds,
        tokenizer    = tokenizer,
        data_collator= DataCollatorWithPadding(tokenizer),
        compute_metrics = metrics,
    )

    
    if MODE == "hpo":
        from transformers.trainer_utils import IntervalStrategy
        from optuna.samplers import TPESampler

        def hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.2),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", [8, 16]),
            }

        best = trainer.hyperparameter_search(
            direction      = "maximize",
            hp_space       = hp_space,
            n_trials       = 15,
            backend        = "optuna",
            sampler        = TPESampler(seed=42),
        )
        print("▶ best trial:", best)

        # resume & save final best model
        trainer.train(resume_from_checkpoint=True)
        trainer.save_model("ckpts/final_model")
        tokenizer.save_pretrained("ckpts/final_model")
        sys.exit(0)

    # ────────── STANDARD TRAIN / EVAL ─────────────────────────────────────────────
    if MODE == "train":
        trainer.train()
        trainer.save_model("ckpts/test_model")
        tokenizer.save_pretrained("ckpts/test_model")

    elif MODE == "eval":
        ckpt_dir = sorted(Path("ckpts").glob("checkpoint-*"),
                        key=lambda p: int(p.name.split('-')[-1]))[-1]
        trainer.model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        res = trainer.evaluate()
        print("\n===  Evaluation  ===")
        for k, v in res.items():
            print(f"{k:10s}: {v:.4f}")

if __name__ == "__main__":
    # Needed on macOS/Windows when using spawn-start multiprocessing
    main()