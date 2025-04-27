# src/model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_biobert_model(num_labels=32):
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_biobert_model()
    print("✅ BioBERT model and tokenizer loaded successfully.")
