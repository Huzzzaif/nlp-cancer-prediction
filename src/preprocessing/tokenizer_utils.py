# utils/tokenizer_utils.py
from transformers import AutoTokenizer

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_text(text, max_length=512):
    return tokenizer(text, padding="max_length", truncation=True, max_length=max_length)

def chunk_text(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=False)  # Get tokens without truncating
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks
