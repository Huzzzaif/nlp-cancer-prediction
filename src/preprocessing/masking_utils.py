import os
import json
import re

def load_clinical_masking_dict():
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'clinical_masking_dict.json')
    with open(file_path, 'r') as f:
        return json.load(f)

def apply_clinical_masking(df, text_column="clean_text"):
    masking_dict = load_clinical_masking_dict()
    all_terms = list({term for terms in masking_dict.values() for term in terms})  # Flatten and deduplicate

    def mask_terms(text):
        for term in all_terms:
            pattern = r"\b" + re.escape(term.lower()) + r"\b"
            text = re.sub(pattern, "[CLINICAL_MASK]", text, flags=re.IGNORECASE)
        return text

    df[text_column] = df[text_column].apply(mask_terms)
    return df
