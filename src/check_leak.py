import pandas as pd
import re
import json
import os
from collections import defaultdict
from src.preprocessing import clean_text

# -------------------------------------------
# 1. Label Leakage Checking
# -------------------------------------------
def check_label_leakage(df, text_column="clean_text", label_column="cancer_type"):
    leaks = []
    for label in df[label_column].unique():
        pattern = rf"\b{re.escape(label.lower())}\b"
        match_rows = df[df[text_column].str.contains(pattern, case=False, regex=True)]
        count = len(match_rows)
        if count > 0:
            leaks.append((label, count))
            print(f"\n🔍 '{label}' found in {count} samples")
            print("Examples:")
            print(match_rows[text_column].iloc[0][:2000])  # print a sample
    if not leaks:
        print("✅ No label leakage detected.")


# -------------------------------------------
# 2. Count Mentions of Labels
# -------------------------------------------
def count_label_mentions(df, labels, text_column="clean_text"):
    counts = defaultdict(int)
    for label in labels:
        pattern = fr"\b{re.escape(label.lower())}\b"
        count = df[text_column].str.contains(pattern, case=False, regex=True).sum()
        counts[label] = count
    return counts

# -------------------------------------------
# 3. Load Clinical Masking Dictionary
# -------------------------------------------
def load_clinical_masking_dict():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical_masking_dict.json')
    with open(file_path, 'r') as f:
        return json.load(f)


# -------------------------------------------
# 5. Apply Global Masking for All High-Risk Terms
# -------------------------------------------
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

# -------------------------------------------
# 6. Run as Standalone
# -------------------------------------------
if __name__ == "__main__":
    reports_df = pd.read_csv("data/TCGA_Reports.csv")
    labels_df = pd.read_csv("data/tcga_patient_to_cancer_type.csv")

    reports_df['patient_id'] = reports_df['patient_filename'].apply(lambda x: x.split('.')[0])
    merged_df = pd.merge(reports_df, labels_df, on='patient_id', how='inner')

    # Step 1: Clean text
    merged_df['clean_text'] = merged_df['text'].apply(clean_text)

    # Step 2: Load clinical terms dictionary
    clinical_dict = load_clinical_masking_dict()

    # ✅ Step 3: Mask direct class label mentions (e.g., "READ", "OV")
    # merged_df = mask_labels_in_text(
    #     merged_df,
    #     text_column="clean_text",
    #     label_column="cancer_type",
    #     mask_dict={label: [label] for label in merged_df['cancer_type'].unique()}
    # )

    # # ✅ Step 4: Load clinical dictionary & mask clinical terms (e.g., "endometrial", "serous")
    # clinical_dict = load_clinical_masking_dict()
    merged_df = apply_clinical_masking(merged_df, text_column="clean_text")

    # Step 5: Check for any label leakage
    print("\n🔍 Re-checking for label leakage after masking:\n")
    check_label_leakage(merged_df)

 
