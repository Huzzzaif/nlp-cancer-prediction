import pandas as pd
from collections import defaultdict

# Load reports and labels
reports_df = pd.read_csv("data/TCGA_Reports.csv")
labels_df = pd.read_csv("data/tcga_patient_to_cancer_type.csv")

# Prepare merged DataFrame
reports_df['patient_id'] = reports_df['patient_filename'].apply(lambda x: x.split('.')[0])
merged_df = pd.merge(reports_df, labels_df, on='patient_id', how='inner')

# If clean_text not already created, do it here
from preprocessing import clean_text
merged_df['clean_text'] = merged_df['text'].apply(clean_text)

# Column names
text_col = "clean_text"
label_col = "cancer_type"

# Terms you want to check
medical_terms = ["endometrial", "carcinosarcoma", "mullerian", "renal", "hepatic", "ovarian"]

# Frequency mapping function
def term_frequency_per_label(df, text_column, label_column, term_list):
    freq_map = defaultdict(lambda: defaultdict(int))

    for term in term_list:
        for label in df[label_column].unique():
            count = df[
                (df[text_column].str.contains(fr"\b{term}\b", case=False, regex=True)) &
                (df[label_column] == label)
            ].shape[0]
            freq_map[term][label] = count

    return freq_map

# Run analysis
freq_results = term_frequency_per_label(merged_df, text_col, label_col, medical_terms)

# Print results
for term, label_counts in freq_results.items():
    print(f"\n📌 Term: '{term}'")
    for label, count in label_counts.items():
        if count > 0:
            print(f" - {label}: {count} occurrences")
