import pandas as pd

# Load your CSV file
file_path = 'data/tcga_patient_to_cancer_type.csv'  # <-- replace with your actual file name
df = pd.read_csv(file_path)

# Replace 'cancer_type' with your actual column name if different
cancer_counts = df['cancer_type'].value_counts()

# Print the counts
print("Cancer Type Counts:")
print(cancer_counts)

# (Optional) Save the counts to a new CSV
cancer_counts.to_csv('cancer_type_counts.csv', header=True)

print("\nCounts saved to 'cancer_type_counts.csv'!")
