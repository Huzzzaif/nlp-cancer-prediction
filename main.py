#------imports for part 1
import pandas as pd
from src.preprocessing import clean_text

#-----imports for part 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

#-----Part 1 : Data merging and cleaning------ 

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

merged_df['clean_text'] = merged_df['text'].apply(clean_text)
# print(merged_df[['clean_text']].head())


#---part 2-----

#not we will convert cleaned text to TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(merged_df['clean_text'])

print("Shape of the TF-IDF vector:",X.shape)

encoder = LabelEncoder()
y = encoder.fit_transform(merged_df['cancer_type'])

print("Classes:", encoder.classes_)

# Stratify to ensure each cancer type is evenly split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#train model
model= LinearSVC()
model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)

#evaluate
print(classification_report(y_test, y_pred, target_names=encoder.classes_ ))