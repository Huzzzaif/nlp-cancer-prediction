# 🧬 Cancer Type Prediction from Pathology Reports using NLP

This project applies Natural Language Processing (NLP) techniques to classify cancer types based on clinical pathology reports from The Cancer Genome Atlas (TCGA). It includes a baseline model using TF-IDF and Support Vector Machines (SVM), with plans to extend the project using domain-specific transformer models like BioBERT.

---

## 📚 Overview

Pathology reports are rich sources of diagnostic information. In this project, we build a machine learning pipeline that:

- Cleans and preprocesses raw medical text
- Converts reports into TF-IDF feature vectors
- Classifies the text into cancer types using a Linear SVM model
- Evaluates model performance with precision, recall, F1-score, and accuracy

---

## 🧠 Techniques Used

- **TF-IDF Vectorization**  
- **Linear Support Vector Classifier (SVM)**  
- **Label Encoding**  
- **Text Preprocessing with NLTK**  
- **Evaluation Metrics (Precision, Recall, F1-score)**  

---

## 📁 Dataset

- **Reports**: TCGA Pathology Reports from [Tatonetti Lab GitHub](https://github.com/tatonetti-lab/tcga-path-reports)  
- **Labels**: TCGA Patient to Cancer Type Mapping

> Each sample contains a free-text pathology report and a corresponding cancer type label.

---

## ⚙️ Project Structure
Cancer_pred_nlp/ 
├── data/ # Raw CSV files (reports + labels) 
├── src/ # Text preprocessing module 
├── notebooks/ # (optional) For EDA or visualization 
├── main.py # Main pipeline: clean → vectorize → train → evaluate
├── requirements.txt # Project dependencies 
└── README.md # You're reading it :)


---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Huzzzaif/nlp-cancer-prediction.git
cd nlp-cancer-prediction

# 2. Set up virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python main.py
