# Cancer Type Prediction from Pathology Reports (NLP)

Classifies **32 cancer types** from free-text TCGA pathology reports. Two models
are implemented: a TF-IDF + Linear SVM baseline and a fine-tuned BioBERT
classifier for contextual clinical-text representations. Headline metrics below
belong to TF-IDF/Linear SVM; BioBERT has not yet been evaluated on an independent
held-out test split.

---

## Results

**Dataset:** 9,523 pathology reports across 32 cancer types (stratified 80/20
split → 7,618 train / 1,905 test, `random_state=42`).

| Model | Accuracy | Macro F1 | Weighted F1 | Evaluation |
|---|---|---|---|---|
| TF-IDF (word 1-2gram + char 3-5gram) + LinearSVC | **0.9454** | **0.9428** | **0.9452** | Clean held-out test set |
| BioBERT (`dmis-lab/biobert-base-cased-v1.1`), 3 epochs | 0.9249 | 0.9058 | — | Validation / checkpoint-selection split; see caveat below |

SVM results are reproducible by running `SVM_main.py`; full output is committed
to [`results/svm_metrics.json`](results/svm_metrics.json) and
[`results/svm_classification_report.txt`](results/svm_classification_report.txt).

**Hardest classes** (SVM): LUSC (F1 0.726) and LUAD (F1 0.732) — lung squamous
cell carcinoma vs. lung adenocarcinoma are the main confusion pair, which is
consistent with their overlapping histological vocabulary. Every other class
scores above F1 0.82.

### ⚠️ Known limitation in the BioBERT number

The BioBERT figures come from `ckpts/checkpoint-2859/trainer_state.json`. That
run used a single 80/20 split with `load_best_model_at_end=True` and
`metric_for_best_model="f1"`, meaning the best epoch was *selected* using the
same split it is reported on. The number is therefore optimistic and is **not a
clean held-out result**. Fixing this requires a three-way train/validation/test
split (validation for checkpoint selection, test evaluated once). Until that
rerun, treat the SVM number as the defensible result.

---

## Approach

**Preprocessing** — lowercase, strip non-medical punctuation, remove stopwords
(retaining negation and clinical qualifiers such as *no*, *denies*, *malignant*,
*metastatic*).

**Clinical term masking** — pathology reports name the diagnosis directly, so a
naive model just memorizes site vocabulary. A hand-built dictionary of **64 distinct terms**
([`data/clinical_masking_dict.json`](data/clinical_masking_dict.json)) replaces
site- and diagnosis-specific terms (*mastectomy*, *glioblastoma*,
*prostatectomy*, …) with `[CLINICAL_MASK]` before vectorization. This reduces
target leakage from the text into the features.

**Features** — word-level TF-IDF (1–2 grams, 4,000 features) concatenated with
character-level TF-IDF (3–5 grams, 4,000 features) to capture both terminology
and spelling variation in dictated reports.

**Class imbalance** — classes range from 1,034 reports (BRCA) to 43 (CHOL).
For the SVM baseline, SMOTE is applied to the training split only.

### SVM evaluation protocol

The train/test split happens on **raw text, before any vectorizer is fit**.
Vectorizers and SMOTE are fit on the training split only; the test split is
transformed with the already-fitted vectorizers and is never resampled. Fitting
TF-IDF on the full corpus would leak test vocabulary, document frequencies, and
`max_features` selection into training.

---

## Project structure

```
├── SVM_main.py                  # TF-IDF + LinearSVC baseline (end-to-end)
├── BioBert_main.py              # BioBERT fine-tuning
├── data/
│   ├── TCGA_Reports.csv         # pathology reports
│   ├── tcga_patient_to_cancer_type.csv
│   ├── clinical_masking_dict.json
│   └── cancer_type_counts.csv
├── src/
│   ├── preprocessing/           # cleaning, masking, tokenization
│   ├── features/                # TF-IDF extraction
│   ├── analysis/                # leakage checks, term frequency
│   └── models/
├── results/                     # committed metrics (model weights excluded)
└── ckpts/                       # BioBERT trainer_state.json only
```

Model weights (`.safetensors`, `.pt`, `.bin`) are excluded from git — they
exceed GitHub's 100 MB limit. The `trainer_state.json` files are committed so
training metrics are inspectable without downloading checkpoints.

---

## Setup

```bash
git clone https://github.com/Huzzzaif/nlp-cancer-prediction.git
cd nlp-cancer-prediction

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The SVM baseline needs nothing further:

```bash
python SVM_main.py
```

BioBERT preprocessing additionally requires the SciSpaCy model, which is not on
PyPI:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

---

Before rerunning `BioBert_main.py`, move oversampling after the training split
and introduce a separate validation split for checkpoint selection. The current
script is experimental and should not be used to claim an independent test score.

## Data

- **Reports:** TCGA pathology reports, from the
  [Tatonetti Lab](https://github.com/tatonetti-lab/tcga-path-reports)
  (publicly distributed).
- **Labels:** TCGA patient → cancer type mapping.

The label file covers 33 cancer types; LAML has no associated pathology reports,
so the inner join yields 32 classes over 9,523 reports.

---

## Known issues

- BioBERT checkpoint selection uses the test split (see caveat above); needs a
  three-way split and a rerun.
- `BioBert_main.py` oversamples before splitting, which would place duplicate
  reports in both train and test. The committed checkpoint predates this code
  (its test set is 1,905 = 20% of 9,523, not the 6,618 oversampling would
  produce), but the file needs fixing before the next run.
- `src/analysis/analyse_top_terms.py` expects a single `tfidf_vectorizer.joblib`
  from an earlier architecture; `SVM_main.py` now saves separate word and
  character vectorizers.
- `vis_SVM.py` and `vis_bert.py` are not standalone runnable.
- Clinical masking is a hand-built dictionary and is unlikely to be exhaustive;
  residual site-specific vocabulary may remain.

---

## License

MIT — see [LICENSE](LICENSE).
