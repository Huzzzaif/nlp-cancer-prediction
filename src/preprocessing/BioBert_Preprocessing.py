import re
import unicodedata
import spacy, re
from spacy.language import Language
from scispacy.abbreviation import AbbreviationDetector
nlp_abbr = spacy.load("en_core_sci_sm")


# Then just:
nlp_abbr.add_pipe("abbreviation_detector", last=True)
print(nlp_abbr.pipe_names)
# Manual fallback dictionary: abbrev → long form

FALLBACK_ABBR = {
    "MRI":  "magnetic resonance imaging",
    "CT":   "computed tomography",
    "ECG":  "electrocardiogram",
    "X-ray":"x-ray",
    "NPO":  "nothing by mouth",
    "PET":  "positron emission tomography",
    "EEG":  "electroencephalogram",
    "IV":   "intravenous",
}


def clean_text(text: str) -> str:
    """
    Clean the text for BioBERT fine-tuning by:
    - Normalizing unicode
    - Replacing special characters
    - Removing extra whitespaces
    - Applying clinical masking
    - Removing non-essential punctuation
    """
    # Normalize unicode characters (e.g., é → e)
    text = unicodedata.normalize("NFC", str(text))
    
    # Replace special characters with standard ones
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-")
    
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove excessive punctuation like repeated "!!!" or "???"
    text = re.sub(r'[!?.]{2,}', '', text)  # Remove multiple punctuation marks

    # Remove non-essential punctuation (only keeping periods, commas, dashes, colons, etc.)
    text = re.sub(r'[^\w\s,;:.()-]', '', text)  # Keep essential punctuation

    # Add domain-specific cleaning (e.g., handle medical abbreviations)
    #text = handle_medical_abbreviations(text)
    
    return text

def handle_medical_abbreviations(text: str) -> str:
    # ---------- 1) Expand abbreviations detected by AbbreviationDetector ----------
    doc = nlp_abbr(text)
    #Print detected abbreviations for each document
    if doc._.abbreviations:
        print("\nDetected Abbreviations:")
        for abrv in doc._.abbreviations:
            print(f"Short: {abrv.text} | Long form: {abrv._.long_form.text}")
    new_text = text
    for abrv in sorted(doc._.abbreviations, key=lambda a: a.start_char, reverse=True):
        long_form = abrv._.long_form.text
        # Replace only the SHORT-FORM span with  "long form <ABBR>"
        new_text = (
            new_text[:abrv.start_char] +
            f"{long_form} <{abrv.text}>" +
            new_text[abrv.end_char:]
        )
    
    # ---------- 2) Fallback expansion for orphan short-forms ----------
    for abbr, long_form in FALLBACK_ABBR.items():
        # If the tag already exists, skip
        pattern = rf"\b{abbr}\b(?!\s*<)"          # short-form not already followed by <ABBR>
        new_text = re.sub(pattern,
                          f"{long_form} <{abbr}>",
                          new_text,
                          flags=re.IGNORECASE)
    
    return new_text

def bio_bert_preprocessing(df):
  
    # Apply text cleaning function
    df["clean_text"] = df["text"].apply(clean_text)
    
    return df

if __name__ == "__main__":
    # Small test example
    test_text = """
    The patient underwent Magnetic Resonance Imaging (MRI). 
Later, an MRI was repeated after chemotherapy.

    """

    print("\nOriginal Text:\n")
    print(test_text)

    # Apply your cleaning
    cleaned_text = clean_text(test_text)

    print("\nCleaned + Expanded Text:\n")
    print(cleaned_text)

    # If you want to see only abbreviations detected
    doc = nlp_abbr(test_text)
    print("\nAbbreviations Detected:")
    for abrv in doc._.abbreviations:
        print(f"Short: {abrv.text} | Long form: {abrv._.long_form.text}")
