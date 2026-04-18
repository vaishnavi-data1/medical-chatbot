import spacy
import re

# Load spaCy model (lightweight pipeline)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# -------------------------------------------------
# SYMPTOM NORMALIZATION MAP
# -------------------------------------------------
SYMPTOM_SYNONYMS = {
    "fever": ["temperature", "high temp"],
    "cough": ["coughing"],
    "cold": ["runny nose", "sneezing"],
    "headache": ["migraine"],
    "vomit": ["vomiting", "nausea"],
    "diarrhea": ["loose motion"],
    "fatigue": ["tired", "weakness"],
    "chest pain": ["tight chest"],
    "stomach pain": ["abdominal pain"],
    "sore throat": ["throat pain"]
}

# Build reverse lookup
WORD_MAP = {}
for symptom, synonyms in SYMPTOM_SYNONYMS.items():
    for word in synonyms + [symptom]:
        WORD_MAP[word] = symptom

PHRASES = [k for k in SYMPTOM_SYNONYMS if " " in k]

# -------------------------------------------------
# EXPAND SYMPTOMS FUNCTION
# -------------------------------------------------
def expand_symptoms(text: str) -> str:
    """
    Converts free-text user input into normalized symptom keywords.
    Used by backend before ML prediction.
    """
    if not text:
        return ""

    text = text.lower()
    doc = nlp(text)

    found = set()

    # Detect multi-word phrases first
    for phrase in PHRASES:
        if phrase in text:
            found.add(phrase)

    # Detect single-word symptoms
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in WORD_MAP:
            found.add(WORD_MAP[lemma])

    # Fallback: use filtered words if nothing matched
    if not found:
        return " ".join(
            token.text for token in doc
            if token.is_alpha and len(token.text) > 2
        )

    return " ".join(found)
