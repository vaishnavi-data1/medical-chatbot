import spacy

# Load spaCy model (lightweight)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# -------------------------------------------------
# MASTER SYMPTOM LIST
# -------------------------------------------------
SYMPTOMS = [
    "fever",
    "cold",
    "cough",
    "headache",
    "vomit",
    "diarrhea",
    "fatigue",
    "weakness",
    "chest pain",
    "stomach pain",
    "sore throat"
]

PHRASES = [s for s in SYMPTOMS if " " in s]
WORDS = set(s for s in SYMPTOMS if " " not in s)

# -------------------------------------------------
# SYMPTOM EXTRACTION
# -------------------------------------------------
def extract_symptoms(text: str):
    """
    Extracts known symptoms from free-text input.
    Used for analytics, logs, or rule-based systems.
    """
    if not text:
        return []

    text = text.lower()
    doc = nlp(text)

    found = set()

    # Multi-word symptom detection
    for phrase in PHRASES:
        if phrase in text:
            found.add(phrase)

    # Single-word symptom detection
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in WORDS:
            found.add(lemma)

    return sorted(found)
