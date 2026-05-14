import numpy as np
import joblib
import nltk
import re
import pathlib
import torch
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from models.search_space import MultiOutputMlp
import json
import sys

if getattr(sys, 'frozen', False):
    BASE_DIR = pathlib.Path(sys._MEIPASS)
else:
    BASE_DIR = pathlib.Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

nltk.download("stopwords", quiet=True)

vectorizers    = joblib.load(DATA_DIR / "tfidf_vectorizers.pkl")
label_encoders = joblib.load(DATA_DIR / "label_encoders.pkl")

with open(RESULTS_DIR / "best_params.json") as f:
    best_params = json.load(f)


model = MultiOutputMlp(best_params, input_dim=1757, n_titre=140, n_parent=27)
model.load_state_dict(torch.load(RESULTS_DIR / "best_model.pth", map_location="cpu"))
model.eval()

IT_NOISE = {
    "bonjour", "merci", "cordialement", "svp", "besoin",
    "problème", "ticket", "bonne", "journée", "salutation"
}
french_stop_words = set(stopwords.words("french")) | IT_NOISE
stemmer = SnowballStemmer("french")

def clean(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', " ", text)          # remove HTML
    text = text.lower()
    text = re.sub(r'[^\w\s]', " ", text)        # remove punctuation
    text = re.sub(r'\d+', " ", text)            # remove numbers
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in french_stop_words and len(w) > 2]
    return " ".join(tokens)

COLUMNS = [
    "Titre",
    "Description",
    "Offre Libellé d'affichage",
    "Votre demande concerne Libellé",
    "Précisez votre demande (1) Libellé",
    "Précisez votre demande (2) Libellé",
    "Service impacté Libellé d'affichage",
]

def vectorize(inputs: dict):
    parts = []
    for col in COLUMNS:
        raw  = inputs.get(col, "")
        cleaned = clean(raw)
        vec = vectorizers[col].transform([cleaned])   
        parts.append(vec)
    combined = sp.hstack(parts)                       
    return torch.tensor(combined.toarray(), dtype=torch.float32)

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(inputs: dict):
    
    x = vectorize(inputs)

    with torch.no_grad():
        titre_logits, parent_logits = model(x)

    titre_probs  = torch.softmax(titre_logits,  dim=1)
    parent_probs = torch.softmax(parent_logits, dim=1)

    titre_idx  = titre_probs.argmax(dim=1).item()
    parent_idx = parent_probs.argmax(dim=1).item()

    titre_label  = label_encoders["Catégorisation Titre"].inverse_transform([titre_idx])[0]
    parent_label = label_encoders["Catégorisation Parent de 1er niveau"].inverse_transform([parent_idx])[0]

    return {
        "titre":             titre_label,
        "parent":            parent_label,
        "titre_confidence":  round(titre_probs[0, titre_idx].item()  * 100, 2),
        "parent_confidence": round(parent_probs[0, parent_idx].item() * 100, 2),
    }