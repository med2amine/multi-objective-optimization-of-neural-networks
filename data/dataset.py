from pathlib import Path
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import scipy.sparse as sp
import joblib

nltk.download('stopwords')

le = LabelEncoder()
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "Demande_20260504_1039404585210306716138048554.xlsx"

dataset = pd.read_excel(file_path)
copydataset = dataset.copy()

copydataset = copydataset.dropna(subset=["Description"])

copydataset["Date/Heure de création"]   = pd.to_datetime(copydataset["Date/Heure de création"])
copydataset["Date/Heure de résolution"] = pd.to_datetime(copydataset["Date/Heure de résolution"])
copydataset["Resolution time"] = (
    copydataset["Date/Heure de résolution"] - copydataset["Date/Heure de création"]
)

copydataset = copydataset.drop(columns=[
    "Date/Heure de création", "Date/Heure de résolution", "ID",
    "Type de résolution", "Code Achèvement", "Demandé pour E-mail",
    "Compteur de rejet", "Compteur de réaffectation", "NPS",
    "Demandé pour Nom", "Affectation actuelle", "Groupe d'affectation Nom",
    "Connaissance(s) associée(s)", "KM à créer", "KM à mettre à jour",
    "Aucun besoin de connaissance"
])

copydataset["Catégorisation Parent de 1er niveau"] = copydataset["Catégorisation Parent de 1er niveau"].fillna("Unknown")
copydataset["Catégorisation Parent de 2e niveau"]  = copydataset["Catégorisation Parent de 2e niveau"].fillna("Unknown")

# ── Text cleaning ─────────────────────────────────────────────────────────────
french_stopwords = set(stopwords.words("french"))
stemmer = SnowballStemmer("french")

# ── NEW: expanded French IT stopwords — these words appear everywhere but
#         carry no discriminative signal for ticket classification
IT_NOISE_WORDS = {
    "bonjour", "merci", "bonne", "journée", "cordialement", "svp", "stp",
    "besoin", "aide", "problème", "probleme", "ticket", "demande", "service",
    "utilisateur", "pouvez", "pourriez", "faire", "avoir", "être", "etre",
    "suite", "objet", "mail", "message", "depuis", "toujours", "encore",
    "bien", "votre", "notre", "avez", "avons", "comme", "plus", "très", "tres"
}
ALL_STOPWORDS = french_stopwords | IT_NOISE_WORDS

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # remove HTML
    text = re.sub(r'<.*?>', " ", text)
    # lowercase
    text = text.lower()
    # remove punctuation and special characters
    text = re.sub(r'[^\w\s]', " ", text)
    # remove numbers
    text = re.sub(r'\d+', " ", text)
    # remove extra whitespace
    text = re.sub(r'\s+', " ", text).strip()
    # remove stopwords + IT noise
    tokens = text.split()
    tokens = [w for w in tokens if w not in ALL_STOPWORDS and len(w) > 2]
    # stem
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

# ── Decision variables ────────────────────────────────────────────────────────
decision_variables = [
    "Titre",
    "Description",
    "Offre Libellé d'affichage",
    "Votre demande concerne Libellé",
    "Précisez votre demande (1) Libellé",
    "Précisez votre demande (2) Libellé",
    "Service impacté Libellé d'affichage"
]

for var in decision_variables:
    copydataset[var] = copydataset[var].apply(clean_text)

print("Sample cleaned text:")
print(copydataset["Description"].head(3))

# ── Labels ────────────────────────────────────────────────────────────────────
x = copydataset[decision_variables]
y = copydataset[["Catégorisation Titre", "Catégorisation Parent de 1er niveau"]].copy()

label_encoders = {}
for col in y.columns:
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])
    label_encoders[col] = le

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y["Catégorisation Titre"]
)
print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")

# ── TF-IDF with bigrams ───────────────────────────────────────────────────────
# max_features: 500 → 1000 (more vocabulary)
# ngram_range: (1,1) → (1,2) (captures "accès refusé", "mot de passe", etc.)
# sublinear_tf: log scaling — dampens very frequent terms
# min_df=2: ignore terms appearing in only 1 document (noise)

# Per-column feature budgets — give more to the most informative columns
FEATURE_BUDGET = {
    "Titre":                                    1500,   # most informative
    "Description":                              2000,   # most informative
    "Offre Libellé d'affichage":                 500,
    "Votre demande concerne Libellé":            500,
    "Précisez votre demande (1) Libellé":        500,
    "Précisez votre demande (2) Libellé":        500,
    "Service impacté Libellé d'affichage":       300,
}

tfidf_vectors    = []
tfidf_vectorizers = {}

for var in decision_variables:
    tfidf = TfidfVectorizer(
        max_features  = FEATURE_BUDGET[var],
        ngram_range   = (1, 2),     # unigrams + bigrams
        sublinear_tf  = True,       # log(1 + tf) instead of raw tf
        min_df        = 2,          # ignore terms in only 1 document
        analyzer      = "word",
    )
    vec = tfidf.fit_transform(x_train[var])
    tfidf_vectorizers[var] = tfidf
    tfidf_vectors.append(vec)
    print(f"{var}: {vec.shape[1]} features")

x_train_tfidf = hstack(tfidf_vectors)
test_vectors  = [tfidf_vectorizers[var].transform(x_test[var]) for var in decision_variables]
x_test_tfidf  = hstack(test_vectors)

print(f"\nFinal input shape — train: {x_train_tfidf.shape} | test: {x_test_tfidf.shape}")

# ── Save ──────────────────────────────────────────────────────────────────────
sp.save_npz(BASE_DIR / "x_train.npz", x_train_tfidf)
sp.save_npz(BASE_DIR / "x_test.npz",  x_test_tfidf)

y_train.to_csv(BASE_DIR / "y_train.csv", index=False)
y_test.to_csv(BASE_DIR / "y_test.csv",   index=False)

joblib.dump(label_encoders,    BASE_DIR / "label_encoders.pkl")
joblib.dump(tfidf_vectorizers, BASE_DIR / "tfidf_vectorizers.pkl")

print("\nData saved successfully.")
for col in y.columns:
    print(f"\n{col}: {y[col].nunique()} classes")

