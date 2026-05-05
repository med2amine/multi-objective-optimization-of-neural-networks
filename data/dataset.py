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

le = LabelEncoder()
nltk.download('stopwords')

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "Demande_20260504_1039404585210306716138048554.xlsx"

dataset = pd.read_excel(file_path)

copydataset = dataset.copy()

copydataset = copydataset.dropna(subset=["Description"])

print(copydataset.columns.tolist())

print(copydataset.isnull().sum())

copydataset["Date/Heure de création"] = pd.to_datetime(copydataset["Date/Heure de création"])
copydataset["Date/Heure de résolution"] = pd.to_datetime(copydataset["Date/Heure de résolution"])

copydataset["Resolution time"] = (
    copydataset["Date/Heure de résolution"] - copydataset["Date/Heure de création"]
)

print(copydataset[["Date/Heure de création", "Date/Heure de résolution", "Resolution time"]].head())

copydataset = copydataset.drop(columns=["Date/Heure de création","Date/Heure de résolution","ID","Type de résolution","Code Achèvement","Demandé pour E-mail","Compteur de rejet","Compteur de réaffectation","NPS","Demandé pour Nom","Affectation actuelle","Groupe d'affectation Nom","Connaissance(s) associée(s)","KM à créer","KM à mettre à jour","Aucun besoin de connaissance"])
print(copydataset.columns.tolist())

copydataset["Catégorisation Parent de 1er niveau"] = copydataset["Catégorisation Parent de 1er niveau"].fillna("Unknown")
copydataset["Catégorisation Parent de 2e niveau"] = copydataset["Catégorisation Parent de 2e niveau"].fillna("Unknown")

french_stopwords = set(stopwords.words("french"))
stemmer = SnowballStemmer("french")

def lower_text(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]',"",text)

def remove_html_tags(text):
    return re.sub(r'<.*?>',"",text)


def remove_stopwords(text):
    tokens = text.split()
    return " ".join(word for word in tokens if word not in french_stopwords)

def stem_text(text):
    tokens = text.split()
    return " ".join(stemmer.stem(word) for word in tokens)

def remove_numbers(text):
    return re.sub(r'\d+',"",text)

def remove_extra_whitespace(text):
    return re.sub(r'\s+'," ",text).strip()

def clean_text(text):
    if not isinstance(text,str):
        return ""
    text = remove_html_tags(text)
    text = lower_text(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    text = remove_extra_whitespace(text)
    return text

decision_variables = ["Titre","Description","Offre Libellé d'affichage","Votre demande concerne Libellé","Précisez votre demande (1) Libellé","Précisez votre demande (2) Libellé","Service impacté Libellé d'affichage"]

for var in decision_variables:
    copydataset[var] = copydataset[var].apply(clean_text)

print(copydataset[decision_variables].head())

x = copydataset[decision_variables]
y = copydataset[["Catégorisation Titre","Catégorisation Parent de 1er niveau"]].copy()

label_encoders = {}
for col in y.columns:
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])
    label_encoders[col] = le  


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

tfidf_vectors = []
tfidf_vectorizers = {}

for var in decision_variables:
    tfidf = TfidfVectorizer(max_features=500)
    vec = tfidf.fit_transform(x_train[var])  # fit only on train!
    tfidf_vectorizers[var] = tfidf
    tfidf_vectors.append(vec)

x_train_tfidf = hstack(tfidf_vectors)

test_vectors = [tfidf_vectorizers[var].transform(x_test[var]) for var in decision_variables]
x_test_tfidf = hstack(test_vectors)

for col in y.columns:
    print(f"\n{col}:\n", copydataset[col].value_counts())

sp.save_npz(BASE_DIR / "x_train.npz", x_train_tfidf)
sp.save_npz(BASE_DIR / "x_test.npz", x_test_tfidf)

y_train.to_csv(BASE_DIR / "y_train.csv", index=False)
y_test.to_csv(BASE_DIR / "y_test.csv", index=False)

joblib.dump(label_encoders, BASE_DIR / "label_encoders.pkl")
joblib.dump(tfidf_vectorizers, BASE_DIR / "tfidf_vectorizers.pkl")

print("Data saved successfully.")