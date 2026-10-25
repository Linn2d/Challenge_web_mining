import streamlit as st
import base64
import json
import requests
import psycopg2
import pdfplumber
import pandas as pd
import string
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import plotly.express as px
from sklearn.manifold import TSNE
from time import sleep
import logging
import os
import math
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import time
import PIL.Image
import pypdfium2 as pdfium

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
API_URL = "https://api.mistral.ai/v1/chat/completions"

#####################################
# Fonctions Utilitaires & Connexion DB
#####################################

#📌 Pour l'insertion des cv
def fix_json_value(val):
    """
    Si val est une valeur NaN (float ou 'nan' en texte),
    on retourne 'null' (littéral JSON).
    Sinon, on retourne la valeur telle quelle.
    """
    if isinstance(val, float) and math.isnan(val):
        return 'null'
    if isinstance(val, str) and val.lower() == 'nan':
        return 'null'
    return val

#📌 Connection a la bdd
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="webmining",
            user="postgres",
            password="postgres",
            host="my_postgres_container",
            port="5432",
        )
        return conn
    except Exception as e:
        logging.error("Erreur de connexion à la base de données: %s", e)
        return None

#####################################
# Partie 1 : Insertion CSV dans la table cv
#####################################

#📌 Fonction pour importer dans la bdd les csv contenant les informations des CSV
def import_csv_to_cv():
    csv_folder = "/app/data/csv"
    # csv_folder = os.path.join(os.path.dirname(__file__), "..", "data", "csv") # chemin du dossier contenant les fichiers CSV
    csv_files = [
        os.path.join(csv_folder, f)
        for f in os.listdir(csv_folder)
        if f.lower().endswith(".csv")
    ]
    logging.info("Fichiers CSV détectés : %s", csv_files)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Création de la table cv si elle n'existe pas
    create_table_query = """
    CREATE TABLE IF NOT EXISTS cv (
        "ID_CV" VARCHAR(5000) PRIMARY KEY,
        "Nom" TEXT,
        "Prenom" TEXT,
        "Adresse" TEXT,
        "CodePostal" TEXT,
        "Ville" TEXT,
        "NumeroTelephone" TEXT,
        "Email" TEXT,
        "Formations" JSONB,
        "Experiences" JSONB,
        "Projets" JSONB,
        "Competences" TEXT
    )
    """
    cursor.execute(create_table_query)
    conn.commit()
    
    # Requête d'insertion dans cv
    insert_query = """
    INSERT INTO cv (
        "ID_CV", 
        "Nom", 
        "Prenom", 
        "Adresse", 
        "CodePostal", 
        "Ville", 
        "NumeroTelephone", 
        "Email", 
        "Formations", 
        "Experiences", 
        "Projets", 
        "Competences"
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, 
        %s::jsonb, 
        %s::jsonb, 
        %s::jsonb, 
        %s
    )
    """
    
    # Boucle sur chaque CSV
    for csv_file_path in csv_files:
        df = pd.read_csv(csv_file_path, dtype=str)
        logging.info("Import de %s - %s lignes trouvées.", csv_file_path, len(df))
        
        for idx, row in df.iterrows():
            row_id_cv      = row.get("ID_CV", None)
            row_nom        = row.get("Nom", None)
            row_prenom     = row.get("Prenom", None)
            row_adresse    = row.get("Adresse", None)
            row_codepostal = row.get("CodePostal", None)
            row_ville      = row.get("Ville", None)
            row_telephone  = row.get("NumeroTelephone", None)
            row_email      = row.get("Email", None)
            # Pour les colonnes JSONB, on remplace NaN par 'null'
            row_formations  = fix_json_value(row.get("Formations", None))
            row_experiences = fix_json_value(row.get("Experiences", None))
            row_projets     = fix_json_value(row.get("Projets", None))
            row_competences = row.get("Competences", None)
            
            cursor.execute(
                insert_query,
                (
                    row_id_cv,
                    row_nom,
                    row_prenom,
                    row_adresse,
                    row_codepostal,
                    row_ville,
                    row_telephone,
                    row_email,
                    row_formations,
                    row_experiences,
                    row_projets,
                    row_competences
                )
            )
        conn.commit()
        logging.info("Fichier %s inséré avec succès dans 'cv'.", csv_file_path)
    
    cursor.close()
    conn.close()
    logging.info("Tous les CSV ont été importés dans 'cv'.")

#####################################
# Partie 2 : Traitement des PDF et insertion dans la table lm
#####################################

#📌Extraction du texte d'un PDF
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text
    except Exception as e:
        logging.error("Erreur d'extraction du texte depuis %s: %s", pdf_path, e)
        return None


def query_mistral(prompt, retries=5, delay=4):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small-2603",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=data,
                timeout=60
            )
            # Succès
            if response.status_code == 200:
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                    return json.loads(content)

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logging.error(
                        "Réponse Mistral invalide : %s", e
                    )
                    return None
            # Limite de débit atteinte
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_time = (
                        float(retry_after)
                        if retry_after is not None
                        else delay * (2 ** (attempt - 1))
                    )
                except ValueError:
                    wait_time = delay * (2 ** (attempt - 1))
                if attempt < retries:
                    logging.warning(
                        "Mistral : limite 429 atteinte. "
                        "Tentative %s/%s dans %.1f secondes.",
                        attempt, retries, wait_time
                    )
                    time.sleep(wait_time)
                    continue
                logging.error(
                    "Mistral : limite 429 toujours atteinte après %s tentatives. "
                    "Réponse : %s",
                    retries, response.text
                )
                return None

            # Autres erreurs HTTP
            logging.error(
                "Erreur API Mistral (status %s) : %s",
                response.status_code,
                response.text
            )

            # Inutile de réessayer pour la plupart des erreurs 4xx
            if 400 <= response.status_code < 500:
                return None

            if attempt < retries:
                time.sleep(delay * (2 ** (attempt - 1)))

        except requests.RequestException as e:
            logging.warning(
                "Tentative %s/%s : erreur réseau Mistral : %s",
                attempt, retries, e
            )

            if attempt < retries:
                time.sleep(delay * (2 ** (attempt - 1)))

    logging.error("Échec de l'appel à Mistral après %s tentatives.", retries)
    return None
# def query_mistral(prompt, retries=3, delay=4):
#     headers = {
#         "Authorization": f"Bearer {MISTRAL_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "model": "mistral-small-2603",
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": 0.7
#     }
#     for attempt in range(1, retries + 1):
#         try:
#             response = requests.post(API_URL, headers=headers, json=data)
#             if response.status_code != 200:
#                 logging.error("Erreur API (status %s): %s", response.status_code, response.text)
#                 sleep(delay)
#                 continue
#             content = response.json()["choices"][0]["message"]["content"]
#             result = json.loads(content)
#             return result
#         except (json.JSONDecodeError, KeyError) as e:
#             logging.warning("Tentative %s: Erreur JSON: %s", attempt, e)
#             sleep(delay)
#         except Exception as e:
#             logging.error("Tentative %s: Erreur lors de l'appel à l'API: %s", attempt, e)
#             sleep(delay)
#     return None

#📌 Extraction des données via l'API Mistral
def extract_data(text):
    prompt = f"""
    Voici une lettre de motivation :
    "{text}"
    Réponds uniquement par un objet JSON formaté exactement comme ceci sans ajouter de texte supplémentaire ni explications :

    {{
      "competences": ["compétence1", "compétence2"],
      "motivations": ["motivation1", "motivation2"]
    }}

    La réponse doit débuter par '{{' et se terminer par '}}'.
    """
    return query_mistral(prompt)

# 📌 Insertion dans la table lm (lettres de motivation) avec clé étrangère vers cv("ID_CV")
def insert_into_db(data, cv_id):
    if not data:
        logging.warning("Pas de données à insérer.")
        return False
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn:
            with conn.cursor() as cursor:
                # Création de la table lm si elle n'existe pas, avec contrainte FOREIGN KEY référant à cv("ID_CV")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS lm (
                        id SERIAL PRIMARY KEY,
                        competences JSON,
                        motivations JSON,
                        cv_id TEXT,
                        CONSTRAINT fk_cv
                          FOREIGN KEY (cv_id)
                          REFERENCES cv("ID_CV")
                    )
                """)
                cursor.execute("""
                    INSERT INTO lm (competences, motivations, cv_id)
                    VALUES (%s, %s, %s)
                """, (
                    json.dumps(data.get("competences")),
                    json.dumps(data.get("motivations")),
                    cv_id
                ))
        logging.info("Données insérées avec succès avec cv_id: %s", cv_id)
        return True
    except Exception as e:
        logging.error("Erreur PostgreSQL: %s", e)
        return False
    finally:
        conn.close()

# 📌 Traitement d'un PDF et insertion dans lm
def process_and_store_lm(pdf_path, cv_id):
    logging.info("Traitement du fichier : %s", pdf_path)
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logging.warning("Aucun texte extrait de %s", pdf_path)
        return False
    result = extract_data(text)
    if result:
        if insert_into_db(result, cv_id):
            logging.info("Résultat extrait : %s", result)
            return True
    logging.warning("Échec de l'extraction via l'API pour %s", pdf_path)
    return False

#📌 Traitement de tous les fichiers PDF du dossier spécifié avec assignation d'IDs
def process_folder():
    # Construction du chemin absolu vers le dossier "data/lm"
    # folder_path = os.path.join(os.path.dirname(__file__), "..", "data", "lm")
    folder_path = "/app/data/lm"
    cv_ids = [
        "ID789012",
        "ID459769",
        "ID459769",
        "ID876543",
        "ID108634",
        "ID789456",
        "ID987654",
        "ID749285",
        "ID931649",
        "ID915571",
        "ID699851",
        "ID153208",
        "ID176282",
        "ID323206",
        "ID485757"
    ]
    
    if not os.path.exists(folder_path):
        logging.error("Le dossier '%s' n'existe pas.", folder_path)
        return

    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]
    
    if not pdf_files:
        logging.error("Aucun fichier PDF trouvé dans le dossier %s.", folder_path)
        return
    
    if len(pdf_files) != len(cv_ids):
        logging.warning("Le nombre de fichiers PDF (%s) ne correspond pas au nombre d'IDs (%s).", len(pdf_files), len(cv_ids))
    
    for pdf_file, cv_id in zip(pdf_files, cv_ids):
        process_and_store_lm(pdf_file, cv_id)



# 📌 Fonction pour afficher un PDF dans Streamlit
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
    pdf_html = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="620" height="700" type="application/pdf"></iframe>'
    st.markdown(pdf_html, unsafe_allow_html=True)

    
# 📌 Récupérer toutes les offres d'emploi depuis la base de données
def get_offres_from_db():
    conn = get_db_connection()
    query = "SELECT * FROM annonces;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# 📌 Nettoyage et prétraitement des descriptions

nltk.download('stopwords')
nltk.download('punkt')

# Charger le modèle de langue française de spaCy
nlp = spacy.load("fr_core_news_sm")

# Charger la liste des stopwords français de NLTK et du fichier CSV
stop_words = set(stopwords.words('french'))
stop_Bastin = pd.read_csv("/app/data/french_stopwords.csv", sep=";")
stop_words.update(set(stop_Bastin["token"]))

# Ajouter les stopwords personnalisés
custom_stopwords = {
    "klanik", "bosch", "activités", "assure", "permettant", "partenaires", "côtés", "digixart", "amené", 
    "pricing", "rythme", "développée", "précision", "daurade", "exécuter", "vacances", "clients", "données", 
    "data", "expérience", "équipes", "compétences", "mission", "groupe", "solutions", "travail", "salesforce", 
    "techniques", "projets", "métiers", "technique", "outils", "équipe", "poste", "traitement", "an", "agence", 
    "commerciale", "marketing", "différents", "idéalement", "activité", "logiciel", "niveau", "connaissance", 
    "maîtrise", "faciliter", "profil", "participer", "bons", "business", "projet", "manipulation", "rejoignez", 
    "légumes", "fruits", "tickets", "building", "entreprise", "assurer", "métier", "tests", "complexes", 
    "gestion", "capacité", "formation", "répondront", "construction", "outil", "team", "offre", "programme", 
    "sein", "mise", "clarté", "travailler", "manuels", "suivi", "accompagner", "définition", "famille", "direction", 
    "prime", "primes", "utilisateurs", "familiale", "annuelle", "issu", "rémunération"
}
stop_words.update(custom_stopwords)

# Initialiser le stemmer français
stemmer = SnowballStemmer("french")


def preprocess_text(text):
    """
    Nettoie et prétraite un texte donné :
    - Minuscule
    - Suppression des caractères spéciaux
    - Suppression des stopwords
    - Tokenisation
    - Stemming et lemmatisation
    """

    # Normalisation des textes
    text = text.lower()
    text = text.replace("\n", " ")
    
    # Remplacement des termes techniques
    text = text.replace("ci/cd", "ci_cd").replace("power-bi", "power_bi").replace("big.data", "big_data")
    
    # Supprimer la ponctuation sauf les underscores
    text = re.sub(r"[^\w\s_]", " ", text)
    
    # Remettre les termes techniques à leur format initial
    text = text.replace("ci_cd", "ci/cd").replace("power_bi", "power-bi").replace("big_data", "big-data")
    
    # Remplacement de certaines expressions
    text = re.sub(r"gestion de projets?", "gestion-de-projet", text)
    
    # Suppression des underscores et des chiffres isolés
    text = re.sub(r"_", " ", text)
    text = re.sub(r'\b\d+\b', '', text)

    # Supprimer les stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Tokenisation
    tokens = word_tokenize(text)

    # Stemming
    stems = " ".join([stemmer.stem(word) for word in tokens])

    # Lemmatisation avec spaCy
    doc = nlp(text)
    lemmas = " ".join([token.lemma_ for token in doc])

    return text, stems, lemmas

# 📌 Génération d'un nuage de mots basé sur les descriptions de poste
def generate_wordcloud(df):
    # Prétraiter les descriptions (utilisation des lemmes)
    processed_texts = [preprocess_text(desc)[2] for desc in df['description'].dropna()]  # [2] pour les lemmes

    # Joindre tout le texte pour le wordcloud
    text = ' '.join(processed_texts)

    if not text.strip():  # Vérifier si le texte est vide après le preprocessing
        st.warning("⚠️ Pas assez de texte après prétraitement pour générer un nuage de mots.")
        return

    # Générer le WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Convertir en image numpy
    img_array = wordcloud.to_array()

    # Créer la figure Plotly
    fig = px.imshow(img_array)
    fig.update_layout(
        title="☁️ Nuage de Mots des Descriptions de Postes",
        coloraxis_showscale=False
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Afficher dans Streamlit
    st.plotly_chart(fig)

# 📌 Analyse de la distribution de l'expérience requise
def plot_experience_distribution(df):
    # Vérifier si la colonne 'experience' contient des données valides
    if df['experience'].dropna().empty:
        st.warning("Aucune donnée disponible pour la distribution des niveaux d'expérience.")
        return

    # Compter le nombre d'offres par niveau d'expérience
    experience_counts = df['experience'].value_counts().reset_index()
    experience_counts.columns = ['Niveau d\'expérience', 'Nombre d\'offres']

    # Création du graphique interactif avec Plotly
    fig = px.bar(
        experience_counts,
        x='Niveau d\'expérience',
        y='Nombre d\'offres',
        text='Nombre d\'offres',
        title="📊 Distribution des Niveaux d'Expérience dans les Offres",
        color='Nombre d\'offres',
        color_continuous_scale='Blues'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_title="Niveau d'expérience", yaxis_title="Nombre d'offres", showlegend=False)

    st.plotly_chart(fig)  # Affichage du graphique dans Streamlit

# 📌 Clustering des offres avec TF-IDF et KMeans
def cluster_offers(df, n_clusters=5):
    # Téléchargement des stopwords en français
    nltk.download('stopwords')
    french_stopwords = stopwords.words('french')

    # Initialisation du vectoriseur TF-IDF
    vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=500)

    # Appliquer preprocess_text et extraire seulement le texte nettoyé (premier élément du tuple)
    X = vectorizer.fit_transform(df['description'].dropna().apply(lambda text: preprocess_text(text)[0]))

    # Appliquer le clustering KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Ajouter un mapping pour afficher le nom du cluster
    cluster_names = {i: f"Cluster {i+1}" for i in range(n_clusters)}
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Afficher les termes les plus importants pour chaque cluster
    cluster_centers = kmeans.cluster_centers_
    feature_names = vectorizer.get_feature_names_out()

    clusters_terms = {}
    for i, center in enumerate(cluster_centers):
        top_terms_idx = center.argsort()[-10:][::-1]  # Obtenir les indices des 10 termes les plus importants
        top_terms = [feature_names[idx] for idx in top_terms_idx]
        clusters_terms[f"Cluster {i+1}"] = top_terms

    # Retourner le DataFrame avec les clusters et les termes associés à chaque cluster
    return df[['reference', 'intitule_poste', 'cluster_name']], clusters_terms, X

# 📌 Clustering visualisation
def plot_tsne(X, labels, title="t-SNE Visualisation des Clusters"):
    # Appliquer t-SNE pour réduire la dimensionnalité
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(X.toarray())  # Convertir en tableau dense si nécessaire

    # Créer un DataFrame pour la visualisation avec Plotly
    df_tsne = pd.DataFrame(reduced_data, columns=['x', 'y'])
    df_tsne['cluster'] = labels

    # Créer le graphique avec Plotly Express
    fig = px.scatter(df_tsne, x='x', y='y', color='cluster', title=title,
                     labels={'x': 't-SNE Axe 1', 'y': 't-SNE Axe 2', 'cluster': 'Cluster'},
                     color_continuous_scale='Viridis')

    fig.update_layout(title=title, title_x=0.5, template='plotly_dark')

    return fig

# Chargement du modèle FlauBERT
HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_uncased", token=HF_TOKEN or None)
model = AutoModel.from_pretrained("flaubert/flaubert_base_uncased", token=HF_TOKEN or None)

def generate_offers_embeddings(offers_df, text_column="description", save_path="embeddings_offres.joblib"):
    """
    Génère les embeddings des offres d'emploi et les sauvegarde.

    Args:
        offers_df (pd.DataFrame): DataFrame contenant les offres d'emploi.
        text_column (str): Nom de la colonne contenant le texte des offres.
        save_path (str): Chemin du fichier de sauvegarde des embeddings.
    
    Returns:
        dict: Dictionnaire des embeddings avec la référence de l'offre.
    """
    from utils import preprocess_text  # Import du prétraitement

    embeddings_dict = {}

    for index, row in offers_df.iterrows():
        offer_id = row["reference"]  # Adapter si nécessaire
        raw_text = row.get(text_column, "")

        if not raw_text.strip():  # Vérifier si le texte est vide
            continue
        
        cleaned_text = preprocess_text(raw_text)[0]  # Nettoyage du texte
        
        # Tokenisation
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Génération des embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        cls_embedding = last_hidden_states[:, 0, :].numpy()  # On prend la sortie du token [CLS]
        
        embeddings_dict[offer_id] = cls_embedding

    # Sauvegarde des embeddings
    with open(save_path, "wb") as f:
        joblib.dump(embeddings_dict, f)
    
    return embeddings_dict


def make_table():
    conn = get_db_connection()

    cur = conn.cursor()

    with open("/app/data/sql/db_webmining.sql", "r") as f:
        create_tables_sql = f.read()

    cur.execute(create_tables_sql)
    conn.commit()
    cur.close()
    conn.close()

def to_db(annonce: dict):
    # Connexion à la base de données PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()

    for competence in annonce["competences"]:
        cur.execute("""
            INSERT INTO competences (nom)
            VALUES (%s)
            ON CONFLICT (nom) DO NOTHING;
        """, (competence,))

    # Insert postes (job positions)
    cur.execute("""
        INSERT INTO annonces (intitule_poste, description, experience, divers, reference)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING reference;
    """, (annonce["intitule_poste"], annonce["description"], annonce["experience"], annonce["divers"], annonce["reference"]))
    
    # print(cur.fetchone())
    annonce_id = cur.fetchone()[0]
    print(annonce_id)

    for competence in annonce["competences"]:
        cur.execute("""
        INSERT INTO annonce_competences (annonce_reference, competence_id)
        SELECT %s, id FROM competences WHERE nom = %s;
        """, (annonce_id, competence))


    # Validation des modifications et fermeture de la connexion
    conn.commit()
    cur.close()
    conn.close()


def restore_annonces():

    import json

    annonces = []
    files = os.listdir("/app/data/annonces")
    from itertools import chain



    for file in files:
        annonce = {}
        with open("/app/data/annonces/"+file, "r") as f:
            doc = json.load(f)

        annonce["reference"] = doc["reference"]
        annonce["intitule_poste"] = doc["intitule_poste"]
        annonce["description"] = doc["description"]
        annonce["divers"] = " ".join(doc["divers"])
        annonce["experience"] = doc["experience"][0]
        annonce["competences"] = [competence if isinstance(competence, str) else competence[0] for competence in doc["competences"]]
        annonces.append(annonce)

    make_table()
    for annonce in annonces:
        print(annonce["reference"])
        to_db(annonce)


def generate_candidate_embeddings(cv_df, text_columns=["Competences", "Experiences", "Formations", "Projets", "motivations"]):
    """
    Génère des embeddings pour chaque candidature à partir des colonnes textuelles spécifiées.
    
    Pour chaque ligne du DataFrame, les valeurs des colonnes indiquées sont combinées
    en une seule chaîne, prétraitées, puis transformées en embedding via le modèle FlauBERT.
    
    Args:
        cv_df (pd.DataFrame): DataFrame issue de la jointure cv-lm.
        text_columns (list): Liste des colonnes à utiliser pour générer l'embedding.
        
    Returns:
        np.array: Matrice d'embeddings (chaque ligne correspond à l'embedding d'une candidature).
    """
    candidate_texts = []
    for idx, row in cv_df.iterrows():
        combined_text = " ".join([str(row.get(col, "")) for col in text_columns])
        candidate_texts.append(combined_text)
    
    # Appliquer le prétraitement sur chaque texte (on conserve le texte nettoyé, qui est le premier élément du tuple)
    preprocessed_texts = [preprocess_text(text)[0] for text in candidate_texts]
    
    embeddings = []
    for text in preprocessed_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # On prend le vecteur du token [CLS] (première position)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding)
    
    embeddings = np.vstack(embeddings)
    return embeddings

# 📌 Récupérer les candidatures
def get_candidatures():
    """
    Récupère la jointure entre la table cv et la table lm
    (cv."ID_CV" et lm.cv_id) et retourne un DataFrame contenant
    toutes les colonnes de cv ainsi que la colonne 'motivations' de lm.
    """
    conn = get_db_connection()
    query = """
        SELECT cv.*, lm.motivations
        FROM cv
        INNER JOIN lm ON cv."ID_CV" = lm.cv_id;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def pdf_to_image(pdf_bytes, scale=2.0, max_pages=3):
    """Convertit un PDF en une seule image (pages empilées verticalement)."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        nb_pages = min(len(pdf), max_pages)
        pages = [
            pdf[i].render(scale=scale).to_pil().convert("RGB")
            for i in range(nb_pages)
        ]
    finally:
        pdf.close()

    if len(pages) == 1:
        return pages[0]

    width = max(p.width for p in pages)
    height = sum(p.height for p in pages)
    combined = PIL.Image.new("RGB", (width, height), "white")
    y = 0
    for p in pages:
        combined.paste(p, (0, y))
        y += p.height
    return combined


def load_cv_image(uploaded_file):
    """Retourne une image PIL, que le fichier envoyé soit une image ou un PDF."""
    if uploaded_file.name.lower().endswith(".pdf"):
        return pdf_to_image(uploaded_file.getvalue())
    return PIL.Image.open(uploaded_file).convert("RGB")