# Challenge de fin d'année
Dans le cadre du Master 2 SISE de l'Université Lumière Lyon 2, nous avons dû développer un projet IA de notre choix en 2 jours.
## Master SISE 2024-2025

Notre équipe était constituée de **_Linh Nhi Le-Dinh_**, **_Daniella Rakotondratsimba_** et **_Joël Sollari_**.


# 🚀 Installation et utilisation de l'application

Bienvenue ! Voici comment installer et utiliser cette application facilement.

---

## 📥 Prérequis

- **Docker Desktop** : Assurez-vous d'avoir [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé sur votre machine.

---

## 📂 Cloner le dépôt

Ouvrez un terminal, naviguez vers le dossier de votre choix et clonez le dépôt GitHub :

```bash
git clone https://github.com/Linn2d/Challenge_web_mining.git
cd Challenge_web_mining
```

---

## 🛠️ Construction de l'image PostgreSQL

Naviguez dans le dossier `Postgres` puis construisez l'image Docker :

```bash
cd Postgres
docker build -t postgres .
```

---

## 📂 Démarrage des conteneurs Docker

Retournez à la racine du projet et accédez au dossier `docker` :

```bash
cd ../docker
docker compose up
```

Cela lancera automatiquement :
- La base de données PostgreSQL
- L'application Streamlit
- Airflow

---

## 🌐 Accéder à l'application

Ouvrez votre navigateur et rendez-vous à l'adresse suivante :

[Application Streamlit](http://localhost:8501)

[Airflow](http://localhost:8080)

Pour accéder à Airflow :
- nom d'utilisateur: airflow
- mot de passe: airflow
---

## 🔄 Arrêter les conteneurs

Pour arrêter proprement les conteneurs :

```bash
docker compose down
```

**Bonne exploration !** 📚🛠️

