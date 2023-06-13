# Utiliser une image officielle de Python comme image parente
FROM python:3.10

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Ajouter le fichier `requirements.txt`
COPY requirements.txt ./

# Installer les dépendances
RUN pip3 install -r requirements.txt

# Copier le reste du code de l'application dans le répertoire de travail
COPY ./api ./api
COPY ./lib ./lib

# Exposer le port sur lequel l'application s'exécute
EXPOSE 8000

# Définir la variable d'environnement pour FastAPI
ENV MODULE_NAME=main APP_NAME=api

# Commande à exécuter au lancement du conteneur
#CMD uvicorn $MODULE_NAME:$APP_NAME --host 0.0.0.0 --port 8000
WORKDIR /app/api
CMD uvicorn $MODULE_NAME:$APP_NAME --host 0.0.0.0 --port 8000
