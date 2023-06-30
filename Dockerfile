# Utiliser une image officielle de Python comme image parente
FROM python:3.10

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Ajouter le fichier `requirements.txt`
COPY requirements.txt ./

# Installer les dépendances
RUN pip3 install -r requirements.txt

# Copier le reste du code de l'application dans le répertoire de travail
COPY api /app/api
COPY lib /app/lib
COPY storage /app/storage

# Exposer le port sur lequel l'application s'exécute
EXPOSE 8000

# Définir la variable d'environnement
ENV MODULE_NAME=main APP_NAME=api
ENV SECRET_KEY="Datascientest_FastAPI"
ENV ALGORITHM="HS256"
ENV ACESS_TOKEN_EXPIRE_MINUTE=120
ENV API_PORT=8000

ENV DB_AUTH_STORAGE_PATH="/app/storage"
#ENV API_USER_EMAIL="admin"
#ENV API_USER_PASSWORD="4dm1N"

ENV STORAGE_PATH="/app/storage"
ENV PROD_MODEL_NAME="mlops_cnn_vit_model_weights"

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

# Commande à exécuter au lancement du conteneur
WORKDIR /app/api
CMD uvicorn $MODULE_NAME:$APP_NAME --host 0.0.0.0 --port $API_PORT

#CMD echo ${STORAGE_PATH}
#CMD ls ${STORAGE_PATH}

# ommande pour build : docker build -t covid_project_app .
# commande pour run: docker run --rm -p 8000:8000 covid_project_app
#  docker run --rm -p 8000:8000 -v $(pwd)/storage_for_docker/storage:/app/storage strivly/covid:main