# DS_MLOps

## API

### Secret values have to be define in /api/.env file or in local environement

> SECRET_KEY = ""
> ALGORITHM = "HS256"
> ACESS_TOKEN_EXPIRE_MINUTE = 120
> API_PORT = 8000
> 
> DB_AUTH_STORAGE_PATH = "/... .../input/db_auth"
> API_USER_EMAIL = ""
> API_USER_PASSWORD = ""
>
> STORAGE_PATH = "/... ..../storage"
> PROD_MODEL_NAME = "mlops_cnn_vit_model_weights"

### Authentication
Authentication file must be name `authentication.csv` under `DB_AUTH_STORAGE_PATH` with the same format as `/api/authentication_tpl.csv`. Password have to be generate with `get_password_hashed`from `/pi/authentication.py`.

The authentication scheme used is `OAuth2PasswordBearer`.

### Storage
Storage repository hierarchy:
* input
    * data: png x-Ray for test
    * db: file x-Ray data base for retraining
    * model: weights of the model use in production
* output
    * learning: directory for learning output saving
    * inference: saving inference history results
    * grad_cam: directory to save grad_cam images to return png file result.

### Launch API

> cd api/
> uvicorn main:api --reload

* Api entry point: 127.0.0.1:{API_PORT},
* Api OpenAPI entry point: 127.0.0.1:{API_PORT}/docs

### Re-training
Use `retraining(storage_path=None, db_storage_path=None)`in `/lib/cnn_vit/retraining.py`, with `storage_path` path to storage repository hierarchy, and `db_storage_path` path to learning db repository (by default is `{storage_path}/input/db`).

