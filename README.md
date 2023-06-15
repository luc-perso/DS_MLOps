# DS_MLOps

## Repository hierarchy
* api: api and test api code
* jupyter: jupyter notebook for development testing
* lib: python package code
    * cnn_vit: tensorFlow code of the model used
    * common: code used by several packages essentially paths management
    * database: paths image database management
    * myLayers: base layers of the model cnn_vitæ©
    * run_exp: tensor Flow code for training
    * visu: code to produce model insights
* storage: contains application inputs and outputs 

## Storage repository hierarchy
Storage repository hierarchy:
* input
    * data: png x-Ray for test
    * db: file x-Ray data base for retraining
    * db_auth: authentication.csv for api authentication
    * model: weight files of the model use in production and history of prvious ones
* output
    * learning: directory for learning output saving
    * inference: saving inference history results
    * grad_cam: directory to save grad_cam images to return png file result.


## Environement variables
Have to be define in /api/.env file or in local environement
> 
> \# for api and retraining\
> STORAGE_PATH = "/... ..../storage"\
> PROD_MODEL_NAME = "mlops_cnn_vit_model_weights"
>
> API_PORT = 8000\
> \# for API authentication\
> SECRET_KEY = "put the api key here"\
> ALGORITHM = "HS256"\
> ACESS_TOKEN_EXPIRE_MINUTE = 120\
> DB_AUTH_STORAGE_PATH = "${STORAGE_PATH}/input/db_auth"\
>
> \# for API authentication during test\
> API_USER_EMAIL = ""\
> API_USER_PASSWORD = ""\
>

## API

### Authentication
Authentication file must be name `authentication.csv` under `DB_AUTH_STORAGE_PATH` with the same format as `/api/authentication_tpl.csv`.

Password have to be generate with `get_password_hashed`from `/api/authentication.py`.

The authentication scheme used is `OAuth2PasswordBearer`.

### Launch api
>
> cd api/\
> uvicorn main:api --reload
>

### Entry points:
* Api : 127.0.0.1:${API_PORT}
* OpenAPI documentation: 127.0.0.1:${API_PORT}/docs

## Tests
Test are produce with PyTest, just launch PyTest in a terminal at the project root repository:
>
> cd root/project/repo/\
> PyTest
>

## Re-training
Use `retraining(storage_path=None, db_storage_path=None)`in `/lib/cnn_vit/retraining.py`:
* `storage_path` path to storage repository hierarchy,
* `db_storage_path` path to learning db repository (if None `${storage_path}/input/db` is used).

