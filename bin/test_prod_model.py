import _mypath
import os
from cnn_vit.retraining import test_model
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    STORAGE_PATH = os.getenv('STORAGE_PATH')
    DB_STORAGE_PATH = os.getenv('DB_STORAGE_PATH')
    PROD_MODEL_NAME = os.getenv('PROD_MODEL_NAME')

    print('hello world!')
    print(STORAGE_PATH)
    print(DB_STORAGE_PATH)
    print(PROD_MODEL_NAME)
    res = test_model(STORAGE_PATH, DB_STORAGE_PATH, PROD_MODEL_NAME, 'model_path')
    # return the results as stdout for airflow
    print(res)
    