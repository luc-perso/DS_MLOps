import _mypath
from fastapi.testclient import TestClient
from fastapi.security import OAuth2PasswordRequestForm
import numpy as np
import cv2

from config import *
from main import api
from authentication import *
from common.init_paths import *
from dotenv import load_dotenv

load_dotenv()

API_USER_EMAIL = os.getenv('API_USER_EMAIL')
API_USER_PASSWORD = os.getenv('API_USER_PASSWORD')
this_dir = os.path.dirname(__file__)
STORAGE_PATH = os.getenv("STORAGE_PATH") or os.path.join(this_dir, '..', 'storage')
init_paths(STORAGE_PATH)

client = TestClient(api)

bearer_token = client.post("/token", 
                           headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                           data={
                               "grant_type": "",
                               "username": API_USER_EMAIL,
                               "password": API_USER_PASSWORD,
                               "scope": "",
                               "client_id": "",
                               "client_secret": "",
                               }
).json()
auth_header = {
    "accept": "application/json",
    "Authorization": f'{bearer_token["token_type"]} {bearer_token["access_token"]}'
}


# def test_token():
#     print(bearer_token)


def test_user_me():
    response = client.get("/user/me", headers=auth_header)
    assert response.status_code == 200
    user = response.json()
    del user["id"]
    assert  user == {
                    "email": API_USER_EMAIL,
                    "role": "admin",
                    "disabled": False
                }


def test_entry():
    response = client.get("/", headers=auth_header)
    assert response.status_code == 200
    assert response.text == '"The API is available."'


def test_image_classif():
    filename = os.path.join(PATHS["data_path"], "covid_1.png")
    with open(filename, "rb") as f:
        response = client.post(
            "/x-ray",
            files={"file": ("file", f, "image/png")},
            headers=auth_header
        )
    assert response.status_code == 200
    data = response.json()
    assert "covid" in data.keys()
    assert "no-covid" in data.keys()
    assert "normal" in data.keys()


def test_image_grad_cam():
    filename = os.path.join(PATHS["data_path"], "covid_1.png")
    with open(filename, "rb") as f:
        response = client.post(
            "/grad_cam",
            files={"file": ("file", f, "image/png")},
            headers=auth_header
        )
    assert response.status_code == 200
    contents = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(contents, 0)
    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE)


