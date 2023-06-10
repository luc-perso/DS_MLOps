import _mypath
import os
from os.path import join, dirname

from config import *
from common.init_paths import *
from cnn_vit.cnn_vit import build_model, build_grad_cam
from visu.grad_cam import grad_cam

from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

import cv2
import uuid

import numpy as np
import pandas as pd
from authentication import *

import uvicorn
from dotenv import load_dotenv


load_dotenv()
ACESS_TOKEN_EXPIRE_MINUTE = int(os.getenv('ACESS_TOKEN_EXPIRE_MINUTE')) 

API_PORT = int(os.getenv('API_PORT'))

this_dir = os.path.dirname(__file__)
STORAGE_PATH = os.getenv("STORAGE_PATH") or os.path.join(this_dir, '..', 'storage')
init_paths(STORAGE_PATH)
# print(PATHS)

model = build_model(IMAGE_SIZE)
MODEL_NAME = os.getenv("PROD_MODEL_NAME")
model_full_path = os.path.join(PATHS["model_path"], MODEL_NAME + ".hdf5")
model.load_weights(model_full_path)

grad_model = build_grad_cam(model)


class AskForm(BaseModel):
    use: str
    subjects: list[str]


api = FastAPI(
    title="Datascientist FastAPI for MLOps",
    description="API powered by FastAPI.",
    version="1.0.0")


@api.get('/')
async def is_launched():
    """Return text if available"""

    return 'The API is available.'


@api.post('/token', response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """Authenticate and get a time limited jwt token """

    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password", headers={"WW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACESS_TOKEN_EXPIRE_MINUTE)
    access_token = create_access_taken(data={"sub": user.email}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "Bearer"}


@api.get('/user/me', response_model=User)
async def read_user_me(current_user: CurrentUser):
    """Get your own profil information"""

    return current_user


@api.post('/x-ray')
async def inference(current_user: CurrentUser, file: UploadFile = File(...)):
    """Classify xRay: O normal, 1 covid, 2 no-covid"""

    if file.content_type != "image/png":
        raise HTTPException(400, detail="Invalid image type")
    
    # upload image
    image = await upload(file)
    image_rect = format_image(image)

    # inference
    proba = model.predict(np.array([image_rect]))[0]

    # save inference results
    inference_full_path = os.path.join(PATHS["inference_path"], "history.csv")
    data = [current_user.id, *proba]
    data_frame = pd.DataFrame(
        [data],
        columns=["id", "covid", "no-covid", "normal"]
    )
    mode = 'w'
    header = True
    if os.path.isfile(inference_full_path):
        mode = 'a'
        header = False
    data_frame.to_csv(inference_full_path, mode=mode, index=False, header=header, sep=';')

    return {
        "covid": str(proba[0]),
        "no-covid": str(proba[1]),
        "normal": str(proba[2]),
    }


@api.post('/grad_cam')
async def grad_cam_image(current_user: CurrentUser, file: UploadFile = File(...)):
    """Build Grad Cam Visualisation"""

    if file.content_type != "image/png":
        raise HTTPException(400, detail="Invalid image type")
    
    # upload image
    image = await upload(file)
    image_rect = format_image(image)

    # grad cam image
    cam = grad_cam(image_rect, grad_model)

    name = uuid.uuid4()
    grad_cam_name = f"{name}.png"
    grad_cam_full_path = os.path.join(PATHS["grad_cam_path"], grad_cam_name)
    cv2.imwrite(grad_cam_full_path, cam)

    return FileResponse(grad_cam_full_path, media_type="image/png")


async def upload(file: UploadFile):
    contents = await file.read()
    contents = np.asarray(bytearray(contents), dtype="uint8")
    image = cv2.imdecode(contents, 0)

    return image


def format_image(image):
    # checks
    image_shape = image.shape
    # checks image is a squared one
    if image_shape[0] != image_shape[1]:
        new_size = np.min(image_shape[0], image_shape[1])
        start_0 = (image_shape[0] - new_size) // 2
        stop_0 = start_0 + new_size
        start_1 = (image_shape[1] - new_size) // 2
        stop_1 = start_1 + new_size

        image = image[start_0:stop_0, start_1:stop_1]
    # check size
    if image_shape[0] != IMAGE_SIZE or image_shape[1] != IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # equalize image
    image_rect = cv2.equalizeHist(image)
    return image_rect


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=API_PORT)
    