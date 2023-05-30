import os
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import pandas as pd

SECRET_KEY = "Datascientest_FastAPI"
ALGORITHM = "HS256"
ACESS_TOKEN_EXPIRE_MINUTE = 120

DB_STORAGE_PATH = os.environ.get("DB_STORAGE_PATH") or os.path.dirname(__file__)
db_path = os.path.join(DB_STORAGE_PATH, "db")
db_fullname = os.path.join(db_path, "authentication.csv")
db = pd.read_csv(db_fullname, sep=';', header=0).to_dict('records')


class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    email: str
    role: str
    disabled: bool or None = None

class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hashed(password):
    return pwd_context.hash(password)

# def init_db():
#     db = [
#     {"id": 0, "email": "alice@gmail.com", "role": "user", "hashed_password": get_password_hashed("wonderland"), "disabled": False},
#     {"id": 1, "email": "bob@gmail.com", "role": "user", "hashed_password": get_password_hashed("builder"), "disabled": False},
#     {"id": 2, "email": "clementine@gmail.com", "role": "user", "hashed_password": get_password_hashed("mandarine"), "disabled": False},
#     {"id": 3, "email": "admin", "role": "admin", "hashed_password": get_password_hashed("4dm1N"), "disabled": False},
#     ]
#     db_frame = pd.DataFrame.from_records(db)
#     db_frame.to_csv(db_fullname, sep=';', header=True)
# init_db()

def get_user(db, email: str):
    for doc in db:
        if email == doc["email"]:
            # print(email)
            return UserInDB(**doc)

def authenticate_user(db, email: str, password: str):
    user = get_user(db, email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    
    return user

def create_access_taken(data: dict, expires_delta: timedelta or None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                         detail="Could not validate credentials", headers={"WW-Authenticate": "Bearer"})
    
    try:
      payload = jwt.decode(token, SECRET_KEY, algorithms=ALGORITHM)
      email: str = payload.get("sub")
      if email is None:
          raise credential_exception
    except JWTError:
        raise credential_exception
    
    user = get_user(db, email=email)
    if user is None:
        raise credential_exception
    
    return user

async def get_current_active_user(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return current_user
    
async def get_current_active_admin(current_user: Annotated[UserInDB, Depends(get_current_active_user)]):
    if current_user.role != "admin":
        raise HTTPException(status_code=400, detail="End point for admin use only.")
    
    return current_user

CurrentUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, Depends(get_current_active_admin)]

