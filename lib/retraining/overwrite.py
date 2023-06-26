import os
import shutil
from retraining.config import *

def build_path_model(storage_path=None, db_storage_path=None, prod_model_name=None, new_model_name=None):
    prod_model_name = manage_var(storage_path, db_storage_path, prod_model_name)

    # build model
    prod_model_full_path = os.path.join(PATHS['model_path'], prod_model_name + "_weights.hdf5")
    new_model_full_path = os.path.join(PATHS['ckpt_path'], new_model_name + "_weights.hdf5")
    new_model_save_full_path = os.path.join(PATHS["model_path"], new_model_name + "_weights.hdf5")
    print(prod_model_full_path)
    print(new_model_full_path)
    print(new_model_save_full_path)
    return (prod_model_full_path, new_model_full_path, new_model_save_full_path)

def overwrite_prod_model(storage_path=None, db_storage_path=None, prod_model_name=None, new_model_name=None):

    prod_model_full_path, new_model_full_path, new_model_save_full_path = build_path_model(storage_path, db_storage_path, prod_model_name, new_model_name)
    shutil.copyfile(new_model_full_path, new_model_save_full_path)
    shutil.copyfile(new_model_full_path, prod_model_full_path)

    return
