import os
from common.init_paths import *

batch_size = 32

image_size = 256
input_shape = (image_size, image_size, 1)

learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 100

label_smoothing = 0.1
lam_recon = 10.
patience = 5
min_delta = 0.005
min_delta_fine_tuning = 0.0005

MODEL_NAME = os.getenv("PROD_MODEL_NAME")


def manage_var(storage_path=None, db_storage_path=None, model_name=None):
    this_dir = os.path.dirname(__file__)
    storage_path = storage_path or os.path.join(this_dir, '../../storage')
    init_paths(storage_path, db_storage_path)

    model_name = model_name or MODEL_NAME

    return model_name
