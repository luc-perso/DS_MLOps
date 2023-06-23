import os

PATHS = {
    "input_path": "",
    "data_path": "",
    "db_path": "",
    "model_path": "",
    "output_path": "",
    "inference_path" : "",
    "learning_path": "",
    "log_path": "",
    "ckpt_path": "",
    "metric_path": "",
    "grad_cam_path": "",
    # "upload_path": "",
}


def init_paths(storage_path, db_storage_path=None):
    # input
    PATHS["input_path"] = os.path.join(storage_path, 'input')

    data_name = 'data'
    PATHS["data_path"] = os.path.join(PATHS["input_path"], data_name)

    if db_storage_path is None:
        db_name = 'db'
        PATHS["db_path"] = os.path.join(PATHS["input_path"], db_name)
    else:
        PATHS["db_path"] = db_storage_path

    model_name = 'model'
    PATHS["model_path"] = os.path.join(PATHS["input_path"], model_name)

    # PATHS["upload_path"] = os.path.join(storage_dir, 'upload')
    # if not os.path.exists(PATHS["upload_path"]):
    #     os.makedirs(PATHS["upload_path"], exist_ok=True)

    # output
    PATHS["output_path"] = os.path.join(storage_path, 'output')
    print(PATHS["output_path"])
    if not os.path.exists(PATHS["output_path"]):
        os.makedirs(PATHS["output_path"], exist_ok=True)

    PATHS["inference_path"] = os.path.join(PATHS["output_path"], 'inference')
    print(PATHS["inference_path"])
    if not os.path.exists(PATHS["inference_path"]):
        os.makedirs(PATHS["inference_path"], exist_ok=True)

    PATHS["learning_path"] = os.path.join(PATHS["output_path"], 'learning')
    print(PATHS["learning_path"])
    if not os.path.exists(PATHS["learning_path"]):
        os.makedirs(PATHS["learning_path"], exist_ok=True)

    PATHS["log_path"] = os.path.join(PATHS["learning_path"], 'log')
    if not os.path.exists(PATHS["log_path"]):
        os.makedirs(PATHS["log_path"], exist_ok=True)

    PATHS["ckpt_path"] = os.path.join(PATHS["learning_path"], 'ckpt')
    if not os.path.exists(PATHS["ckpt_path"]):
        os.makedirs(PATHS["ckpt_path"], exist_ok=True)

    PATHS["metric_path"] = os.path.join(PATHS["learning_path"], 'metric')
    if not os.path.exists(PATHS["metric_path"]):
        os.makedirs(PATHS["metric_path"], exist_ok=True)

    PATHS["grad_cam_path"] = os.path.join(PATHS["learning_path"], 'grad_cam')
    if not os.path.exists(PATHS["grad_cam_path"]):
        os.makedirs(PATHS["grad_cam_path"], exist_ok=True)
