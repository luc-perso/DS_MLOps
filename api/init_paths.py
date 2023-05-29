import os

paths = {
    "input_path": "",
    "data_path": "",
    "model_path": "",
    "output_path": "",
    "log_path": "",
    "ckpt_path": "",
    "metric_path": "",
    "grad_cam_path": "",
    "upload_path": "",
}


def init_paths(work_dir):
    # input
    # print(paths)
    paths["input_path"] = os.path.join(work_dir, 'input')

    data_name = 'data'
    paths["data_path"] = os.path.join(paths["input_path"], data_name)

    model_name = 'model'
    paths["model_path"] = os.path.join(paths["input_path"], model_name)

    paths["upload_path"] = os.path.join('..', 'upload')
    if not os.path.exists(paths["upload_path"]):
        os.makedirs(paths["upload_path"], exist_ok=True)

    # output
    paths["output_path"] = os.path.join('..', 'output', 'learning')
    if not os.path.exists(paths["output_path"]):
        os.makedirs(paths["output_path"], exist_ok=True)

    paths["log_path"] = os.path.join(paths["output_path"], 'log')
    if not os.path.exists(paths["log_path"]):
        os.makedirs(paths["log_path"], exist_ok=True)

    paths["ckpt_path"] = os.path.join(paths["output_path"], 'ckpt')
    if not os.path.exists(paths["ckpt_path"]):
        os.makedirs(paths["ckpt_path"], exist_ok=True)

    paths["metric_path"] = os.path.join(paths["output_path"], 'metric')
    if not os.path.exists(paths["metric_path"]):
        os.makedirs(paths["metric_path"], exist_ok=True)

    paths["grad_cam_path"] = os.path.join(paths["output_path"], 'grad_cam')
    if not os.path.exists(paths["grad_cam_path"]):
        os.makedirs(paths["grad_cam_path"], exist_ok=True)
