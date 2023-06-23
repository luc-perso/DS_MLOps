import os
import shutil
import pandas as pd
import joblib
import datetime
from retraining.config import *
from database.path_origin_data import build_data_paths
from database.dataset import build_dataset_from_db_repo
from cnn_vit.cnn_vit_config import *
from cnn_vit.cnn_vit import build_model
from run_exp.standard import run_experiment
from run_exp.test import compile_test_model


def retraining_test(storage_path=None, db_storage_path=None, model_name=None):
    model_name = manage_var(storage_path, db_storage_path, model_name)

    model_full_path = os.path.join(PATHS["model_path"], model_name + "_weights.hdf5")

    current_date = datetime.date.today()
    retrained_model_name = str(current_date)
    checkpoint_filename = os.path.join(PATHS["ckpt_path"], retrained_model_name + '_weights.hdf5')

    print(model_full_path)
    print(checkpoint_filename)
    shutil.copyfile(model_full_path, checkpoint_filename)

    return retrained_model_name


def retraining(storage_path=None, db_storage_path=None, model_name=None):
    model_name = manage_var(storage_path, db_storage_path, model_name)

    res = False

    # build dataset
    data_paths = build_data_paths()
    ds_train, ds_test, ds_valid = build_dataset_from_db_repo(PATHS["db_path"], data_paths['path'])
    print(ds_train.cardinality().numpy())

    # build model
    model = build_model(image_size)
    model_full_path = os.path.join(PATHS["model_path"], model_name + "_weights.hdf5")
    model.load_weights(model_full_path)

    # estimate model scores on new dataset
    _, _, _, _, report = compile_test_model(
        model,
        ds_test, batch_size,
        from_logits=False, label_smoothing=label_smoothing
    )
    # save report
    report
    print(report)
    f_name = os.path.join(PATHS["metric_path"], model_name + '_report.joblib')
    joblib.dump(report, f_name)
    macro_f1 = report['macro avg']['f1-score']

    # retraining
    current_date = datetime.date.today()
    retrained_model_name = str(current_date)
    history = run_experiment(
        model,
        ds_train, ds_valid, ds_test,
        batch_size=batch_size, num_epochs=num_epochs,
        learning_rate=learning_rate / 2., weight_decay=weight_decay,
        from_logits=False, label_smoothing=label_smoothing,
        patience=patience, min_delta=min_delta,
        log_path=PATHS["log_path"], ckpt_path=PATHS["ckpt_path"],
        prefix=retrained_model_name
    )

    # save history
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history)
    # or save to csv: 
    hist_csv_file = os.path.join(PATHS["learning_path"], retrained_model_name + '_history.csv')
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # reload best retrained models
    checkpoint_filename = os.path.join(PATHS["ckpt_path"], retrained_model_name + '_weights.hdf5')
    model.load_weights(checkpoint_filename)

    # estimate best model score
    _, _, _, _, report = compile_test_model(
        model,
        ds_test, batch_size,
        from_logits=False, label_smoothing=label_smoothing
    )
    # save report
    print(report)
    f_name = os.path.join(PATHS["metric_path"], retrained_model_name + '_report.joblib')
    joblib.dump(report, f_name)
    retrained_macro_f1 = report['macro avg']['f1-score']

    # replace operationnal model if new one is better
    if retrained_macro_f1 > macro_f1:
        model_filename = os.path.join(PATHS["model_path"], retrained_model_name + "_weights.hdf5")
        shutil.copyfile(checkpoint_filename, model_filename)
        shutil.copyfile(checkpoint_filename, model_full_path)
        res = True
    
    return res








