import os
import joblib
from retraining.config import *
from database.path_origin_data import build_data_paths
from database.dataset import build_dataset_from_db_repo
from cnn_vit.cnn_vit_config import *
from cnn_vit.cnn_vit import build_model
from run_exp.test import compile_test_model


def test_model_test(storage_path=None, db_storage_path=None, model_name=None, rep='model_path'):
    macro_f1 = 0.86 if (rep == 'model_path') else 0.98

    return macro_f1


def test_model(storage_path=None, db_storage_path=None, model_name=None, rep='model_path'):
    model_name = manage_var(storage_path, db_storage_path, model_name)

    # build dataset
    data_paths = build_data_paths()
    ds_train, ds_test, ds_valid = build_dataset_from_db_repo(PATHS["db_path"], data_paths['path'])
    print(ds_test.cardinality().numpy())

    # build model
    model = build_model(image_size)
    model_full_path = os.path.join(PATHS[rep], model_name + "_weights.hdf5")
    print(model_full_path)
    model.load_weights(model_full_path)

    # estimate model scores on new dataset
    _, _, _, _, report = compile_test_model(
        model,
        ds_test.take(100), batch_size,
        from_logits=False, label_smoothing=label_smoothing
    )
    # save report
    report
    print(report)
    f_name = os.path.join(PATHS["metric_path"], model_name + '_report.joblib')
    joblib.dump(report, f_name)
    macro_f1 = report['macro avg']['f1-score']
    
    return macro_f1
