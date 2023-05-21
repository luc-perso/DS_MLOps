import os
import pandas as pd
import tensorflow as tf

from database.path_origin_data import lung_name, infection_name
from database.path_origin_data import train_name, test_name, valid_name
from database.path_origin_data import normal_name, covid_name, no_covid_name
from database.path_origin_data import images_name, lung_mask_name, infection_mask_name


def build_dataset_base(db_path, data_paths, label=0,
                       color_mode='grayscale',
                       image_size=(256, 256),
                       shuffle=True,
                       seed=123):
    def make_label(img):
        return img, label

    dataset = None
    file_paths = None
    for path in data_paths:
        full_file_path = os.path.join(db_path, path)
        # print(full_file_path)

        data_tmp = tf.keras.utils.image_dataset_from_directory(
            directory=full_file_path,
            labels=None,
            label_mode=None,
            color_mode=color_mode,
            batch_size=None,
            image_size=image_size,
            shuffle=shuffle,
            seed=seed
        )

        file_paths_tmp = data_tmp.file_paths

        dataset_tmp = data_tmp.map(make_label)
        if dataset is None:
            dataset = dataset_tmp
            file_paths = file_paths_tmp
        else:
            dataset = dataset.concatenate(dataset_tmp)
            file_paths.extend(file_paths_tmp)
        
        # print(file_paths[-1])

    return dataset, file_paths


# build dataset from data_path dataframe
def build_dataset(db_path, data_paths,
                  db=[lung_name], ds=[train_name], data_type=[images_name],
                  **kwargs):

    idx = pd.IndexSlice

    paths = data_paths.loc[idx[db, ds, normal_name, data_type]]
    dataset, file_paths = build_dataset_base(db_path, paths, label=[1, 0, 0], **kwargs)

    paths = data_paths.loc[idx[db, ds, covid_name, data_type]]
    ds_tmp, file_paths_tmp = build_dataset_base(db_path, paths, label=[0, 1, 0], **kwargs)
    dataset = dataset.concatenate(ds_tmp)
    file_paths.extend(file_paths_tmp)

    paths = data_paths.loc[idx[db, ds, no_covid_name, data_type]]
    ds_tmp, file_paths_tmp = build_dataset_base(db_path, paths, label=[0, 0, 1], **kwargs)
    dataset = dataset.concatenate(ds_tmp)
    file_paths.extend(file_paths_tmp)

    return dataset, file_paths
