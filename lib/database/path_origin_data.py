import os
import pandas as pd

# database names
lung_name = 'lung'
infection_name = 'infection'

# dataset names
train_name = 'train'
test_name = 'test'
valid_name = 'valid'

# label / desease names
no_covid_name = 'no_covid'
covid_name = 'covid19'
normal_name = 'normal'

# data types names
images_name = 'images'
masks_name = 'masks'
lung_mask_name = lung_name + '_' + masks_name
infection_mask_name = infection_name + '_' + masks_name

# database directory names
lung_dirname = 'Lung Segmentation Data'
infection_dirname = 'Infection Segmentation Data'

# data set names
train_dirname = 'Train'
test_dirname = 'Test'
valid_dirname = 'Val'

# label / desease names
no_covid_dirname = 'Non-COVID'
covid_dirname = 'COVID-19'
normal_dirname = 'Normal'

# data type names
images_dirname = 'images'
lung_masks_dirname = 'lung masks'
infection_masks_dirname = 'infection masks'


# build all the path available in databases
def build_data_paths():
    """
    output: data_path dataFrame
        * with index= ['db_name', 'ds_name', 'desease_name', 'data_type_name']
          for database name, dataset name, desease or label name and data type name
        * path: folders hierarchical string path

    access to the desire paths with:
    idx = pd.IndexSlice
    work_index = idx[db_name_idx, ds_name_idx, desease_name_idx, data_type_name_idx]
    work_paths = data_paths['path'].loc[work_index]

        * db_name_idx values: lung_name and infection_name
        * ds_name_idx values: train_name, valid_name,and test_name
        * desease_name_iidx values: normal_name, covid_name and no_covid_name
        * data_type_name_idx values: images_name, lung_mask_name and infection_mask_name
    
    example:
    work_index = idx[[lung_name, infection_name],:,:,[images_name]]
    and
    work_index = idx[:,:,:,[images_name]]
    Give the indexes to access all the repertories in data_paths for:
        * lung and infection databases,
        * train valid and test dataset
        * normal, covid, n-covid labels / deseases
        * images repertories
    """

    # dataframe structure to link code names and real names
    db_names = pd.DataFrame(data={
        'db_name': [lung_name, infection_name],
        'db_dirname': [lung_dirname, infection_dirname]
    })
    ds_names = pd.DataFrame(data={
        'ds_name': [train_name, test_name, valid_name],
        'ds_dirname': [train_dirname, test_dirname, valid_dirname]
    })
    desease_names = pd.DataFrame(data={
        'desease_name': [no_covid_name, covid_name, normal_name],
        'desease_dirname': [no_covid_dirname, covid_dirname, normal_dirname]
    })
    data_type_names = pd.DataFrame(data={
        'data_type_name': [images_name, lung_mask_name, infection_mask_name],
        'data_type_dirname': [images_dirname, lung_masks_dirname, infection_masks_dirname]
    })

    # add keys for merging dataframes
    db_names['keys'] = 1
    ds_names['keys'] = 1
    desease_names['keys'] = 1
    data_type_names['keys'] = 1

    # merging
    data_paths = pd.merge(db_names, ds_names, on='keys')
    data_paths = pd.merge(data_paths, desease_names, on='keys')
    data_paths = pd.merge(data_paths, data_type_names, on='keys')
    data_paths = data_paths.drop(columns=['keys'])

    # build data_paths
    data_paths = data_paths.set_index(
        ['db_name', 'ds_name', 'desease_name', 'data_type_name'])
    data_paths['path'] = data_paths.apply(
        lambda x: os.path.join(*(x.to_list())), axis=1)
    data_paths = data_paths.iloc[:, -1:]

    # drop unpresent infection mask under lung data base
    idx = pd.IndexSlice
    index = data_paths.loc[idx[[lung_name], :, :, infection_mask_name]].index
    data_paths = data_paths.drop(index)

    return data_paths


