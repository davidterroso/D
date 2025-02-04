from os import walk
from random import shuffle
from sklearn.model_selection import KFold
from pandas import DataFrame, concat
import numpy as np

# Volumes from Topcon T-1000: 
# Training - 056, 062
# Testing - None

def k_fold_split_segmentation(k=5, folders_path=""):
    """
    k_fold_split_segmentation iterates through the directories of the RETOUCH training
    dataset and extracts the volumes that belong to each vendor and splits
    in k folds according to k value specified in the arguments

    Args:
        k (int): number of k folds
        folders_path (string): absolute or relative path to the RETOUCH dataset 
            location

    Return: 
        None
    """
    # Extracts the volumes available for training and appends it to a dictionary
    dictionary = {}
    exceptions_list = []
    for (root, _, _) in walk(folders_path):
        train_or_test = root.split("-")
        if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
            vendor_volume = train_or_test[2].split("""\\""")
            if ((len(vendor_volume) == 1) and (vendor_volume[0] not in dictionary)):
                dictionary[vendor_volume[0]] = []
            elif len(vendor_volume) == 2:
                vendor = vendor_volume[0]
                volume = vendor_volume[1][-3:]
                # Restriction implemented to make sure the volumes 056 and 062,
                # from Topcon, which are not the same size as the others, do not
                # stay on the same fold, promoting class balancement
                if (volume != "056") and (volume != "062"):
                    dictionary[vendor].append(volume)
                else:
                    exceptions_list.append(volume)
    
    # Initiates KFold object from sklearn that splits the
    # dataset with provided k value
    kf = KFold(n_splits=k)
    all_train_folds = []
    all_test_folds = []
    shuffle(exceptions_list)
    for key in dictionary:
        # Shuffles the list of volumes available
        tmp_list = dictionary[key]
        shuffle(tmp_list)
        dictionary[key] = tmp_list
        # Splits in train and test according to the number of folds
        vendor_train_folds = []
        vendor_test_folds = []
        for i, (train_index, test_index) in enumerate(kf.split(dictionary[key])):
            train_volumes = np.array(dictionary[key])[train_index]
            test_volumes = np.array(dictionary[key])[test_index]
            # Restriction implemented to make sure the volumes 056 and 062,
            # from Topcon, which are not the same size as the others, do not
            # stay on the same fold, promoting class balancement
            if key == "Topcon":
                if i != 1:
                    train_volumes = np.append(train_volumes, exceptions_list[0])
                    np.random.shuffle(train_volumes)
                else:
                    test_volumes = np.append(test_volumes, exceptions_list[0])
                    np.random.shuffle(test_volumes)
                if i != 2:
                    train_volumes = np.append(train_volumes, exceptions_list[1])
                    np.random.shuffle(train_volumes)
                else:
                    test_volumes = np.append(test_volumes, exceptions_list[1])
                    np.random.shuffle(test_volumes)
            vendor_train_folds.append(train_volumes)
            vendor_test_folds.append(test_volumes)
        all_train_folds.append(vendor_train_folds)
        all_test_folds.append(vendor_test_folds)

    # Joins all the volumes from the same fold of different vendors in one list
    train_folds = []
    test_folds = []
    for j in range(k):
        tmp_list_train = []
        tmp_list_test = []
        for l in range(len(dictionary.keys())):
            tmp_list_train = tmp_list_train + all_train_folds[l][j].tolist()
            tmp_list_test = tmp_list_test + all_test_folds[l][j].tolist()
        shuffle(tmp_list_train)
        shuffle(tmp_list_test)
        train_folds.append(tmp_list_train)
        test_folds.append(tmp_list_test)

    # Iterates through the train-test lists and saves each as an individual 
    # column in a Pandas Dataframe
    train_df = DataFrame()
    for l in range(k):
        tmp_df = DataFrame()
        tmp_vols = []
        name_column_vol = f"Fold{l + 1}_Volumes" 
        for m in range(len(train_folds[l])):
            tmp_vols.append(train_folds[l][m])
        if l == 0:
            train_df[name_column_vol] = tmp_vols
        else:
            tmp_df[name_column_vol] = tmp_vols
        train_df = concat([train_df, tmp_df], axis=1, sort=False)

        # Saves the results from the split in a CSV file just for the train
        train_df.to_csv(path_or_buf="./splits/segmentation_train_splits.csv", index=False)

    # Iterates through the train-test lists and saves each as an individual 
    # column in a Pandas Dataframe
    test_df = DataFrame()
    for l in range(k):
        tmp_df = DataFrame()
        tmp_vols = []
        name_column_vol = f"Fold{l + 1}_Volumes" 
        for m in range(len(test_folds[l])):
            tmp_vols.append(test_folds[l][m])
        if l == 0:
            test_df[name_column_vol] = tmp_vols
        else:
            tmp_df[name_column_vol] = tmp_vols
        test_df = concat([test_df, tmp_df], axis=1, sort=False)

        # Saves the results from the split in a CSV file just for the train
        test_df.to_csv(path_or_buf="./splits/segmentation_test_splits.csv", index=False)

def k_fold_split_generation(k=5, folders_path=""):
    """
    k_fold_split_generation iterates through the directories of the RETOUCH 
    training and testing folders and extracts the volumes that belong to each
    vendor and splits in k folds according to k value specified in the arguments

    Args:
        k (int): number of k folds
        folders_path (string): absolute or relative path to the RETOUCH dataset 
            location

    Return: 
        None
    """

    # Extracts the volumes available for the generation task and appends it
    # to a dictionary
    dictionary = {}
    for (root, _, _) in walk(folders_path):
        train_or_test = root.split("-")
        if (len(train_or_test) == 3):
            if (train_or_test[1] == "TrainingSet"):
                belonging_set = "train"
            elif (train_or_test[1] == "TestSet"):
                belonging_set = "test"
            vendor_volume = train_or_test[2].split("""\\""")
            if ((len(vendor_volume) == 1) and (vendor_volume[0] not in dictionary)):
                dictionary[vendor_volume[0]] = []
            elif len(vendor_volume) == 2:
                vendor = vendor_volume[0]
                volume = vendor_volume[1][-3:]
                volume_set = (volume, belonging_set)
                dictionary[vendor].append(volume_set)
    
    # Initiates KFold object from sklearn that splits the
    # dataset with provided k value
    kf = KFold(n_splits=k)
    all_train_folds = []
    all_test_folds = []
    for key in dictionary:
        # Shuffles the list of volumes available
        tmp_list = dictionary[key]
        shuffle(tmp_list)
        dictionary[key] = tmp_list
        # Splits in train and test according to the number of folds
        vendor_train_folds = []
        vendor_test_folds = []
        for (train_index, test_index) in (kf.split(dictionary[key])):
            train_volumes = np.array(dictionary[key])[train_index]
            test_volumes = np.array(dictionary[key])[test_index]
            vendor_train_folds.append(train_volumes)
            vendor_test_folds.append(test_volumes)
        all_train_folds.append(vendor_train_folds)
        all_test_folds.append(vendor_test_folds)

    # Joins all the volumes from the same fold of different vendors in one list
    train_folds = []
    test_folds = []
    for j in range(k):
        tmp_list_train = []
        tmp_list_test = []
        for l in range(len(dictionary.keys())):
            tmp_list_train = tmp_list_train + all_train_folds[l][j].tolist()
            tmp_list_test = tmp_list_test + all_test_folds[l][j].tolist()
        shuffle(tmp_list_train)
        shuffle(tmp_list_test)
        train_folds.append(tmp_list_train)
        test_folds.append(tmp_list_test)

    # Iterates through the train-test lists and saves each as an individual 
    # column in a Pandas Dataframe
    train_df = DataFrame()
    for l in range(k):
        tmp_df = DataFrame()
        tmp_vols = []
        tmp_ts_sets = []
        name_column_vol = f"Fold{l + 1}_Volumes" 
        name_column_sets = f"Fold{l + 1}_Sets"
        for m in range(len(train_folds[l])):
            tmp_vols.append(train_folds[l][m][0])
            tmp_ts_sets.append(train_folds[l][m][1])
        if l == 0:
            train_df[name_column_vol] = tmp_vols
            train_df[name_column_sets] = tmp_ts_sets
        else:
            tmp_df[name_column_vol] = tmp_vols
            tmp_df[name_column_sets] = tmp_ts_sets
        train_df = concat([train_df, tmp_df], axis=1, sort=False)

        # Saves the results from the split in a CSV file just for the train
        train_df.to_csv(path_or_buf="./splits/generation_train_splits.csv", index=False)

    test_df = DataFrame()
    for l in range(k):
        tmp_df = DataFrame()
        tmp_vols = []
        tmp_ts_sets = []
        name_column_vol = f"Fold{l + 1}_Volumes" 
        name_column_sets = f"Fold{l + 1}_Sets"
        for m in range(len(test_folds[l])):
            tmp_vols.append(test_folds[l][m][0])
            tmp_ts_sets.append(test_folds[l][m][1])
        if l == 0:
            test_df[name_column_vol] = tmp_vols
            test_df[name_column_sets] = tmp_ts_sets
        else:
            tmp_df[name_column_vol] = tmp_vols
            tmp_df[name_column_sets] = tmp_ts_sets
        test_df = concat([test_df, tmp_df], axis=1, sort=False)

        # Saves the results from the split in a CSV file just for the test
        test_df.to_csv(path_or_buf="./splits/generation_test_splits.csv", index=False)
        