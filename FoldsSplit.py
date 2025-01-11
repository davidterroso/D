from os import walk
from random import shuffle
from sklearn.model_selection import KFold
import numpy as np
from csv import writer

def k_fold_split_segmentation(k=5, folders_path=""):
    """
    k_fold_split iterates through the directories of the RETOUCH training
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
            if len(vendor_volume) == 1:
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
            # print("Fold:", i + 1)
            train_volumes = np.array(dictionary[key])[train_index]
            # print("{} Train: {}".format(key, train_volumes))
            test_volumes = np.array(dictionary[key])[test_index]
            # print("{} Test: {}".format(key, test_volumes))
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

    # Saves the results from the split in a CSV file just for the train
    with open("./outputs/train_splits.csv", "w", newline="") as f:
        write = writer(f)
        write.writerows(train_folds)    
    
    # Saves the results from the split in a CSV file just for the test
    with open("./outputs/test_splits.csv", "w", newline="") as f:
        write = writer(f)
        write.writerows(test_folds)

if __name__ == "__main__":
    # Declares the path and the k-value, and calls the function
    RETOUCH_path = "D:\RETOUCH"
    k = 5
    k_fold_split_segmentation(k=k, folders_path=RETOUCH_path)
    print("EOF.")