import numpy as np
import pandas as pd
from itertools import permutations
from math import ceil, isnan
from os import walk
from random import shuffle
from sklearn.model_selection import KFold

# Volumes from Topcon T-1000 with 64 slices: 
# Training - 056, 062
# Testing - None
def random_k_fold_segmentation(k: int=5, folders_path: str=""):
    """
    random_k_fold_segmentation iterates through the directories of the RETOUCH training
    dataset and extracts the volumes that belong to each vendor and splits
    in k folds according to k value specified in the arguments

    Args:
        k (int): number of k folds
        folders_path (string): absolute or relative path to the RETOUCH dataset 
            location

    Return: 
        None
    """
    # Extracts the volumes available for training directly from the folders 
    # and appends it to a dictionary
    # Creates a dictionary whose keys will be the vendors and the values 
    # will be the number of the volumes
    dictionary = {}
    # Initializes a list that will contain the volumes that are exceptions
    exceptions_list = []
    # Iterates through the files of the RETOUCH dataset
    for (root, _, _) in walk(folders_path):
        train_or_test = root.split("-")
        if ((len(train_or_test) == 3) and (train_or_test[1] == "TrainingSet")):
            vendor_volume = train_or_test[2].split("""\\""")
            if ((len(vendor_volume) == 1) and (vendor_volume[0] not in dictionary)):
                dictionary[vendor_volume[0]] = []
            elif len(vendor_volume) == 2:
                # Gets the name of the vendor and 
                # the name of the volume from the file name
                vendor = vendor_volume[0]
                volume = vendor_volume[1][-3:]
                # Restriction implemented to make sure the volumes 056 and 062,
                # from Topcon, which are not the same size as the others, do not
                # stay on the same fold, promoting class balancement
                # Appends the volumes to the corresponding dictionary list and 
                # the exceptions to the specific list 
                if (volume != "056") and (volume != "062"):
                    dictionary[vendor].append(volume)
                else:
                    exceptions_list.append(volume)

    # Initializes a dictionary that will hold 
    # the list of the number of volumes per 
    # fold per vendor
    # Each key (vendor) will have a list of 
    # size k where at the fold index will 
    # have the number of volumes that will 
    # be present in the fold of same index 
    vols_per_fold_dict = {}
    # Iterates through all the vendors
    for vendor in dictionary.keys():
        # Calculates how many volumes of 
        # each vendor will be placed into a fold
        quotient, remainder = divmod(len(dictionary[vendor]), k)
        num_vols = [quotient] * k
        for i in range(remainder):
            num_vols[i] += 1
        # Appends the list of values to the 
        # dictionary key
        vols_per_fold_dict[vendor] = num_vols

    # Initializes a dictionary that will have a list of
    # volumes indexes associated to each fold (key)
    fold_vol_dict = {}
    # Iterates through the folds
    for fold in range(k):
        # In each fold initializes an empty list that 
        # will hold the indexes of the volumes
        fold_volumes = []
        # Iterates throgh all the possible vendors
        for vendor in dictionary.keys():
            # Shuffles the list that contains the 
            # volumes indices of the said vendor
            shuffle(dictionary[vendor])
            # Checks how many volumes are supposed to extract
            num_vols = vols_per_fold_dict[vendor][fold]
            # Extracts the calculated volumes
            vols = dictionary[vendor][-num_vols:]
            # Converts their indices from string to int
            for index, vol in enumerate(vols):
                vols[index] = int(vol)
            # Updates the dictionary, removing the extracted volumes
            dictionary[vendor] = dictionary[vendor][:-num_vols]
            # Appends the newly extracted volumes to the list that 
            # will hold the volumes for this vendor
            fold_volumes = fold_volumes + vols
        # Places in the key corresponding 
        # to the fold the indexes of the 
        # volumes that compose it 
        fold_vol_dict[fold] = fold_volumes
    # Handles the exceptional volumes by shuffling 
    # the list that contains their indexes and 
    # randomly assigning them to fold 1 and 2
    shuffle(exceptions_list)
    fold_vol_dict[0].append(int(exceptions_list[0]))
    fold_vol_dict[1].append(int(exceptions_list[1]))

    # Initiates the final DataFrame that will be saved
    final_df = pd.DataFrame()
    # Iterates through all the folds
    for fold, data in fold_vol_dict.items():
        # Converts the data in each fold to a DataFrame 
        # and joins them
        tmp_df = pd.DataFrame(columns=[fold], data=data)
        # The data is saved as int64 to handle the cases where there are NaN values, 
        # without converting the numerical values from integer to float 
        final_df = pd.concat([final_df, tmp_df], axis=1, sort=False).astype("Int64")
    # Savesthe file as CSV
    final_df.to_csv("..\\splits\\segmentation_fold_selection.csv", index=False)

def random_k_fold_generation(k: int=5, folders_path: str=""):
    """
    random_k_fold_generation iterates through the directories of the RETOUCH 
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
    train_df = pd.DataFrame()
    for l in range(k):
        tmp_df = pd.DataFrame()
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
        train_df = pd.concat([train_df, tmp_df], axis=1, sort=False).astype("Int64")

        # Saves the results from the split in a CSV file just for the train
        train_df.to_csv(path_or_buf="./splits/generation_train_splits.csv", index=False)

    test_df = pd.DataFrame()
    for l in range(k):
        tmp_df = pd.DataFrame()
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
        test_df = pd.concat([test_df, tmp_df], axis=1, sort=False).astype("Int64")

        # Saves the results from the split in a CSV file just for the test
        test_df.to_csv(path_or_buf="./splits/generation_test_splits.csv", index=False)

def iterate_permutations(sample, expected, errors):
    """
    Used to iterate all the possible permutations within a sample

    Args:
        sample (Pandas DataFrame): sample that will be iterated
        expected (List[float]): list of expected voxel counts per 
            class calculated according to the number of volumes in
            each fold
        errors (List[float]): list that contains the count of 
            voxels per class

    Returns:
        best_distribution (List[int]): list with the indexes of 
            the volumes in the sample in the order that gets the 
            best error in order  
    """
    # Initiates a list with the best distribution that 
    # will contain the best order of the five randomly 
    # extracted volumes 
    best_distribution = []
    # Sets the minimum error to infinite
    min_error = float("inf")
    # Iterates through all possible permutations
    for perm in permutations(sample.iterrows(), sample.shape[0]):
        # Sets a possible combination of volumes
        vols_dict = {i: row for i, (index, row) in enumerate(perm)}
        # Initiates a list that will hold the error of each permutation
        tmp_error = [0] * 3
        # Iterates through all the volumes and their voxel counts
        for key, item in vols_dict.items():
            # Updates the error of this permutation
            tmp_error[0] = tmp_error[0] + item["IRF"]
            tmp_error[1] = tmp_error[1] + item["SRF"]
            tmp_error[2] = tmp_error[2] + item["PED"]
        # Adds the errors of the previous volumes to the error
        tmp_error[0] = tmp_error[0] + errors[0]
        tmp_error[1] = tmp_error[1] + errors[1]
        tmp_error[2] = tmp_error[2] + errors[2]        
        # Calculates the final error as the sum of all errors
        final_error = np.sum(np.abs(np.array(errors) - np.array(expected)))
        # In case the error is better than one of the previously 
        # calculated, then stores this permutation as the best 
        # and its respective error, while updating the value of 
        # the minimal error
        if final_error < min_error:
            best_distribution = []
            best_errors = []
            for vol in vols_dict.keys():
                best_distribution.append(vols_dict[vol]["VolumeNumber"])
            best_errors = tmp_error
            min_error = final_error
    return best_distribution, best_errors

def factorial_k_fold_segmentation(k: int=5, random=True):
    """
    In this function, a sample with the size of the number of folds 
    is extracted from the available volumes and all the k! possible 
    solutions are explored with the one that obtains the minimal error
    This function is significantly more expensive because it reaches
    n! complexity (in O notation, O(n!)), but still does not reach 
    the most optimal solution 

    Args:
        k (int): number of folds in the split
        random (bool): flag that indicates whether the volumes are 
            extracted randomly from the DataFrame or by decreasing 
            order in total number of voxels

    Return:
        None
    """
    # Reads the information about each volumes voxel count per class
    df = pd.read_csv("..\\splits\\volumes_info.csv")
    # Gets all the name of the vendors
    vendors_names = df["Vendor"].unique()
    # Creates the dictionary on which it will be added the desired volumes
    selected_volumes = {i: [] for i in range(k)}
    # Iterates through the vendors
    for vendor in vendors_names:
        # Initiates the array that will store the errors 
        # for this vendor
        errors = [0] * 3
        # Gets a sub DataFrame limited to the vendor that 
        # is being iterated
        df_vendor = df[df["Vendor"] == vendor]
        # In case the selection is not random, sorts the DataFrame 
        # in descending order by the total number of voxels
        if not random:
            df_vendor = df_vendor.copy()
            df_vendor.loc[:, "Total"] = df_vendor.iloc[:, 2:].sum(axis=1)
            df_vendor = df_vendor.sort_values(by="Total", ascending=False).drop(columns="Total")
        # Defines the expected value for each class as the
        # maximum number of volumes of a vendor in a fold
        # and the mean of voxel counts in this vendor
        irf_expected = df_vendor.loc[:, "IRF"].mean() * ceil(df_vendor.shape[0] / k)
        srf_expected = df_vendor.loc[:, "SRF"].mean() * ceil(df_vendor.shape[0] / k)
        ped_expected = df_vendor.loc[:, "PED"].mean() * ceil(df_vendor.shape[0] / k)
        expected = [irf_expected, srf_expected, ped_expected]
        while df_vendor.shape[0] != 0:
            # Gets one row for each fold 
            while df_vendor.shape[0] < k:
                df_vendor.loc[df_vendor.shape[0]] = [0,0,0,0,0]
            if random:
                sample = df_vendor.sample(k)
            else:
                sample = df_vendor.head(k)

            # Iterates through the possible permutations of the sample, determining which 
            # is the best possible distribution and the best errors
            best_distribution, best_errors = iterate_permutations(sample, expected, errors)

            # Updates the error values
            for index, error in enumerate(best_errors):
                errors[index] = errors[index] + error
            # Adds the volumes to the dictionary
            for key in selected_volumes.keys():
                if best_distribution != []:
                    selected_volumes[key].append(best_distribution[key])
            # Drops the volumes that already have been selected
            for index, row in df_vendor.iterrows():
                if row["VolumeNumber"] in best_distribution:
                    df_vendor = df_vendor.drop(index)

    # Initiates an empty DataFrame that 
    # will store the volumes selected 
    # per fold
    options_df = pd.DataFrame()
    # Iterates through the dictionary that contains 
    # the list of the values selected by the each fold 
    for key, values in selected_volumes.items():
        # Creates a temporary DataFrame with the values selected and 
        # then concatenates them to the previously initiated DataFrame
        # Saves the values in int instead of float
        tmp_df = pd.DataFrame(values).astype(np.int8)
        tmp_df = tmp_df.replace(0, pd.NA)
        tmp_df = tmp_df.dropna(ignore_index=True)
        options_df = pd.concat([options_df, tmp_df], axis=1, sort=False).astype("Int64")
    # Names the columns in the DataFrame
    options_df.columns = selected_volumes.keys()
    # Saves the DataFrame as a CSV file with no index with a name according to 
    # whether the volumes were randomly sampled or not
    if random:
        options_df.to_csv("..\\splits\\factorial_fold_selection.csv", index=False)
    else:
        options_df.to_csv("..\\splits\\sortedfactorial_fold_selection.csv", index=False)

def competitive_k_fold_segmentation(k: int=5):
    """
    A k number of agents/folds competes for the best possible set of 
    volumes, in order to attain a balanced split according not only 
    to the number of volumes and vendor but also according to the 
    quantity of fluid in each volume. In a random order, the agents 
    are able to choose the volume that minimizes the most the loss, 
    which is defined as the sum of the absolute differences between 
    the expected number of voxel counts per fluid (calculated by 
    multiplying the mean of voxel counts in a determined vendor 
    by maximum the number of volumes a fold can contain, which is 
    5 in all vendors) and the obtained value.
    The folds are also named agents because this approach was 
    inspired by the competition seen in reinforcement learning systems.

    Args:
        k (int): number of folds this split must have

    Return:
        None
    """
    # Initiates the list of agents/folds
    agents_list = np.arange(k)
    # Reads the information about each volumes voxel count per class
    df = pd.read_csv("..\\splits\\volumes_info.csv")
    # Gets all the name of the vendors
    vendors_names = df["Vendor"].unique()
    # Initiates a dictionary that will store the volumes 
    # selected by each agent 
    agents_choices = {i: [] for i in range(k)}
    # Iterates through all the possible vendors
    for vendor in vendors_names:
        # Initiates the matrix that will contain the errors
        # obtained per fold per fluid
        # It has shape (k,3) because there are k agents/folds
        # and three classes
        errors = np.zeros((k,3))
        # Gets a sub DataFrame limited to the vendor that 
        # is being iterated in the previous for cycle
        df_vendor = df[df["Vendor"] == vendor]
        # Randomizes the order of the agents to select a 
        # volume
        shuffle(agents_list)
        # Defines the expected value for each class as the
        # maximum number of volumes of a vendor in a fold
        # and the mean of voxel counts in this vendor
        irf_expected = df_vendor.loc[:, "IRF"].mean() * ceil(df_vendor.shape[0] / k)
        srf_expected = df_vendor.loc[:, "SRF"].mean() * ceil(df_vendor.shape[0] / k)
        ped_expected = df_vendor.loc[:, "PED"].mean() * ceil(df_vendor.shape[0] / k)
        # Iterates through the list of volumes in a vendor 
        # until all volumes have been selected
        while df_vendor.shape[0] != 0:
            # Iterates through the agents/folds in the 
            # list of agents/folds
            for agent in agents_list:
                # Initiates the minimum error as infinite
                min_error = float("inf")
                # Iterates through the volumes in the DataFrame 
                # that contain the volumes available
                for index, row in df_vendor.iterrows():
                    # Extracts the voxel count per class in the 
                    # volume 
                    irf_value = row["IRF"]
                    srf_value = row["SRF"]
                    ped_value = row["PED"]
                    # Calculates the error this volume brings, 
                    # according to the previous volumes selected
                    error = np.abs(irf_expected - (irf_value + errors[agent,0])) \
                    + np.abs(srf_expected - (srf_value + errors[agent,1])) \
                    + np.abs(ped_expected - (ped_value + errors[agent,2]))
                    # In case the error obtained is better 
                    # than the best previous best error, 
                    # its state is saved
                    if error < min_error:
                        best_irf_value = row["IRF"]
                        best_srf_value = row["SRF"]
                        best_ped_value = row["PED"]
                        min_error = error
                        vol_to_append = row["VolumeNumber"]
                        index_to_remove = index
                # Updates the error values according 
                # to the selected volume
                errors[agent,0] += best_irf_value
                errors[agent,1] += best_srf_value
                errors[agent,2] += best_ped_value
                # Saves the selected agent by appending it 
                # to a list of volumes
                agents_choices[agent].append(vol_to_append)
                # Removes the row that contains the volume 
                # selected
                df_vendor = df_vendor.drop(index_to_remove)
                # In case there are no more volumes to select,
                # breaks the second for loop
                if df_vendor.shape[0] == 0:
                    break
            # Shuffles the order in which
            # agents/folds pick their volumes 
            shuffle(agents_list)
    # Initiates an empty DataFrame that 
    # will store the volumes selected 
    # per fold
    options_df = pd.DataFrame()
    # Iterates through the dictionary that contains 
    # the list of the values selected by the each fold 
    for key, values in agents_choices.items():
        # Creates a temporary DataFrame with the values selected and 
        # then concatenates them to the previously initiated DataFrame
        # Saves the values in int instead of float
        tmp_df = pd.DataFrame(values).astype(np.int8)
        options_df = pd.concat([options_df, tmp_df], axis=1, sort=False).astype("Int64")

    # Names the columns in the DataFrame
    options_df.columns = agents_choices.keys()
    # Saves the DataFrame as a CSV file with no index
    options_df.to_csv("..\\splits\\competitive_fold_selection.csv", index=False)

def calculate_error(path: str):
    """
    Calculates the errors associated with the volumes selected in the splits before
    and saves them to a CSV file

    Args:
        path (str): path to the CSV file that contains the division of volumes per 
            folds
    
    Return:
        None
    """
    # Reads the CSV file that contains the division per fold and converts it to a 
    # DataFrame    
    df = pd.read_csv(path)
    # Reads the CSV file that contains the information of the voxels in each volume 
    # and converts it to a DataFrame
    info_df = pd.read_csv("..\\splits\\volumes_info.csv")

    # Iterates through all the folds
    for fold in range(df.shape[1]):
        # Gets the list of vendors
        columns = info_df["Vendor"].unique()
        # Initiates a dictionary responsible 
        # for storing the number of volumes 
        # of a certain vendor
        vendor_count = dict.fromkeys(columns, 0)
        # Initiates a DataFrame with zeros
        results_df = pd.DataFrame(np.zeros((3,3)), columns=columns, index=["IRF", "SRF", "PED"])
        # Iterates through the volumes selected by a fold
        for row in df[str(fold)]:
            if isnan(row):
                break
            # Gets the information of the number of voxels according to the number of the volume
            info = info_df.loc[info_df["VolumeNumber"] == row]
            # Gets the vendor of the volume
            vendor = info["Vendor"].item()
            # Updates the counter of volumes per vendor
            vendor_count[vendor] += 1
            # Updates the total voxel count in the fold directly in the DataFrame
            results_df.at["IRF", vendor] = info["IRF"].item() + results_df.at["IRF", vendor]
            results_df.at["SRF", vendor] = info["SRF"].item() + results_df.at["SRF", vendor] 
            results_df.at["PED", vendor] = info["PED"].item() + results_df.at["PED", vendor]
        # Iterates through the columns of the 
        # DataFrame, which represent in this 
        # case a fold
        for col in results_df.columns:
            # Gets a sub DataFrame of the data
            df_vendor = info_df[info_df["Vendor"] == col]
            # Iterates through the rows in the DataFrame
            for fluid, row in results_df[col].items():
                # Calculates the expected value according to 
                # the mean and the number of volumes of a 
                # certain vendor present in the fold or according 
                # to the expected value used to calculate the 
                # error during the fold splitting
                # expected = df_vendor.loc[:, fluid].mean() * vendor_count[col]
                expected = df_vendor.loc[:, fluid].mean() * ceil(df_vendor.shape[0] / df.shape[1])
                # Calculates the difference and updates 
                # the value in the DataFrame
                results_df.at[fluid, col] = row - expected   
        # Initiates strings that will be used as 
        # separators but if indicated inside the 
        # f-string will interrupt ""
        backslash = "\\"
        underscore = "_"
        # Saves the DataFrame as a CSV file
        pd.DataFrame(results_df).to_csv(path_or_buf=f"..\\splits\\{path.split(backslash)[2].split(underscore)[0]}_errors_fold{fold}.csv")
    # Calls the function that will present the average error and its standard deviation
    quantify_errors(file_name=f"..\\splits\\{path.split(backslash)[2].split(underscore)[0]}_errors_fold", k=df.shape[1])

def quantify_errors(file_name: str, k: int=5):
    """
    This function is called to calculate the mean error and standard deviation of 
    errors across all folds, for each specific vendor and fluid type, resulting in
    a table of shape 3x3 that is saved as a CSV file

    Args:
        file_name (str): path of the files that contain the error in each fold 
            (e.g. "..\\splits\\manual_errors_fold"). This path does not contain 
            nor a number of the fold nor the file extension (.csv)
        k (int): number of folds considered

    Return: 
        None
    """
    # Loads the errors of the first fold to initialize the arrays with the errors 
    # and calculate the number of rows and columns that will be handled
    example_df = pd.read_csv(f"{file_name}0.csv", index_col=0, header=0)
    # Initializes the matrices that will store 
    # the mean error and their standard deviation 
    mean_results = np.zeros(example_df.shape)
    std_results = np.zeros(example_df.shape)
    # Iterates through all the columns and rows of the 
    # error matrices
    for col_index, col in enumerate(example_df.columns):
        for row_index, row in enumerate(example_df.index):
            # Initializes a list that will store the values for 
            # the indicated row and column across all folds
            values = []
            # Iterates through all folds
            for fold in range(k):
                # Reads the corresponding fold CSV file
                df = pd.read_csv(f"{file_name}{fold}.csv", index_col=0, header=0)
                # Appends the results to the list                
                values.append(np.abs(df[col].loc[row]))
            # Calculates the mean and std results for this specific row 
            # and col, across all folds
            mean_results[row_index, col_index] = np.array(values).mean()
            std_results[row_index, col_index] = np.array(values).std()
    # Declares the name in which it is expected to save
    previous_file_name = file_name.split("\\")[2].split("_")
    new_file_name = file_name.split("\\")[0] \
    + "\\" + file_name.split("\\")[1] \
    + "\\" + previous_file_name[0] \
    + "_" + previous_file_name[1] + "_"
    # Saves both matrices to two different CSV files
    pd.DataFrame(columns=example_df.columns, 
                 index=example_df.index, 
                 data=mean_results).round(decimals=2).to_csv(new_file_name + "mean.csv")
    pd.DataFrame(columns=example_df.columns, 
                 index=example_df.index, 
                 data=std_results).round(decimals=2).to_csv(new_file_name + "std.csv")
    
def folds_information(file_name: str, k: int=5):
    info_df = pd.read_csv("..\\splits\\volumes_info.csv")
    split_df = pd.read_csv(file_name)

    columns = []
    for vendor in info_df["Vendor"].unique():
        for fluid in ["IRF", "SRF", "PED"]:
            columns.append(vendor + fluid)
    complete_df = pd.DataFrame(columns=columns)
    for fold in range(k):
        vols_list = split_df[str(fold)].dropna().to_list()
        row = {fold: []}
        for vendor in info_df["Vendor"].unique():
            df_vendor = info_df[info_df["Vendor"] == vendor]
            vendor_per_volume_list = df_vendor["VolumeNumber"].unique()
            totals = [0] * 3
            for volume in vols_list:
                if volume in vendor_per_volume_list:
                    info_row = info_df.loc[volume - 1]
                    for index, fluid in enumerate(["IRF", "SRF", "PED"]):
                        totals[index] = totals[index] + info_row[fluid]
            row[fold] = row[fold] + totals
        complete_df.loc[len(complete_df)] = row[fold] 

    fluid_df = pd.DataFrame(columns=["IRF", "SRF", "PED"])
    for fold in range(k):
        vols_list = split_df[str(fold)].dropna().to_list()
        row = {fold: []}
        totals = [0] * 3
        for volume in vols_list:
            info_row = info_df.loc[volume - 1]
            for index, fluid in enumerate(["IRF", "SRF", "PED"]):
                totals[index] = totals[index] + info_row[fluid]
        row[fold] = row[fold] + totals
        fluid_df.loc[len(fluid_df)] = row[fold] 

    vendor_to_slices_dict = {
        "Cirrus": 128,
        "Spectralis": 128,
        "Topcon": 128,
    }
    columns = []
    for vendor in info_df["Vendor"].unique():
        for suffix in ["Slices", "Volumes"]:
            col_name = vendor + suffix
            columns.append(col_name)
    volumes_slices_df = pd.DataFrame(columns=columns)
    for fold in range(k):
        vols_list = split_df[str(fold)].dropna().to_list()
        row = {fold: []}
        for vendor in info_df["Vendor"].unique():
            slices_volumes = [0] * 2
            df_vendor = info_df[info_df["Vendor"] == vendor]
            vendor_per_volume_list = df_vendor["VolumeNumber"].unique()
            for volume in vols_list:
                if volume in vendor_per_volume_list:
                    slices_volumes[1] = slices_volumes[1] + 1
                    if (vendor == "Topcon") and ((volume == 56) or (volume == 62)):
                        slices_volumes[0] = slices_volumes[0] + 64
                    else:
                        slices_volumes[0] = slices_volumes[0] + vendor_to_slices_dict[vendor]
            row[fold] = row[fold] + slices_volumes
        volumes_slices_df.loc[len(volumes_slices_df)] = row[fold] 
    
    name_to_save = file_name[:-4]
    complete_df.to_csv(name_to_save + "_overall.csv")
    fluid_df.to_csv(name_to_save + "_fluid.csv")
    volumes_slices_df.to_csv(name_to_save + "_volslices.csv")

if __name__ == "__main__":
    # factorial_k_fold_segmentation()
    random_k_fold_segmentation(folders_path="D:\RETOUCH", k=5)
    # calculate_error(path="..\\splits\\sortedfactorial_fold_selection.csv")
    # folds_information("..\\splits\\manual_fold_selection.csv")