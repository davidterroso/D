import numpy as np
import pandas as pd
from itertools import permutations, product
from math import ceil, isnan
from os import makedirs, walk
from random import shuffle
from statistics import stdev

#############################################
# Volumes from Topcon T-1000 with 64 slices:# 
# Training - 056, 062                       #
# Testing - None                            #
#############################################

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
    # Saves the file as CSV
    final_df.to_csv(".\\splits\\segmentation_fold_selection.csv", index=False)

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
            # Saves the set to which the data 
            # belongs, whether it is train or test
            # This is important because the number 
            # of the volume is not enough to 
            # identify it
            if (train_or_test[1] == "TrainingSet"):
                belonging_set = "train"
            elif (train_or_test[1] == "TestSet"):
                belonging_set = "test"
            vendor_volume = train_or_test[2].split("""\\""")
            if ((len(vendor_volume) == 1) and (vendor_volume[0] not in dictionary)):
                dictionary[vendor_volume[0]] = []
            elif len(vendor_volume) == 2:
                # Saves the number of the volume and 
                # the set to which their belong as a 
                # tuple in the dictionary
                # Each volume is now represented as,
                # for example: (001, train)
                vendor = vendor_volume[0]
                volume = vendor_volume[1][-3:]
                volume_set = (volume, belonging_set)
                dictionary[vendor].append(volume_set)

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
    # tuples that will have the number of the volume and 
    # the set to which they belong associated to each 
    # fold (key)
    fold_vol_dict = {}
    # Iterates through the folds
    for fold in range(k):
        # In each fold initializes an empty list that 
        # will hold the indexes of the volumes and the 
        # set they belong to
        fold_volumes = []
        # Iterates throgh all the possible vendors
        for vendor in dictionary.keys():
            # Shuffles the list that contains the 
            # tuples that hold the volumes indices 
            # of the said vendor and the sets they 
            # belong to
            shuffle(dictionary[vendor])
            # Checks how many volumes are supposed to extract
            num_vols = vols_per_fold_dict[vendor][fold]
            # Extracts the calculated volumes
            vols = dictionary[vendor][-num_vols:]
            # Converts the volume information 
            # from a tuple to a string
            # Example: 1_train
            for index, vol in enumerate(vols):
                vol_num = str(int(vol[0]))
                vol_set = vol[1]
                vol_info = f"{vol_num}_{vol_set}"
                vols[index] = vol_info
            # Updates the dictionary, removing the extracted volumes
            dictionary[vendor] = dictionary[vendor][:-num_vols]
            # Appends the newly extracted volumes to the list that 
            # will hold the volumes for this vendor
            fold_volumes = fold_volumes + vols
        # Places in the key corresponding 
        # to the fold the string that 
        # identifies each volume that 
        # composes it 
        fold_vol_dict[fold] = fold_volumes

    # Initiates the final DataFrame that will be saved
    final_df = pd.DataFrame()
    # Iterates through all the folds
    for fold, data in fold_vol_dict.items():
        # Converts the data in each fold to a DataFrame 
        # and joins them
        tmp_df = pd.DataFrame(columns=[fold], data=data)
        final_df = pd.concat([final_df, tmp_df], axis=1, sort=False)
    # Saves the file as CSV
    final_df.to_csv(".\\splits\\generation_fold_selection.csv", index=False)

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

def factorial_k_fold_segmentation(k: int=5, random: bool=True, 
                                  fluid: str=None, test_fold: int=1):
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
        fluid (str): string that indicates for which fluid the fold 
            split will be performed. Must be IRF, SRF, or PED. The 
            default is None
        test_fold (int): when a fold split is done to all the fluids, 
            one fold is being held out for comparison between different 
            implementations. Therefore, none of the volumes contained 
            in this fold can be used in training of other implementations
            (e.g. implementations that perform binary segmentation on a 
            specific fluid). The default value is one because that is 
            the fold that was not used in training or validation of the
            U-Net model

    Return:
        None
    """
    # Reads the information about each volumes voxel count per class
    df = pd.read_csv(".\\splits\\volumes_info.csv")
    # Gets all the name of the vendors
    vendors_names = df["Vendor"].unique()
    # Creates the dictionary on which it will be added the desired volumes
    selected_volumes = {i: [] for i in range(k)}
    if fluid is not None:
        # Checks if the input fluid name is available
        fluids_list = ["IRF","SRF","PED"]
        assert fluid in fluids_list,\
            f"Invalid fluid: {fluid}. Available fluids: {', '.join(fluids_list)}"
        # Reads the competitive fold selection 
        # and records the prohibited volumes
        full_split = pd.read_csv(".\\splits\\competitive_fold_selection.csv")
        # Gets the list of all the volumes in the test fold
        prohibited_vols = full_split[str(test_fold)].to_list()
        # Removes the prohibited volumes from the available volumes DataFrame
        df = df[~df["VolumeNumber"].isin(prohibited_vols)]
        # Removes the fold being used in testing from the dictionary 
        selected_volumes.pop(test_fold)
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
        if fluid is None:
            irf_expected = df_vendor.loc[:, "IRF"].mean() * ceil(df_vendor.shape[0] / k)
            srf_expected = df_vendor.loc[:, "SRF"].mean() * ceil(df_vendor.shape[0] / k)
            ped_expected = df_vendor.loc[:, "PED"].mean() * ceil(df_vendor.shape[0] / k)
            expected = [irf_expected, srf_expected, ped_expected]
        else:
            fluid_expected = df_vendor.loc[:, fluid].mean() * ceil(df_vendor.shape[0] / (k - 1))
            expected = [fluid_expected]

        while df_vendor.shape[0] != 0:
            # Gets one row for each fold 
            if fluid is None:
                while df_vendor.shape[0] < k:
                    df_vendor.loc[df_vendor.shape[0]] = [0,0,0,0,0]
                if random:
                    sample = df_vendor.sample(k)
                else:
                    sample = df_vendor.head(k)
            else:
                while df_vendor.shape[0] < (k - 1):
                    df_vendor.loc[df_vendor.shape[0]] = [0,0,0,0,0]
                if random:
                    sample = df_vendor.sample(k - 1)
                else:
                    sample = df_vendor.head(k - 1)

            # Iterates through the possible permutations of the sample, determining which 
            # is the best possible distribution and the best errors
            best_distribution, best_errors = iterate_permutations(sample, expected, errors)

            # Updates the error values
            for index, error in enumerate(best_errors):
                errors[index] = errors[index] + error
            # Adds the volumes to the dictionary
            for index, key in enumerate(selected_volumes.keys()):
                if best_distribution != []:
                    selected_volumes[key].append(best_distribution[index])
            # Drops the volumes that already have been selected
            for index, row in df_vendor.iterrows():
                if row["VolumeNumber"] in best_distribution:
                    df_vendor = df_vendor.drop(index)

    if fluid is not None:
        # Re-adds the prohibited volumes to 
        # the list
        dict_keys = list(selected_volumes.keys())
        dict_values = list(selected_volumes.values())
        keys_index = dict_keys.index(test_fold - 1)
        new_keys = dict_keys[:keys_index + 1] + [test_fold] + dict_keys[keys_index + 1:]
        new_values = dict_values[:keys_index + 1] + [prohibited_vols] + dict_values[keys_index + 1:]
        selected_volumes = dict(zip(new_keys, new_values))

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
    if fluid is None:
        if random:
            options_df.to_csv(".\\splits\\factorial_fold_selection.csv", index=False)
        else:
            options_df.to_csv(".\\splits\\sorted_factorial_fold_selection.csv", index=False)
    else:
        if random:
            options_df.to_csv(f".\\splits\\factorial_fold_selection_{fluid}.csv", index=False)
        else:
            options_df.to_csv(f".\\splits\\sorted_factorial_fold_selection_{fluid}.csv", index=False)

def competitive_k_fold_segmentation(k: int=5, fluid: str=None, test_fold: int=1):
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
        fluid (str): string that indicates for which fluid the fold 
            split will be performed. Must be IRF, SRF, or PED. The 
            default is None
        test_fold (int): when a fold split is done to all the fluids, 
            one fold is being held out for comparison between different 
            implementations. Therefore, none of the volumes contained 
            in this fold can be used in training of other implementations
            (e.g. implementations that perform binary segmentation on a 
            specific fluid). The default value is one because that is 
            the fold that was not used in training or validation of the
            U-Net model

    Return:
        None
    """
    # Initiates the list of agents/folds
    agents_list = np.arange(k)
    # Reads the information about each volumes voxel count per class
    df = pd.read_csv(".\\splits\\volumes_info.csv")
    # Gets all the name of the vendors
    vendors_names = df["Vendor"].unique()
    # Initiates a dictionary that will store the volumes 
    # selected by each agent 
    agents_choices = {i: [] for i in range(k)}
    if fluid is not None:
        # Checks if the input fluid name is available
        fluids_list = ["IRF","SRF","PED"]
        assert fluid in fluids_list,\
            f"Invalid fluid: {fluid}. Available fluids: {', '.join(fluids_list)}"
        # Reads the competitive fold selection 
        # and records the prohibited volumes
        full_split = pd.read_csv(".\\splits\\competitive_fold_selection.csv")
        # Gets the list of all the volumes in the test fold
        prohibited_vols = full_split[str(test_fold)].to_list()
        # Deletes from the list of agents the fold that is being used in test
        agents_list = np.delete(agents_list, np.where(agents_list==test_fold))
        # Removes the prohibited volumes from the available volumes DataFrame
        df = df[~df["VolumeNumber"].isin(prohibited_vols)]
        # Removes the fold being used in testing from the dictionary 
        agents_choices.pop(test_fold)
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
        if fluid is None:
            irf_expected = df_vendor.loc[:, "IRF"].mean() * ceil(df_vendor.shape[0] / k)
            srf_expected = df_vendor.loc[:, "SRF"].mean() * ceil(df_vendor.shape[0] / k)
            ped_expected = df_vendor.loc[:, "PED"].mean() * ceil(df_vendor.shape[0] / k)
        else:
            fluid_expected = df_vendor.loc[:, fluid].mean() * ceil(df_vendor.shape[0] / (k - 1))
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
                    if fluid is None:
                        irf_value = row["IRF"]
                        srf_value = row["SRF"]
                        ped_value = row["PED"]
                        # Calculates the error this volume brings, 
                        # according to the previous volumes selected
                        error = np.abs(irf_expected - (irf_value + errors[agent,0])) \
                        + np.abs(srf_expected - (srf_value + errors[agent,1])) \
                        + np.abs(ped_expected - (ped_value + errors[agent,2]))
                    else:
                        fluid_value = row[fluid]
                        error = np.abs(fluid_expected - (fluid_value + errors[agent,0]))
                    # In case the error obtained is better 
                    # than the best previous best error, 
                    # its state is saved
                    if error < min_error:
                        if fluid is None:
                            best_irf_value = row["IRF"]
                            best_srf_value = row["SRF"]
                            best_ped_value = row["PED"]
                        else:
                            best_fluid_value = row[fluid]
                        min_error = error
                        vol_to_append = row["VolumeNumber"]
                        index_to_remove = index
                # Updates the error values according 
                # to the selected volume
                if fluid is None:
                    errors[agent,0] += best_irf_value
                    errors[agent,1] += best_srf_value
                    errors[agent,2] += best_ped_value
                else:
                    errors[agent,0] += best_fluid_value

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
    if fluid is not None:
        # Re-adds the prohibited volumes to 
        # the list
        dict_keys = list(agents_choices.keys())
        dict_values = list(agents_choices.values())
        keys_index = dict_keys.index(test_fold - 1)
        new_keys = dict_keys[:keys_index + 1] + [test_fold] + dict_keys[keys_index + 1:]
        new_values = dict_values[:keys_index + 1] + [prohibited_vols] + dict_values[keys_index + 1:]
        agents_choices = dict(zip(new_keys, new_values))
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
    if fluid is None:
        options_df.to_csv(".\\splits\\competitive_fold_selection.csv", index=False)
    else:
        options_df.to_csv(f".\\splits\\competitive_fold_selection_{fluid}.csv", index=False)

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
    info_df = pd.read_csv(".\\splits\\volumes_info.csv")

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
        pd.DataFrame(results_df).to_csv(path_or_buf=f".\\splits\\{path.split(backslash)[1].split(underscore)[0]}_errors_fold{fold}.csv")
    # Calls the function that will present the average error and its standard deviation
    quantify_errors(file_name=f".\\splits\\{path.split(backslash)[1].split(underscore)[0]}_errors_fold", k=df.shape[1])

def quantify_errors(file_name: str, k: int=5):
    """
    This function is called to calculate the mean error and standard deviation of 
    errors across all folds, for each specific vendor and fluid type, resulting in
    a table of shape 3x3 that is saved as a CSV file

    Args:
        file_name (str): path of the files that contain the error in each fold 
            (e.g. ".\\splits\\manual_errors_fold"). This path does not contain 
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
    
def folds_information(file_name: str):
    """
    Useful for the extraction of fold information regarding fluid voxel 
    counts in each vendors, fluid voxel counts across all vendors, and 
    volumes and slice quantification  

    Args:
        file_name (str): name of the CSV file that contains the volumes 
            distribution across folds
        
    Return:
        None
    """
    # Loads the CSV files that contain the data regarding 
    # all the volumes and the volumes fold split 
    info_df = pd.read_csv(".\\splits\\volumes_info.csv")
    split_df = pd.read_csv(file_name)
    
    # Creates the name of the columns, joining 
    # both the vendor and the fluid type to 
    # which the data refers
    columns = []
    for vendor in info_df["Vendor"].unique():
        for fluid in ["Background", "IRF", "SRF", "PED"]:
            columns.append(vendor + fluid)

    # Initiates the DataFrame that will handle 
    # the data
    complete_df = pd.DataFrame(columns=columns)
    # Iterates through each fold
    for fold in range(split_df.shape[1]):
        # Gets the list of volumes of the fold
        vols_list = split_df[str(fold)].dropna().to_list()
        # Initiates a dictionary that will have the number of the fold 
        # and the respective desired data
        row = {fold: []}
        # Iterates through all the possible vendors
        for vendor in info_df["Vendor"].unique():
            # Gets a sub informative DataFrame corresponding to the 
            # vendor that is being analyzed
            df_vendor = info_df[info_df["Vendor"] == vendor]
            # Gets the list of all the volumes from this vendor
            vendor_per_volume_list = df_vendor["VolumeNumber"].unique()
            # Initiates a list that will have the voxel count for this 
            # volume across all three classes
            totals = [0] * 4
            # Iterates through all the volumes in 
            # the volumes list of this vendor
            for volume in vols_list:
                # Only procedes if the said volume is 
                # in the list of volumes of the fold 
                # to analyze
                if volume in vendor_per_volume_list:
                    # Gets the information of the volume 
                    # from the informative DataFrame
                    # The index of the volume is the number of the volume 
                    # minus one because the index starts in zero and the 
                    # number of the volumes start in one
                    info_row = info_df.loc[volume - 1]
                    # Iterates through all the possible classes
                    for index, fluid in enumerate(["Background", "IRF", "SRF", "PED"]):
                        # Sums the total number of voxels of the said 
                        # class to the ones previously summed in the 
                        # other volumes of the fold
                        totals[index] = totals[index] + info_row[fluid]
            # Appends the data of this vendor 
            # to the row of data
            row[fold] = row[fold] + totals
        # Adds the row of data to the DataFrame
        complete_df.loc[len(complete_df)] = row[fold] 

    # Initiates a DataFrame with the three possible fluids 
    # as columns
    fluid_df = pd.DataFrame(columns=["Background", "IRF", "SRF", "PED"])
    # Iterates through all the folds
    for fold in range(split_df.shape[1]):
        # Gets the list of volumes in this fold
        vols_list = split_df[str(fold)].dropna().to_list()
        # Initiates the list of total values 
        # as zero, with one value for each class
        totals = [0] * 4
        # Iterates through all the 
        # volumes in the list of 
        # volumes of this fold
        for volume in vols_list:
            # Gets the information of the volume 
            # regarding voxel counts
            # The index of the volume is the number of the volume 
            # minus one because the index starts in zero and the 
            # number of the volumes start in one
            info_row = info_df.loc[volume - 1]
            # Iterates through the possible fluids
            for index, fluid in enumerate(["Background", "IRF", "SRF", "PED"]):
                # Adds the number of voxels in this volume to 
                # the other ons in this fold
                totals[index] = totals[index] + info_row[fluid]
        # Appends to the data of this fold to the DataFrame
        fluid_df.loc[len(fluid_df)] = totals

    # Dictionary that indicates 
    # the number of slices per volume
    vendor_to_slices_dict = {
        "Cirrus": 128,
        "Spectralis": 49,
        "Topcon": 128,
    }
    # Initiates a list that will contain the name 
    # of the columns in the DataFrame
    columns = []
    # Iterates through the vendors
    for vendor in info_df["Vendor"].unique():
        # Appends to the name of the vendors 
        # Slices or Volumes, indicating that 
        # the values in the column correspond 
        # to the number of slices in the 
        # volume and the total number of 
        # volumes
        for suffix in ["Slices", "Volumes"]:
            col_name = vendor + suffix
            columns.append(col_name)
    # Initiates a DataFrame with the columns created
    volumes_slices_df = pd.DataFrame(columns=columns)
    # Iterates through all the folds
    for fold in range(split_df.shape[1]):
        # Gets the list of volumes in a fold
        vols_list = split_df[str(fold)].dropna().to_list()
        # Initiates the row dictionary that will 
        # have a corresponding list of values to 
        # each fold
        row = {fold: []}
        # Iterates through all the possible vendors
        for vendor in info_df["Vendor"].unique():
            # Initiates the list as 
            # an array of zeros, one
            # zero for slice count 
            # and one for volume count
            slices_volumes = [0] * 2
            # Gets a sub DataFrame that corresponds to the 
            # information of the volumes of the vendor that 
            # is being analyzed
            df_vendor = info_df[info_df["Vendor"] == vendor]
            # Gets a list of all the volumes from this vendor
            vendor_per_volume_list = df_vendor["VolumeNumber"].to_list()
            # Iterates through the volumes in this fold
            for volume in vols_list:
                # Only procedes if the volume is in 
                # the list of volumes in this fold 
                if volume in vendor_per_volume_list:
                    # Sums one to the total number of volumes
                    slices_volumes[1] = slices_volumes[1] + 1
                    # Adds the total number of slices according to the dictionary initiated
                    # The volumes 56 and 62 are handled differently because their number of 
                    # slices is not the same as other Topcon volumes
                    if (vendor == "Topcon") and ((volume == 56) or (volume == 62)):
                        slices_volumes[0] = slices_volumes[0] + 64
                    else:
                        slices_volumes[0] = slices_volumes[0] + vendor_to_slices_dict[vendor]
            # Adds the information of the volumes to the row 
            row[fold] = row[fold] + slices_volumes
        # Adds this row of information as a row in the DataFrame
        volumes_slices_df.loc[len(volumes_slices_df)] = row[fold] 
    
    # Gets the under which the files will be saved
    name_to_save = file_name[:-4]
    # Saves each file with an adequate name
    complete_df.to_csv(name_to_save + "_overall.csv")
    fluid_df.to_csv(name_to_save + "_fluid.csv")
    volumes_slices_df.to_csv(name_to_save + "_volslices.csv")

def split_info(file_name: str):
    """
    Receives the name of a fold split and calculates 
    the total number of voxels per class per fold

    Args:
        file_name (str): name of the fold split to 
            calculate the values
    Returns:
        None
    """
    # Gets the split identifier
    split_identifier = file_name.split("\\")[2].split("_")[0]
    if len(file_name.split("\\")[2].split("_")) == 4:
        fluid_identifier = file_name.split("\\")[2].split("_")[-1].split(".")[0]
    # Reads as a DataFrame the CSV file 
    # that contains the volume in each split
    df = pd.read_csv(file_name)
    # Reads as a DataFrame the CSV file that contains all the 
    # information of the volumes
    volumes_info_df = pd.read_csv("..\\splits\\volumes_info.csv")

    # Initiates the DataFrame to save
    mean_std_df = pd.DataFrame()
    sum_df = pd.DataFrame()

    # Iterates through the folds 
    # in the splits dataframe
    for column in df.columns:
        # Gets the information 
        # from the desired column
        column_df = df[column]
        # Merges the two DataFrames on the condition of having the same number of volume
        merged_df = volumes_info_df.merge(column_df, left_on="VolumeNumber", right_on=column)
        # Resumes the information calculating its mean and standard deviation
        merged_df_mean = merged_df.groupby("Vendor").mean().drop([column, "VolumeNumber"], axis=1).round(2)
        merged_df_std = merged_df.groupby("Vendor").std().drop([column, "VolumeNumber"], axis=1).round(2)
        # Resumes the information in one single table
        merged_df_mean_std = merged_df_mean.astype(str) + " (" + merged_df_std.astype(str) + ")"
        # Concatenates the DataFrame into a single DataFrame vertically
        mean_std_df = pd.concat([mean_std_df, merged_df_mean_std], ignore_index=False, axis=0) 
        # Resumes the information calculating its sum
        merged_df_sum = merged_df.groupby("Vendor").sum().drop([column, "VolumeNumber"], axis=1).round(2)
        # Concatenates the DataFrame into a single DataFrame vertically
        sum_df = pd.concat([sum_df, merged_df_sum], ignore_index=False, axis=0)

    # Declares the name under which the DataFrames will be saved
    if "fluid_identifier" in locals():
        save_name = f"..\splits\\{split_identifier}_{fluid_identifier}_volumes"
    else:
        save_name = f"..\splits\\{split_identifier}_volumes"
    mean_std_save_name = save_name + "_mean.csv"
    sum_save_name = save_name + "_sum.csv"
    # Saves the DataFrames
    mean_std_df.to_csv(mean_std_save_name)
    sum_df.to_csv(sum_save_name)

def generation_5_fold_split():
    """
    This function is used to generate an even
    five fold split for the image generation 
    network. In this function, the arrays with 
    the distributed quantity of volumes in each 
    fold are already given, handling the special 
    cases separately. An array that contains the
    number of slices per volume is also given. 
    Then, all the possible combinations of this 
    distributions are calculated and the one that 
    provides a smaller standard deviation 
    regarding the total number of slices per fold
    is selected. The special cases may not be in 
    the same fold together or in the same fold as 
    the number with highest number of volumes in 
    their respective device. Lastly, it is saved
    in a CSV file 

    Args:
        None
    
    Returns:
        None
    """
    # Initiates the array with the ideal 
    # partition of volumes of each device 
    # per fold
    cirrus_split = [8, 8, 8, 7, 7]
    spectralis_split = [8, 8, 8, 7, 7]
    t1000_split = [3, 3, 3, 2, 2]
    t2000_split = [5, 4, 4, 4, 4]
    # Two special cases with 64 slices each
    t2000_special_split_1 = [1, 0, 0, 0, 0]
    t2000_special_split_2 = [1, 0, 0, 0, 0]

    # Initiates the with the number of sliced 
    # per device
    slices_num = [128, 49, 128, 128, 64, 64]

    # Calculates all the possible permutations for each device
    cirrus_perms = set(permutations(cirrus_split))
    spectralis_perms = set(permutations(spectralis_split))
    t1000_perms = set(permutations(t1000_split))
    t2000_perms = set(permutations(t2000_split))
    t2000_special_split_1_perms = set(permutations(t2000_special_split_1))
    t2000_special_split_2_perms = set(permutations(t2000_special_split_2))

    # Saves the row where 
    # this data is in the 
    # matrix
    T2000_INDEX = 3
    SPECIAL_1_INDEX = 4
    SPECIAL_2_INDEX = 5

    # Initiates the optimal 
    # minimum standard 
    # deviation and best 
    # matrix of data
    min_std = float('inf')
    best_matrix = None

    # Iterates through all the possible combinations
    for combo in product(
        cirrus_perms,
        spectralis_perms,
        t1000_perms,
        t2000_perms,
        t2000_special_split_1_perms,
        t2000_special_split_2_perms
    ):
        # Sets the combination as 
        # a NumPy matrix of shape 
        # (6, 5)
        matrix = np.array(combo)

        # Ensures that no column has 5 T-2000 
        # volumes and one the each special cases
        invalid = False
        for col in range(5):
            if (
                matrix[T2000_INDEX, col] == 5 and
                (matrix[SPECIAL_1_INDEX, col] == 1 or matrix[SPECIAL_2_INDEX, col] == 1)
            ):
                invalid = True
                break
        # In case the condition is broken, 
        # the combination is ignored
        if invalid:
            continue

        # Ensures that the special cases can not 
        # be in the same column
        col_1 = np.argmax(matrix[SPECIAL_1_INDEX])
        col_2 = np.argmax(matrix[SPECIAL_2_INDEX])
        # If that happens, 
        # the combination is 
        # ignored
        if col_1 == col_2:
            continue

        # Transposes the matrix for row-wise 
        # slice multiplication
        transposed = matrix.T
        weighted = transposed * slices_num
        # Sums all the values along the columns
        col_sums = weighted.sum(axis=1)
        # Calculates the standard deviation of 
        # the matrix
        std_val = stdev(col_sums)

        # If this value is better than the 
        # best case, then it is saved
        if std_val < min_std:
            min_std = std_val
            best_matrix = matrix.copy()

    # Loads the data with the train volumes info
    train_volumes = pd.read_csv("..\\splits\\volumes_info.csv")
    # Sets a column indicating it is from 
    # the train set
    train_volumes["origin_set"] = "train"

    # Loads the data with the test volumes info
    test_volumes = pd.read_csv("..\\splits\\volumes_info_test.csv")
    # Sets a column indicating it is from 
    # the test set
    test_volumes["origin_set"] = "test"

    # Combines both sets into a single DataFrame
    full_data = pd.concat([train_volumes, test_volumes], ignore_index=True)

    # Create a unified identifier column with shape "1_train"
    full_data["VolumeNumber"] = full_data["VolumeNumber"].astype(str) + "_" + full_data["origin_set"]

    # Createsa dictionary that contains 
    # a list of volumes for each fold
    folds = {i: [] for i in range(5)}
    # Initiates a set that 
    # contains all the used 
    # volumes
    used_indices = set()

    # Creates a matrix of 
    # devices and assign 
    # the value False to 
    # indicate that they 
    # are not special cases
    matrix_devices = [
        ('Cirrus', False),
        ('Spectralis', False),
        ('T-1000', False),
        ('T-2000', False)
    ]

    # Iterates through all the devices available
    for device_idx, (device_name, _) in enumerate(matrix_devices):
        # Gets the row that explains the device partition
        matrix_row = best_matrix[device_idx]
        # Gets a sub-DataFrame with the data of this device
        device_volumes = full_data[full_data['Device'] == device_name]
        # Shuffles the list to ensure the volumes are 
        # picked randomly
        device_volumes = device_volumes.sample(frac=1)

        # Iterates through all the five folds
        for fold_idx, num_samples in enumerate(matrix_row):
            # Counts the total number 
            # of samples
            count = int(num_samples)
            # Appends the 
            # selected volumes 
            # to a list
            selected = []

            # Iterates through all the volumes in 
            # of the said device
            for _, row in device_volumes.iterrows():
                # Gets the ID of 
                # the device
                row_id = row.name
                # Checks if it has not been 
                # used yet
                if row_id not in used_indices:
                    # In case it is unused, 
                    # appends it to the list
                    # of the volumes of this 
                    # split and to the list of
                    # seen volumes
                    selected.append(row_id)
                    used_indices.add(row_id)
                    # Whenever all the volumes 
                    # have been seen, it breaks
                    if len(selected) == count:
                        break

            # Gets the volumes selected to a list
            fold_ids = full_data.loc[selected, 'VolumeNumber'].tolist()
            # Assigns the list to the dictionary 
            # with the volumes identifiers
            folds[fold_idx].extend(fold_ids)

    # Handles the two special cases manually
    # Selects the data and shuffles them
    special_volumes = full_data[
        (full_data['Device'] == 'T-2000') & (full_data['SlicesNumber'] == 64)
    ].sample(frac=1)

    # Gets the folds to which it is expected 
    # to append
    special_1_fold = np.argmax(best_matrix[4])
    special_2_fold = np.argmax(best_matrix[5])

    # Assigns each one to the target folder
    folds[special_1_fold].append(special_volumes.iloc[0]['VolumeNumber'])
    folds[special_2_fold].append(special_volumes.iloc[1]['VolumeNumber'])


    # Creates the output DataFrame with indexes from 0 to 4 as the names
    split_df = pd.DataFrame({str(k): pd.Series(v) for k, v in folds.items()})

    # Saves the data to a CSV
    split_df.to_csv("..\\splits\\generation_5_fold_split.csv", index=False)

def generation_4_fold_split():
    """
    This function is used to generate an even
    four fold split for the image generation 
    network. In this function, the arrays with 
    the distributed quantity of volumes in each 
    fold are already given, handling the special 
    cases separately. An array that contains the
    number of slices per volume is also given. 
    Then, all the possible combinations of this 
    distributions are calculated and the one that 
    provides a smaller standard deviation 
    regarding the total number of slices per fold
    is selected. The special cases may not be in 
    the same fold together or in the same fold as 
    the number with highest number of volumes in 
    their respective device. Lastly, it is saved
    in a CSV file.
    The difference between this function and the
    generation_5_fold split is that only four 
    folds are considered, as the fifth fold is 
    considered to be the fold used in testing for 
    the segmentation task

    Args:
        None
    
    Returns:
        None
    """
    # Initiates the array with the ideal 
    # partition of volumes of each device 
    # per fold
    cirrus_split = [9, 9, 8, 8]
    spectralis_split = [9, 8, 8, 8]
    t1000_split = [3, 2, 2, 2]
    t2000_split = [5, 5, 5, 4]
    # Two special cases with 64 slices each
    t2000_special_split_1 = [1, 0, 0, 0]

    # Initiates the with the number of sliced 
    # per device
    slices_num = [128, 49, 128, 128, 64]

    # Calculates all the possible permutations for each device
    cirrus_perms = set(permutations(cirrus_split))
    spectralis_perms = set(permutations(spectralis_split))
    t1000_perms = set(permutations(t1000_split))
    t2000_perms = set(permutations(t2000_split))
    t2000_special_split_1_perms = set(permutations(t2000_special_split_1))

    # Saves the row where 
    # this data is in the 
    # matrix
    T2000_INDEX = 3
    SPECIAL_1_INDEX = 4

    # Initiates the optimal 
    # minimum standard 
    # deviation and best 
    # matrix of data
    min_std = float('inf')
    best_matrix = None

    # Iterates through all the possible combinations
    for combo in product(
        cirrus_perms,
        spectralis_perms,
        t1000_perms,
        t2000_perms,
        t2000_special_split_1_perms
    ):
        # Sets the combination as 
        # a NumPy matrix of shape 
        # (5, 4)
        matrix = np.array(combo)

        # Ensures that no column has 5 T-2000 
        # volumes and one the each special cases
        invalid = False
        for col in range(4):
            if (
                (matrix[T2000_INDEX, col] == 5) and
                (matrix[SPECIAL_1_INDEX, col] == 1)
            ):
                invalid = True
                break
        # In case the condition is broken, 
        # the combination is ignored
        if invalid:
            continue

        # Transposes the matrix for row-wise 
        # slice multiplication
        transposed = matrix.T
        weighted = transposed * slices_num
        # Sums all the values along the columns
        col_sums = weighted.sum(axis=1)
        # Calculates the standard deviation of 
        # the matrix
        std_val = stdev(col_sums)

        # If this value is better than the 
        # best case, then it is saved
        if std_val < min_std:
            min_std = std_val
            best_matrix = matrix.copy()

    # Loads the data with the train volumes info
    train_volumes = pd.read_csv("..\\splits\\volumes_info.csv")
    # Sets a column indicating it is from 
    # the train set
    train_volumes["origin_set"] = "train"

    # Loads the data with the test volumes info
    test_volumes = pd.read_csv("..\\splits\\volumes_info_test.csv")
    # Sets a column indicating it is from 
    # the test set
    test_volumes["origin_set"] = "test"

    # Combines both sets into a single DataFrame
    full_data = pd.concat([train_volumes, test_volumes], ignore_index=True)

    # Create a unified identifier column with shape "1_train"
    full_data["VolumeNumber"] = full_data["VolumeNumber"].astype(str) + "_" + full_data["origin_set"]

    # Createsa dictionary that contains 
    # a list of volumes for each fold
    folds = {i: [] for i in range(4)}
    # Initiates a set that 
    # contains all the used 
    # volumes
    used_indices = set()

    # Creates a matrix of 
    # devices and assign 
    # the value False to 
    # indicate that they 
    # are not special cases
    matrix_devices = [
        ('Cirrus', False),
        ('Spectralis', False),
        ('T-1000', False),
        ('T-2000', False)
    ]

    # Iterates through all the devices available
    for device_idx, (device_name, _) in enumerate(matrix_devices):
        # Gets the row that explains the device partition
        matrix_row = best_matrix[device_idx]
        # Gets a sub-DataFrame with the data of this device
        device_volumes = full_data[full_data['Device'] == device_name]
        # Shuffles the list to ensure the volumes are 
        # picked randomly
        device_volumes = device_volumes.sample(frac=1)

        # Iterates through all the five folds
        for fold_idx, num_samples in enumerate(matrix_row):
            # Counts the total number 
            # of samples
            count = int(num_samples)
            # Appends the 
            # selected volumes 
            # to a list
            selected = []

            # Iterates through all the volumes in 
            # of the said device
            for _, row in device_volumes.iterrows():
                # Gets the ID of 
                # the device
                row_id = row.name
                # Checks if it has not been 
                # used yet
                if row_id not in used_indices:
                    # In case it is unused, 
                    # appends it to the list
                    # of the volumes of this 
                    # split and to the list of
                    # seen volumes
                    selected.append(row_id)
                    used_indices.add(row_id)
                    # Whenever all the volumes 
                    # have been seen, it breaks
                    if len(selected) == count:
                        break

            # Gets the volumes selected to a list
            fold_ids = full_data.loc[selected, 'VolumeNumber'].tolist()
            # Assigns the list to the dictionary 
            # with the volumes identifiers
            folds[fold_idx].extend(fold_ids)

    # Handles the two special cases manually
    # Selects the data and shuffles them
    special_volumes = full_data[
        (full_data['Device'] == 'T-2000') & (full_data['SlicesNumber'] == 64)
    ].sample(frac=1)

    # Gets the folds to which it is expected 
    # to append
    special_1_fold = np.argmax(best_matrix[4])

    # Assigns each one to the target folder
    folds[special_1_fold].append(special_volumes.iloc[0]['VolumeNumber'])

    # Gets the information from the first fold of the train split
    fold_1 = pd.read_csv("..\\splits\\competitive_fold_selection.csv")["1"]
    fold_1 = [str(int(fold_1[k])) + "_train" for k in range(len(fold_1))] + [pd.NA] * (len(folds.get(0)) - len(fold_1))

    # Creates the output DataFrame with indexes from 0 to 4 as the names
    split_df = pd.DataFrame({"0": folds.get(0)})
    split_df["1"] = fold_1
    split_df["2"] = folds.get(1)
    split_df["3"] = folds.get(2)
    split_df["4"] = folds.get(3)

    # Saves the data to a CSV
    split_df.to_csv("..\\splits\\generation_4_fold_split.csv", index=False)

def splits_to_25d(split_path: str, test_fold: int=1):
    """
    This function is used to convert the splits 
    obtained with the functions used in this project 
    to the ones used in the PTL repository 
    (https://github.com/davidterroso/PTL). Resuming,
    in the current splits that are going to be read, 
    the OCT volumes are separated in multiple folds
    while in the second implementation, each slice
    is associated to each own split, even though 
    slices of the same volume can never be separated.
    In this function, we are converting the whole 
    volume to its constituting slices, associating 
    to it a couple of values that are relevant for
    the split used in the other project 
    
    Args:
        split_path (str): path to the split we want 
            to convert
        test_fold (int): number of the fold that 
            will be used in testing

    Returns:
        None
    """
    # Declares the path to the CSV file with the information of each 
    # slice obtained in the repository PTL
    slice_info_path=r'D:\PTL\RETOUCHdata\pre_processed\slice_gt.csv'
    # Loads the information of each slice
    slice_df = pd.read_csv(slice_info_path)
    
    # Loads the split that will be converted
    split_df = pd.read_csv(split_path, header=None)

    # Gets the number of folds
    num_folds = split_df.shape[1]

    # Creates a dictionary that will associate folds to the volumes
    fold_to_volumes = {
        i: [f"TRAIN{int(v):03d}" for v in split_df[i].dropna().astype(int)]
        for i in range(num_folds)
    }

    # Creates the directory in case it does 
    # not exist already
    makedirs("..\\splits_ptl", exist_ok=True)

    # Iterates through all the folds
    for i in range(num_folds):
        # Gets all the volumes that compose a single fold
        fold_volumes = fold_to_volumes[i]
        # Converts the slices that compose them to a single DataFrame
        val_df = slice_df[slice_df['image_name'].isin(fold_volumes)].copy()
        # Shuffles the DataFrame
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Saves the DataFrame to a CSV file
        val_out_path = f"..\\splits_ptl\\competitive_fold_selection_{i}_val.csv"
        val_df.to_csv(val_out_path, index=False)

    # Iterates through all the folds in the split
    for fold_idx in range(num_folds):
        # In case the fold is the 
        # one used in testing,
        # nothing happens
        if fold_idx == test_fold:
            continue

        # Uses all folds except the current fold and the fold used in testing
        include_folds = [f for f in range(num_folds) if f not in {fold_idx, test_fold}]
        
        # Initiates an empty list that will store 
        # the volumes used in training
        train_volumes = []
        # Iterates through all 
        # the included folds and 
        # adds to the list the 
        # volumes that belong to it 
        for f in include_folds:
            train_volumes.extend(fold_to_volumes[f])

        # Creates a DataFrame that will contain the information of the slices
        # that will be used in training and shuffles them
        train_df = slice_df[slice_df['image_name'].isin(train_volumes)].copy()
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Saves the results as a CSV file
        out_path = f"..\\splits_ptl\\competitive_fold_selection_{fold_idx}.csv"
        train_df.to_csv(out_path, index=False)
