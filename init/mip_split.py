import pandas as pd
import os
import sys
from math import ceil, floor

# Supresses the messages informing the 
# loading of DLL files that come with 
# the pywraplp import
sys.stdout = open(os.devnull, "w")

from ortools.linear_solver import pywraplp

# Restores the stderr definitions after the 
# import is completed
sys.stdout = sys.__stdout__

def mip_k_fold_split(k: int=5):
    """
    
    Args:
        k (int): number of folds this split must have

    Returns:
        None
    """
    # Load the information of each volume using Pandas,
    # containing the number of the volume, the vendor 
    # of the device used to obtain the device and the 
    # total number of voxels per class, in three classes
    # which are IRF, SRF, and PED
    df = pd.read_csv("..\\splits\\volumes_info.csv")

    # Extracts the number of volumes to split
    num_volumes = len(df)
    # Calculates the number of volumes per fold
    fold_sizes = [num_volumes // k + (1 if i < num_volumes % k else 0) for i in range(k)]
    # Declares the name of the three relevant classes of voxels
    voxel_cols = ["IRF", "SRF", "PED"]
    # Gets the list of all vendors in the dataset
    vendors = df["Vendor"].unique()
    
    # Creates the solver
    # SCIP: Solving Constraint Integer Programs
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Enables verbose logging to see solver iterations
    solver.SetSolverSpecificParametersAsString("display/verblevel = 3")
    solver.EnableOutput()

    # Boolean decision variable x[v, f] that 
    # indicates whether the volume v is in fold f  
    x = {}
    # Iterates through the volumes
    for volume in range(num_volumes):
        # Iterates through all the folds
        for fold in range(k):
            # Initializes the variables in the solver
            x[(volume, fold)] = solver.BoolVar(f"x_{volume}_{fold}")

    # Declares the first constraint, so that each volume is assigned only to one fold
    for volume in range(num_volumes):
        solver.Add(sum(x[volume, fold] for fold in range(k)) == 1)

    # Declares the second constraint, so that each fold has the same number of volumes
    for fold in range(k):
        solver.Add(sum(x[volume, fold] for volume in range(num_volumes)) == fold_sizes[fold])

    # Declares the fourth constraint that ensures 
    # that fold has a similar number of volumes per fold
    vendor_groups = df.groupby("Vendor").groups
    for vendor, indices in vendor_groups.items():
        vendor_folds = {fold: solver.Sum(x[volume, fold] for volume in indices) for fold in range(k)}
        for fold in range(k):
            solver.Add(vendor_folds[fold] >= floor(len(indices) / k))
            solver.Add(vendor_folds[fold] <= ceil(len(indices) / k))


    # Calculates what is the expected voxel 
    # counts for each vendor and each class
    # Initiates a dictionary that will contain 
    # the expected value that each vendor
    expected_voxel_counts = {}
    # Iterates through each class in each vendor
    for vendor in vendors:
        for column, voxel_col in enumerate(voxel_cols):
            # Calculates the mean value for each class in each vendor
            expected_voxel_counts[(vendor, column)] = sum(df[df["Vendor"] == vendor][voxel_col]) / k

    # Create a dictionary that converts the VolumeNumber to the volume index 
    # in the DataFrame
    volume_to_index = {df.loc[i, "VolumeNumber"]: i for i in range(len(df))}
    # Calculates the values obtained in each 
    # fold, for each vendor, per class
    voxel_sums = {}
    for vendor in vendors:
        for column, voxel_col in enumerate(voxel_cols):
            for fold in range(k):
                indices = df[df["Vendor"] == vendor]["VolumeNumber"].tolist()
                indices = [volume_to_index[v] for v in indices]
                # The sum is obtained as the total number of voxels of the class that is being considered in a fold
                # This is done by multiplying the value of voxels in a specific volume by the boolean variable that 
                # indicates whether the said volume index is present in the fold that is being analyzed 
                voxel_sums[(vendor, column, fold)] = solver.Sum(df.loc[index, voxel_col] * x[index, fold] for index in indices)

    # Calculates the difference between the 
    # obtained values and the values expected
    # Initiates the dictionary that will store
    # these differences
    diff = {}
    # Iterates through the 
    # vendors, the columns,
    # and the folds
    for vendor in vendors:
        for column in range(len(voxel_cols)):
            for fold in range(k):
                # Difference is the value aimed to minimized 
                diff[vendor, column, fold] = solver.NumVar(0, solver.infinity(), f"diff_{volume}_{column}_{fold}")
                solver.Add(diff[vendor, column, fold] >= voxel_sums[(vendor, column, fold)] - expected_voxel_counts[(vendor, column)])
                solver.Add(diff[vendor, column, fold] >= expected_voxel_counts[(vendor, column)] - voxel_sums[(vendor, column, fold)])

    # Declares that it is desired to minimize the total sum of differences, across all the folds in all vendors and classes
    solver.Minimize(solver.Sum(diff[volume, column, fold] for volume in vendors for column in range(len(voxel_cols)) for fold in range(k)))

    # Starts solving
    status = solver.Solve()

    # Outputs the solution
    if status == solver.OPTIMAL or status == solver.FEASIBLE:
        # Initiates the dictionary that will handle the fold 
        # to which each volume will belong
        # Create a dictionary where each key (fold) gets a list of assigned volume IDs
        fold_assignments = {fold: [] for fold in range(k)}
        # Iterates through the volumes and folds to append 
        # the volumes whose solution value is greater than 0.5
        for volume in range(num_volumes):
            for fold in range(k):
                if x[volume, fold].solution_value() > 0.5:
                    fold_assignments[fold].append(df.iloc[volume]["VolumeNumber"])

        # Find the maximum number of volumes assigned to any fold
        max_len = max(len(lst) for lst in fold_assignments.values())

        # Pad each fold's list with empty strings so 
        # that all lists have equal length
        for fold in range(k):
            fold_assignments[fold] += [""] * (max_len - len(fold_assignments[fold]))

        # Create a new DataFrame with fold numbers as columns
        fold_df = pd.DataFrame(fold_assignments)

        # Name the columns in the DataFrame
        fold_df.columns = [str(fold) for fold in range(k)]

        # Saves the new DataFrame to CSV
        fold_df.to_csv("..\\splits\\mip_fold_selection.csv", index=False)
        print("Fold assignment has been completed.")

    else:
        print("No feasible solution found.")

if __name__ == "__main__":
    mip_k_fold_split()
