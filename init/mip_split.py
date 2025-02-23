import pandas as pd
from math import ceil, floor
from ortools.linear_solver import pywraplp

# Load the dataset using Pandas
df = pd.read_csv("..\\splits\\volumes_info.csv")

# Declares the number of folds
num_folds = 5
# Extracts the number of volumes
num_volumes = len(df)
# Calculates the number of volumes per fold
fold_sizes = [num_volumes // num_folds + (1 if i < num_volumes % num_folds else 0) for i in range(num_folds)]
# Declares the name of the three relevant classes of voxels
voxel_cols = ["IRF", "SRF", "PED"]

# Creates the solver
# SCIP: Solving Constraint Integer Programs
solver = pywraplp.Solver.CreateSolver("SCIP")

# Boolean decision variable x[v, f] that indicates whether the volume v is in fold f  
x = {}
# Iterates through the volumes, starting in 1
for v in range(num_volumes):
    # Iterates through all the folds
    for f in range(num_folds):
        # Initializes the variables in the solver
        x[v, f] = solver.BoolVar(f"x_{v}_{f}")

# Declares the first constraint, so that each volume is assigned only to one fold
for v in range(num_volumes):
    solver.Add(sum(x[v, f] for f in range(num_folds)) == 1)

# Declares the second constraint, so that each fold has the same number of volumes
for f in range(num_folds):
    solver.Add(sum(x[v, f] for v in range(num_volumes)) == fold_sizes[f])

# Declares the third constraint, so that the number of voxels per class 
# is at least similar between folds
# Starts by finding what are the biggest voxel counts in each class
max_voxel_value = max(df[voxel_cols].sum())

# Create a variable that will track max deviation
max_deviation = solver.NumVar(0, solver.infinity(), "max_deviation")
# Counts the total number of voxels in a volume per class
voxel_sums = [{f: solver.Sum(df[voxel_cols[i]][v] * x[v, f] for v in range(num_volumes)) for f in range(num_folds)} for i in range(3)]
# Iterates through each class
for i in range(3):
    # Calculates the expected value per fold of each classe's voxel count
    mean_voxel = solver.Sum(voxel_sums[i][f] for f in range(num_folds)) / num_folds
    # Iterates through the number of folds 
    for f in range(num_folds):
        # Restricts that the difference between the total number of
        # voxels of a certain class in a fold must be, in absolute 
        # value, smaller than the maximum deviation allowed
        solver.Add(voxel_sums[i][f] - mean_voxel <= max_deviation)
        solver.Add(mean_voxel - voxel_sums[i][f] <= max_deviation)

# Declares the fourth constraint that ensures 
# balancement between vendors across all folds
vendor_groups = df.groupby("Vendor").groups
for vendor, indices in vendor_groups.items():
    vendor_folds = {f: solver.Sum(x[v, f] for v in indices) for f in range(num_folds)}
    for f in range(num_folds):
        solver.Add(vendor_folds[f] >= floor(len(indices) / num_folds))
        solver.Add(vendor_folds[f] <= ceil(len(indices) / num_folds))

# Declares that it is desired 
# to minimize the max deviation
solver.Minimize(max_deviation)

# Starts solving
status = solver.Solve()

# Outputs the solution
if status == solver.OPTIMAL or status == solver.FEASIBLE:
    # Initiates the dictionary that will handle the fold 
    # to which each volume will belong
    # Create a dictionary where each key (fold) gets a list of assigned volume IDs
    fold_assignments = {f: [] for f in range(num_folds)}
    # Iterates through the volumes and folds to append 
    # the volumes whose solution value is greater than 0.5
    for v in range(num_volumes):
        for f in range(num_folds):
            if x[v, f].solution_value() > 0.5:
                # Assuming the volume ID is stored in a column called "VolumeNumber"
                fold_assignments[f].append(df.iloc[v]["VolumeNumber"])

    # Find the maximum number of volumes assigned to any fold
    max_len = max(len(lst) for lst in fold_assignments.values())

    # Pad each fold's list with empty strings so 
    # that all lists have equal length
    for f in range(num_folds):
        fold_assignments[f] += [""] * (max_len - len(fold_assignments[f]))

    # Create a new DataFrame with fold numbers as columns
    fold_df = pd.DataFrame(fold_assignments)

    # Name the columns in the DataFrame
    fold_df.columns = [str(f) for f in range(num_folds)]

    # Save the new DataFrame to CSV
    fold_df.to_csv("..\\splits\\mip_fold_selection.csv", index=False)
    print("Fold assignment has been completed.")

else:
    print("No feasible solution found.")
