from pandas import DataFrame, ExcelWriter, read_csv, Series
from os.path import exists, join
from os import listdir

def runs_resume(prefix: str, starting_run: int, 
                ending_run: int, folder: str):
    """
    Function used to resume all the results from 
    a run in a single CSV file making it easier 
    to load in an Excel file

    Args:
        prefix (str): prefix of the files, 
            indicating whether it is a run or 
            an iteration that is being evaluated
        starting_run (int): first run that will 
            be evaluated
        ending_run (int): last run that will be 
            evaluated
        folder (str): path to the folder in which 
            the results are located
    
    Return:
        None
    """

    assert prefix in ["Run", "Iteration"],\
    """ Invalid Prefix: select "Run" or "Iteration" """

    # Declares the name of the files that will be handled,
    # depending on which the images have been resized or not
    if prefix == "Run":
        files_names = ["vendor_dice", 
                    "vendor_dice_wfluid", 
                    "vendor_dice_wofluid", 
                    "class_dice", 
                    "class_dice_wfluid", 
                    "class_dice_wofluid", 
                    "fluid_dice"]
        files_names_resized = ["vendor_dice_resized", 
                            "vendor_dice_resized_wfluid", 
                            "vendor_dice_resized_wofluid", 
                            "class_dice_resized", 
                            "class_dice_resized_wfluid", 
                            "class_dice_resized_wofluid", 
                            "fluid_dice_resized"]
    else:
        files_names = ["vendors_results", 
                    "vendors_results_wfluid", 
                    "vendors_results_wofluid", 
                    "classes_results", 
                    "classes_results_wfluid", 
                    "classes_results_wofluid", 
                    "fluids_results"]
        files_names_resized = ["vendors_results_resized", 
                            "vendors_results_resized_wfluid", 
                            "vendors_results_resized_wofluid", 
                            "classes_results_resized", 
                            "classes_results_resized_wfluid", 
                            "classes_results_resized_wofluid", 
                            "fluids_results_resized"]
    # Iterates through all the runs that were indicated
    for run_number in range(starting_run, ending_run + 1):
        # Sets the name of the files 
        # according to the number of 
        # the run. The files with a 
        # run number higher than 23 
        # have been extracted in 
        # resized images
        if prefix == "Run":
            if run_number < 23:
                files_to_load = files_names
            else:
                files_to_load = files_names_resized            
        else:
            if run_number < 5:
                files_to_load = files_names
            else:
                files_to_load = files_names_resized

        # Changes the number to match the one 
        # used in the files (e.g. 1 -> 001)
        if prefix == 'Run':
            run_number = str(run_number).zfill(3)
        else:
            run_number = str(run_number)
        # Initiates the results 
        # with the number of the run 
        results = [prefix + run_number]
        # Declares the path to each file
        files_paths = [folder + prefix + run_number + "_" + fn + 
                       ".csv" for fn in files_to_load]
        # Assumes that every file exists
        files_exist = True
        # Checks if every file exists
        for fp in files_paths:
            # In case at least one 
            # file path does not exist, 
            # the flag is changed to 
            # false and does not read
            # procede for that run number
            if not exists(fp):
                files_exist = False
        # In case all 
        # files exist, 
        # continues
        if files_exist:
            # Reads the files that contains the 
            # Dice for the vendors and fluids
            vendor_dice = read_csv(files_paths[0])
            vendor_dice_wfluid = read_csv(files_paths[1])
            vendor_dice_wofluid = read_csv(files_paths[2])
            # Iterates through all the vendors and fluids and 
            # appends the values to the results array
            if prefix == "Run": iter_vendor_dice = vendor_dice["Vendor"]
            else: iter_vendor_dice = vendor_dice["vendors"]
            for idx, vendor in enumerate(iter_vendor_dice):
                for fluid in vendor_dice.columns[1:]:
                    results.append(vendor_dice.at[idx, fluid]) 
                    results.append(vendor_dice_wfluid.at[idx, fluid])
                    results.append(vendor_dice_wofluid.at[idx, fluid])

            # Loads the CSV files that contain the Dice 
            # coefficient for each class
            class_dice = read_csv(files_paths[3])
            class_dice_wfluid = read_csv(files_paths[4])
            class_dice_wofluid = read_csv(files_paths[5])

            # Iterates through the fluids in the class files and 
            # appends them to the results list
            for fluid in class_dice.columns:
                results.append(class_dice.at[0, fluid])
                results.append(class_dice_wfluid.at[0, fluid])
                results.append(class_dice_wofluid.at[0, fluid])

            # Reads the file that contains the Dice 
            # resulting from the binarization of 
            # the fluid masks
            fluid_dice = read_csv(files_paths[6], header=None, index_col=False).iloc[1].tolist()
            
            # Iterates through the values and 
            # appends them to the list of results
            for value in fluid_dice:
                results.append(value)

            # In case Runs are being handled, 
            # the number of decimal cases 
            # will be reduced
            if prefix == "Run":
                # The first value is unchanged 
                # because it is a string
                results_simple = [results[0]]
                # Iterates through the results
                for result in results[1:]:
                    # Splits the value by separating the 
                    # part that corresponds to the mean 
                    # and to the standard deviation  
                    mean_part, std_part = result.split("(")
                    # Removes any remaining whitespaces 
                    # or brackets and converts the values 
                    # to float
                    mean = float(mean_part.strip())
                    std = float(std_part.strip(" )"))
                    # Keeps only three decimal places of the values
                    results_simple.append(f"{mean:.3f} ({std:.3f})")
                results = results_simple

            # Saves the results as a CSV file that will contain the 
            # Dice values in the following order:
            # "Run", 
            # "CirrusIRF", "CirrusIRF_wfluid", "CirrusIRF_wofluid", 
            # "CirrusSRF", "CirrusSRF_wfluid", "CirrusSRF_wofluid", 
            # "CirrusPED", "CirrusPED_wfluid", "CirrusPED_wofluid", 
            # "SpectralisIRF", "SpectralisIRF_wfluid", "SpectralisIRF_wofluid", 
            # "SpectralisSRF", "SpectralisSRF_wfluid", "SpectralisSRF_wofluid", 
            # "SpectralisPED", "SpectralisPED_wfluid", "SpectralisPED_wofluid", 
            # "TopconIRF", "TopconIRF_wfluid", "TopconIRF_wofluid", 
            # "TopconSRF", "TopconSRF_wfluid", "TopconSRF_wofluid", 
            # "TopconPED", "TopconPED_wfluid", "TopconPED_wofluid", 
            # "IRF", "IRF_wfluid", "IRF_wofluid", 
            # "SRF", "SRF_wfluid", "SRF_wofluid", 
            # "PED", "PED_wfluid", "PED_wofluid", 
            # "Fluid", "Fluid_wfluid", "Fluid_wofluid"
            # This will be saved under the name of Run001_resumed.csv, as an example
            Series(results).to_frame().T.to_csv(f".\\results\\{prefix + run_number}_resumed.csv")

def combine_csvs_to_excel(folder_path, output_excel_path):
    """
    Combines the different CSV files in a single excel, 
    separated in two sheets: one contains the information 
    of the runs while the other contains the information 
    of the iterations
    
    Args:
        folder_path (str): path to the folder that 
            contains all the results
        output_excel_path (str): path to the folder in 
            which the resulting excel file will be placed

    Returns:
        None
    """
    # Initiates the data list that will contain the 
    # different DataFrames from the runs and iterations
    run_rows = []
    iteration_rows = []

    # Iterates through the files in the 
    # folder that is being iterated
    for filename in listdir(folder_path):
        # Only reads the ones that end with 
        # _resumed.csv
        if filename.endswith("_resumed.csv"):
            # Reads the CSV file as a DataFrame
            filepath = join(folder_path, filename)
            df = read_csv(filepath, header=None, index_col=False)

            # Gets the data from the CSV file
            row = df.iloc[-1].tolist()

            # Appends the results to the 
            # respective list
            if filename.startswith("Run"):
                run_rows.append(row)
            elif filename.startswith("Iteration"):
                iteration_rows.append(row)

    # Combines all the DataFrames in a single one for each group
    run_df = DataFrame(run_rows)
    iteration_df = DataFrame(iteration_rows)

    # Saves the DataFrames to separate Excel sheets
    with ExcelWriter(output_excel_path) as writer:
        run_df.to_excel(writer, sheet_name="Runs", index=False, header=False)
        iteration_df.to_excel(writer, sheet_name="Iterations", index=False, header=False)

runs_resume(starting_run=1, ending_run=63, folder=".\\results\\", prefix="Run")
runs_resume(starting_run=1, ending_run=19, folder=".\\results\\", prefix="Iteration")
combine_csvs_to_excel(folder_path=".\\results\\", output_excel_path=".\\results\\combined_output.xlsx")
