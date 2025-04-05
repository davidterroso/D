from pandas import read_csv, Series
from os.path import exists

starting_run = 1
ending_run = 56
folder = ".\\results\\"

def runs_resume(starting_run, ending_run, folder):
    """
    Function used to resume all the results from 
    a run in a single CSV file making it easier 
    to load in an Excel file

    Args:
        starting_run (int): first run that will 
            be evaluated
        ending_run (int): last run that will be 
            evaluated
        folder (str): path to the folder in which 
            the results are located
    
    Return:
        None
    """
    # Declares the name of the files that will be handled, depending on which 
    # the images have been resized or not
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

    # Iterates through all the runs that were indicated
    for run_number in range(starting_run, ending_run + 1):
        # Sets the name of the files 
        # according to the number of 
        # the run. The files with a 
        # run number higher than 23 
        # have been extracted in 
        # resized images
        if run_number < 23:
            files_to_load = files_names
        else:
            files_to_load = files_names_resized
        # Changes the number to match the one 
        # used in the files (e.g. 1 -> 001)
        run_number = str(run_number).zfill(3)
        # Initiates the results 
        # with the number of the run 
        results = [run_number]
        # Declares the path to each file
        files_paths = [folder + "Run" + run_number + "_" + fn + 
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
            for idx, vendor in enumerate(vendor_dice["Vendor"]):
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
            fluid_dice = read_csv(files_paths[6])
            
            # Iterates through the values and 
            # appends them to the list of results
            for value in fluid_dice.values():
                results.append(value)

            # Saves the results as a CSV file that will contain the Dice values in the 
            # following order:
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
            Series(results).to_frame().T.to_csv(f".\\results\\Run{run_number}_resumed.csv")
