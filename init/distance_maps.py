import numpy as np
from os import makedirs
from PIL import Image
from skimage.io import imread

def get_distance_maps(mask_path: str, ilm_class: int, bm_class: int, 
                      save_folder: str):
    """
    Function that receives as input the retinal layers mask, the 
    class of the internal limiting membrane (ILM) class, and the
    Bruch's membrane (BM) class

    Args:
        mask_path (str): path to the retinal layers mask
        ilm_class (int): class of the internal limiting membrane 
            (ILM)
        bm_class (int): class of the Bruch's membrane (ILM) 
        save_folder (str): folder to which it is expected to save
            the relative distance maps
    Returns:
        None
    """
    # Loads the image as a NumPy array
    mask = imread(mask_path)
    # Initiates the relative 
    # distance maps as a matrix 
    # of zeros 
    rm_map = np.zeros_like(mask)

    # Gets the image height
    mask_height = mask.shape[0]

    # Iterates through 
    # the columns that 
    # compose the image
    for col in range(mask.shape[1]):
        # Gets the values that 
        # make up the column
        column = mask[:,col]

        # Initiates a flag that indicates 
        # whether the ILM or BLM class is 
        # missing or not
        missing_class = False

        # Of those values, gets the ones that are 
        # ILM and the ones that are BM
        ilm_position = np.where(column == ilm_class)[0]
        bm_position = np.where(column == bm_class)[0]

        # Calculates the highest coordinate 
        # and lowest index of the point within 
        # the ILM class. In case there is none, 
        # sets the flag to True
        if ilm_position.size > 0:
            top_ilm_index = ilm_position[0] 
            top_ilm_coordinate = mask_height - 1 - top_ilm_index
        else:
            missing_class = True

        # Calculates the lowest coordinate 
        # and highest index of the point within 
        # the BM class. In case there is none, 
        # sets the flag to True
        if bm_position.size > 0:
            bottom_bm_index = bm_position[0] 
            bottom_bm_coordinate = mask_height - 1 - bottom_bm_index
        else: 
            missing_class = True

        # In case one of the classes is missing, 
        # sets all the values in the column to zero
        if missing_class:
            rm_map[:, col] = np.zeros_like(column)
        else:
            # Iterates through all the rows in the column
            for row in range(top_ilm_index, bottom_bm_index + 1):
                # In case the pixel is located within the classes, the 
                # relative distance map is calculated
                if row > top_ilm_index and row < bottom_bm_coordinate:
                    # Calculates the coordinate of the row
                    row_coordinate = mask_height - 1 - row
                    # Calculates the relative distance map value for this pixel
                    rm_map[row, col] = (row_coordinate - top_ilm_coordinate) / (top_ilm_coordinate - bottom_bm_coordinate)
                # If the pixel is not 
                # located between the 
                # classes it is set 
                # to zero
                else:
                    rm_map[row, col] = 0

    # Gets the path to save from the image and creates the folder in 
    # case it does not exist
    path_to_save = save_folder + "\\OCT_images\\segmentation\\rdms\\"
    makedirs(path_to_save, exist_ok=True)

    # Saves the image keeping the same name as the input image 
    # in a different folder
    path_to_save = path_to_save + mask_path.split["\\"][-1]
    Image.fromarray(rm_map).save(path_to_save)
