import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
from os.path import exists
from skimage.filters.rank import entropy
from skimage.io import imread
from skimage.util import img_as_float
from skimage.morphology import disk
from paths import IMAGES_PATH

def showImages(original_slice, mask, roi):
    # Defines the threshold used to
    # calculate the entropy masks
    threshold = 1e-2

    # Declares the values that are zero in
    # the mask as NaN so that it does not
    # appear in the image shown
    mask = np.array(mask, dtype=np.float32)
    mask[mask == 0] = np.nan

    # Declares the values that are zero in
    # the ROI as NaN so that it does not
    # appear in the image shown
    roi = np.array(roi, dtype=np.float32)
    roi[roi == 0] = np.nan

    # Slice is converted to float to calculate the 
    # entropy values, which are decimal 
    slice = img_as_float(original_slice.astype(np.float32) / 128. - 1.)
    slice = entropy(slice, disk(15))
    # Slice values are normalized and only 
    # those greater than the threshold are kept
    slice = slice / (np.max(slice) + 1e-16)
    entropy_mask = np.asarray(slice > threshold, dtype=np.uint8)
    entropy_mask = np.array(entropy_mask, dtype=np.float32)
    # Values labeled as zero are changed to NaN
    # to prevent them from showing in the displayed images
    entropy_mask[entropy_mask == 0] = np.nan

    # Initializes the subplots
    fig, ax_array = plt.subplots(2, 2, figsize=(10, 8))

    # The first subplot only displays the original B-Scan
    ax_array[0, 0].imshow(original_slice, cmap=plt.cm.gray)
    # Declares the name of the subplot
    ax_array[0, 0].set_title("B-Scan")
    # Removes unnecessary axes
    ax_array[0, 0].axis("off")

    # For the second subplot the colors of the
    # labels "IRF", "SRF", and "PED" are selected
    # Red for IRF, green for SRF, and blue for PED
    fluid_colors = ["red", "green", "blue"]
    fluid_cmap = mcolors.ListedColormap(fluid_colors)
    # Declares in which part of the color bar each
    # label is going to be placed
    fluid_bounds = [1, 2, 3, 4]
    # Normalizes the color map according to the 
    # bounds declared.
    fluid_norm = mcolors.BoundaryNorm(fluid_bounds, fluid_cmap.N)
    fluids = ["IRF", "SRF", "PED"]

    # Displays the B-Scan with the fluid masks according
    # to the selected colors
    ax_array[0, 1].imshow(original_slice, cmap=plt.cm.gray)
    fluids_img = ax_array[0, 1].imshow(mask, alpha=0.3, cmap=fluid_cmap, norm=fluid_norm)
    # Title of the image shown
    ax_array[0, 1].set_title("B-Scan with Fluid Masks")
    # Removes unnecessary axes
    ax_array[0, 1].axis("off")

    # Creates the color bar and adds it to the image 
    # with the respective ticks
    fluids_cbar = fig.colorbar(fluids_img, ax=ax_array[0, 1], ticks=[1.5, 2.5, 3.5])
    # Assigns a label to each fluid
    fluids_cbar.ax.set_yticklabels(fluids)
    # Labels the color bar
    fluids_cbar.set_label("Fluid Types")

    # Defines the color of the mask used for
    # the entropy mask as purple
    entropy_color = ["purple"] 
    entropy_cmap = mcolors.ListedColormap(entropy_color)
    # Declares the bound of the colors used 
    # and normalizes the colors according to it
    entropy_bounds = [0,1]
    entropy_norm = mcolors.BoundaryNorm(entropy_bounds, entropy_cmap.N)
    # Color label
    entropy_label = ["Entropy"]

    # Displays the B-Scan with the entropy mask
    ax_array[1, 0].imshow(original_slice, cmap=plt.cm.gray)
    entropy_img = ax_array[1, 0].imshow(entropy_mask, alpha=0.3, cmap=entropy_cmap, norm=entropy_norm)
    # Declares the title of the subplot
    ax_array[1, 0].set_title("B-Scan with Entropy Mask")
    # Removes unnecessary axes
    ax_array[1, 0].axis("off")

    # Creates the color bar for the entropy mask
    entropy_cbar = entropy_img.colorbar(entropy_img, ax=ax_array[1, 0])
    entropy_cbar.set_label("Entropy > 0.01")

    # Declares the color 
    # of the ROI mask
    roi_colors = ["cyan"]

    # Creates the color map for the ROI image
    roi_cmap = mcolors.ListedColormap(roi_colors)
    # Declares the bounds 
    # of the color map
    roi_bounds = [0,1]
    # Normalizes the color map according to the declared
    # bounds
    roi_norm = mcolors.BoundaryNorm(roi_bounds, roi_cmap.N)
    # Names the label 
    # that is going to
    # be displayed
    roi_label = ["ROI"]

    # Displays the B-Scan with the ROI mask
    ax_array[1, 1].imshow(original_slice, cmap=plt.cm.gray)
    roi_img = ax_array[1, 1].imshow(roi, alpha=0.3, cmap=roi_cmap, norm=roi_norm)
    # Declares the title of the subplot
    ax_array[1, 1].set_title("B-Scan with ROI Mask")
    # Removes unnecessary axes
    ax_array[1, 1].axis("off")

    # Add color bar for the ROI region
    cbar = fig.colorbar(roi_img, ax=ax_array[1, 1])
    # Labels the color bar
    cbar.set_label("ROI Mask")

    # Shows the resulting image
    plt.show()

class VendorInputApp:
    def __init__(self, master):
        """
        Initiates the UI and asks for the vendor, volume, 
        and slice 

        Args:
            "self": the VendorInputApp object
            "master": the intialized window
        
        Return:
            None
        """
        # Initiates the window with the desired
        # name and size
        self.master = master
        master.title("Visualize Scans")
        master.geometry("350x250")

        # Centers the units that constitute the interface
        frame = tk.Frame(master)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        # Receives the input declaring which vendor the volume is from
        tk.Label(frame, text="Vendor:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.vendor_entry = tk.Entry(frame)
        self.vendor_entry.grid(row=0, column=1, padx=10, pady=5)

        # Receives the input declaring which volume the slice is from
        tk.Label(frame, text="Volume:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.volume_entry = tk.Entry(frame)
        self.volume_entry.grid(row=1, column=1, padx=10, pady=5)

        # Receives the input declaring which slice of the volume is going to be visualized
        tk.Label(frame, text="Slice:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.slice_entry = tk.Entry(frame)
        self.slice_entry.grid(row=2, column=1, padx=10, pady=5)

        # Visualize button
        visualize_button = tk.Button(frame, text="Visualize", command=self.visualize)
        visualize_button.grid(row=5, column=0, columnspan=2, pady=10)

    def visualize(self):
        """
        Calls the function that exhibits the images and calls the 
        errors that incur along the process

        Args:
            self: the VendorInputApp object

        Return:
            None
        """
        try:
            # Gets the inputs from the window
            vendor = self.vendor_entry.get()
            volume = int(self.volume_entry.get())
            slice_val = int(self.slice_entry.get())

            # Checks if the inputs were given
            if not (vendor or volume or slice_val) :
                messagebox.showerror("Error", "Arguments cannot be empty")
                return

            vendor = vendor.lower().capitalize()
            volume = str(volume).zfill(3)
            slice_num = str(slice_num).zfill(3)
            slice_path = IMAGES_PATH + "\\OCT_images\\segmentation\\slices\\uint8\\" + vendor + "_TRAIN" + volume + "_" + slice_num + ".tiff"
            masks_path = IMAGES_PATH + "\\OCT_images\\segmentation\\masks\\uint8\\" + vendor + "_TRAIN" + volume + "_" + slice_num + ".tiff"
            roi_path = IMAGES_PATH + "\\OCT_images\\segmentation\\roi\\uint8\\" + vendor + "_TRAIN" + volume + "_" + slice_num + ".tiff"
        
            if not (exists(slice_path) or exists(masks_path) or exists(roi_path)) :
                messagebox.showerror("Error", "The slice requested does not exist.")
                return

            original_slice = imread(slice_path)
            mask = imread(masks_path)
            mask = mask * 3 / 255
            roi = imread(roi_path)
            roi = roi / 255

            # Calls the function responsible for
            # showing the images
            showImages(original_slice, mask, roi)

        # Shows an error message in case the values given are not integers
        except ValueError:
            messagebox.showerror("Error", "Volume and Slice must be integers")

if __name__ == "__main__":
    # Creates the main window for the interface
    root = tk.Tk()
    VendorInputApp(root)
    root.mainloop()
