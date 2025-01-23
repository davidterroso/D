import tkinter as tk
from tkinter import messagebox
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.rank import entropy
from skimage.io import imread
from skimage.util import img_as_float
from skimage.morphology import disk
from paths import IMAGES_PATH

def showImages(vendor, volume, slice_num):
    vendor = vendor.lower().capitalize()
    volume = str(volume).zfill(3)
    slice_num = str(slice_num).zfill(3)
    slice_path = IMAGES_PATH + "\\OCT_images\\segmentation\\slices\\uint8\\" + vendor + "_TRAIN" + volume + "_" + slice_num + ".tiff"
    masks_path = IMAGES_PATH + "\\OCT_images\\segmentation\\masks\\uint8\\" + vendor + "_TRAIN" + volume + "_" + slice_num + ".tiff"
    roi_path = IMAGES_PATH + "\\OCT_images\\segmentation\\roi\\uint8\\" + vendor + "_TRAIN" + volume + "_" + slice_num + ".tiff"
    threshold = 1e-2

    original_slice = imread(slice_path)
    mask = imread(masks_path)
    mask = mask * 3 / 255
    roi = imread(roi_path)
    roi = roi / 255

    mask = np.array(mask, dtype=np.float32)
    mask[mask == 0] = np.nan

    roi = np.array(roi, dtype=np.float32)
    roi[roi == 0] = np.nan

    slice = img_as_float(original_slice.astype(np.float32) / 128. - 1.)
    slice = entropy(slice, disk(15))
    slice = slice / (np.max(slice) + 1e-16)
    entropy_mask = np.asarray(slice > threshold, dtype=np.uint8)
    entropy_mask = np.array(entropy_mask, dtype=np.float32)
    entropy_mask[entropy_mask == 0] = np.nan

    fig, ax_array = plt.subplots(2, 2, figsize=(10, 8))

    # Display original B-Scan
    ax_array[0, 0].imshow(original_slice, cmap=plt.cm.gray)
    ax_array[0, 0].set_title("B-Scan")
    ax_array[0, 0].axis("off")

    # Define colormap and normalization for mask
    colors = ["red", "green", "blue"]  # Black for background, red, green, blue for fluids
    cmap = mcolors.ListedColormap(colors)
    bounds = [1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    fluids = ["IRF", "SRF", "PED"]

    # Display B-Scan with fluid masks
    ax_array[0, 1].imshow(original_slice, cmap=plt.cm.gray)
    img = ax_array[0, 1].imshow(mask, alpha=0.7, cmap=cmap, norm=norm)
    ax_array[0, 1].set_title("B-Scan with Fluid Masks")
    ax_array[0, 1].axis("off")

    # Add color bar for fluid masks
    cbar = fig.colorbar(img, ax=ax_array[0, 1], ticks=[1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(fluids)  # Assign labels to each value
    cbar.set_label("Fluid Types")

    # Define colormap and normalization for mask
    colors = ["purple"]  # Black for background, red, green, blue for fluids
    cmap = mcolors.ListedColormap(colors)
    bounds = [0,1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    entropy_label = ["Entropy"]

    # Display B-Scan with entropy mask
    ax_array[1, 0].imshow(original_slice, cmap=plt.cm.gray)
    img = ax_array[1, 0].imshow(entropy_mask, alpha=0.7, cmap=cmap)
    ax_array[1, 0].set_title("B-Scan with Entropy Mask")
    ax_array[1, 0].axis("off")

    # Add color bar for fluid masks
    cbar = fig.colorbar(img, ax=ax_array[1, 0], ticks=[0.5])
    cbar.ax.set_yticklabels(entropy_label)  # Assign labels to each value
    cbar.set_label("Entropy > 1e-2")

    # Display B-Scan with ROI mask
    ax_array[1, 1].imshow(original_slice, cmap=plt.cm.gray)
    ax_array[1, 1].imshow(roi, alpha=0.7, cmap=plt.cm.summer)
    ax_array[1, 1].set_title("B-Scan with ROI Mask")
    ax_array[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

class VendorInputApp:
    def __init__(self, master):
        self.master = master
        master.title("Visualize Scans")
        master.geometry("350x250")  # Adjusted for extra space

        frame = tk.Frame(master)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        # Vendor Input
        tk.Label(frame, text="Vendor:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.vendor_entry = tk.Entry(frame)
        self.vendor_entry.grid(row=0, column=1, padx=10, pady=5)

        # Volume Input
        tk.Label(frame, text="Volume:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.volume_entry = tk.Entry(frame)
        self.volume_entry.grid(row=1, column=1, padx=10, pady=5)

        # Slice Input
        tk.Label(frame, text="Slice:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.slice_entry = tk.Entry(frame)
        self.slice_entry.grid(row=2, column=1, padx=10, pady=5)

        # Visualize Button
        submit_button = tk.Button(frame, text="Visualize", command=self.submit)
        submit_button.grid(row=5, column=0, columnspan=2, pady=10)

    def submit(self):
        try:
            vendor = self.vendor_entry.get()
            volume = int(self.volume_entry.get())
            slice_val = int(self.slice_entry.get())

            # Validate inputs
            if not vendor:
                tk.messagebox.showerror("Error", "Vendor cannot be empty")
                return

            showImages(vendor, volume, slice_val)

        except ValueError:
            messagebox.showerror("Error", "Volume and Slice must be integers")

if __name__ == "__main__":
    root = tk.Tk()
    app = VendorInputApp(root)
    root.mainloop()
