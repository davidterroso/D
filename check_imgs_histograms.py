import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

def plot_images_and_histograms(real_path, fake_path):
    # Load images
    real_img = io.imread(real_path)
    fake_img = io.imread(fake_path)

    # Compute histograms manually
    real_hist, _ = np.histogram(real_img.ravel(), bins=256, range=(0, 255))
    fake_hist, _ = np.histogram(fake_img.ravel(), bins=256, range=(0, 255))
    max_count = max(real_hist.max(), fake_hist.max())

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Display real image
    axs[0, 0].imshow(real_img, cmap='gray')
    axs[0, 0].set_title(f'Real Image\n{os.path.basename(real_path)}')
    axs[0, 0].axis('off')

    # Display fake image
    axs[0, 1].imshow(fake_img, cmap='gray')
    axs[0, 1].set_title(f'Fake Image\n{os.path.basename(fake_path)}')
    axs[0, 1].axis('off')

    # Real image histogram
    axs[1, 0].hist(real_img.ravel(), bins=256, range=(0, 255), color='blue', alpha=0.7)
    axs[1, 0].set_xlim(0, 255)
    axs[1, 0].set_ylim(0, max_count)
    axs[1, 0].set_title('Real Image Histogram')

    # Fake image histogram
    axs[1, 1].hist(fake_img.ravel(), bins=256, range=(0, 255), color='red', alpha=0.7)
    axs[1, 1].set_xlim(0, 255)
    axs[1, 1].set_ylim(0, max_count)
    axs[1, 1].set_title('Fake Image Histogram')

    plt.tight_layout()
    plt.show()

# Example paths
real_image_path = r'D:\DavidTerroso\Images\OCT_images\generation\slices_resized\Spectralis_TEST026_036.tiff'
fake_image_path = r'D:\DavidTerroso\Images\OCT_images\generation\predictions\Run016\Spectralis_TEST026_036_generated.tiff'

# Call the function
plot_images_and_histograms(real_image_path, fake_image_path)
