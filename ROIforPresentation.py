from os import walk, makedirs
from os.path import isfile, exists
from shutil import rmtree
from PIL import Image
from skimage.io import imread
from skimage.util import img_as_float
from skimage.morphology import disk, binary_closing
from skimage.filters.rank import entropy
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

threshold = 1e-2

save_path = "C:\\Users\\david\\Desktop\\test\\"
slice_path = "D:\\D\\OCT_images\\segmentation\\slices\\uint8\\Cirrus_TRAIN001_058.tiff"
mask_path = "D:\\D\\OCT_images\\segmentation\\masks\\uint8\\Cirrus_TRAIN001_058.tiff"

original_slice = imread(slice_path)
slice = img_as_float(original_slice.astype(np.float32) / 128. - 1.)
original_mask = imread(mask_path)

slice = entropy(slice, disk(15))
slice = slice / (np.max(slice) + 1e-16)
slice_ = np.asarray(slice > threshold, dtype=np.uint8)
slice = np.bitwise_or(slice_, original_mask)
selem = disk(55)
slice = binary_closing(slice, footprint=selem)

h, w = slice.shape
rnge = list()
for x in range(0, w):
    col = slice[:, x]
    col = np.nonzero(col)[0]
    if len(col) > 0:
        y_min = np.min(col)
        y_max = np.max(col)
        rnge.append(int((float(y_max) - y_min)/h*100.))
        slice[y_min:y_max, x] = 1

slice_to_view = (slice * 255).astype(np.uint8)

entropy_mask = (slice_ * 255).astype(np.uint8)
entropy_mask =  Image.fromarray(entropy_mask, mode='L')
entropy_mask.save(str(save_path + "EntropyMask.tiff"))

mask = (original_mask * 255 / 3).astype(np.uint8)
mask =  Image.fromarray(mask, mode='L')
mask.save(str(save_path + "FluidMask.tiff"))

slice_to_view = Image.fromarray(slice_to_view, mode='L')
slice_to_view.save(str(save_path + "CombinedMasks.tiff"))

original_mask = np.array(original_mask, dtype=np.float32)
original_mask[original_mask==0] = np.nan

entropy_mask = np.array(slice_, dtype=np.float32)
entropy_mask[entropy_mask==0] = np.nan


plt.figure()
plt.imshow(original_slice, cmap=plt.cm.gray)
plt.imshow(original_mask, alpha=0.7, cmap=plt.cm.jet)
plt.imshow(entropy_mask, alpha=0.7, cmap=plt.cm.viridis)
plt.show()