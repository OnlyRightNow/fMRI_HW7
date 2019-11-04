""" Voxel correlation exercise
"""
#: compatibility with Python 2
from __future__ import print_function, division

#: import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
import nibabel as nib

# - import events2neural from stimuli module
from stimuli import events2neural

# - Load the ds114_sub009_t2r1.nii image
file = nib.load('ds114_sub009_t2r1.nii')
# - Get the number of volumes in ds114_sub009_t2r1.nii
n_vols = file.shape[-1]
print('number of volumes in file: ', n_vols)
#: TR (time between scans)
TR = 2.5

# - Call the events2neural function to generate the on-off values for
# - each volume.  Plot these values.
on_off = events2neural('ds114_sub009_t2r1_cond.txt', 2.5, n_vols) # work also for other values than 2.5, like 0.25
plt.plot(on_off)
plt.show()
# - Drop the first 4 volumes, and the first 4 on-off values.
image_data = file.get_data()
image_data = image_data[..., 4:]
on_off = on_off[4:]
# - Make a brain-volume-size array of 0 to hold the correlations
zeros = np.zeros(image_data.shape[:-1])
# - Loop over all voxel indices.
# - Extract the voxel time courses at each voxel.
# - Get correlation value for voxel time course with on-off vector.
# - Fill value in the correlations array.
corr = zeros
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        for k in range(image_data.shape[2]):
            corr[i, j, k] = np.corrcoef(on_off, image_data[i, j, k])[1, 0]
# - Plot the middle slice of the third axis from the correlations array
plt.imshow(corr[:, :, 15])
plt.show()
