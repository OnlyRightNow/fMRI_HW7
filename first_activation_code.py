""" First go at brain activation exercise
"""

#: import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
import nibabel as nib

# Display array values to 6 digits of precision
np.set_printoptions(precision=4, suppress=True)

# - Read the file into an array called "task".
# - "task" should have 3 columns (onset, duration, amplitude)
# - HINT: np.loadtxt
file = np.loadtxt('ds114_sub009_t2r1_cond.txt')
print('shape file: ', file.shape)
# - Select first two columns and divide by TR
TR = 2.5  # take same as in first exercise
columns = file[:, :2] / TR

# - Load the image and check the image shape to get the number of TRs
img_file = nib.load('ds114_sub009_t2r1.nii')
print('img_file shape: ', img_file.shape)
# - Make new zero vector
zeros_img_file = np.zeros(img_file.shape[-1])
#: try running this if you don't believe me
len(range(4, 16))

# - Fill in values of 1 for positions of on blocks in time course
oneset_duration = np.round(columns).astype(int)
time = zeros_img_file
for onset, duration in oneset_duration:
    time[onset: onset + duration] = 1

# - Plot the time course
plt.plot(time)
plt.show()
# - Make two boolean arrays encoding task, rest
task = (time == 1)
rest = (time == 0)

# - Create a new 4D array only containing the task volumes
img_data = img_file.get_data()
task_vol = img_data[..., task]
# - Create a new 4D array only containing the rest volumes
rest_vol = img_data[..., rest]
# - Create the mean volume across all the task volumes.
# - Then create the mean volume across all the rest volumes
task_mean = task_vol.mean(axis=-1)
rest_mean = rest_vol.mean(axis=-1)
# - Create a difference volume
difference_vol = task_mean - rest_mean
# - Show a slice over the third dimension
plt.imshow(difference_vol[:, :, 22])
plt.show()
# - Calculate the SD across voxels for each volume
# - Identify the outlier volume
voxels = img_data.reshape((-1, img_data.shape[-1]))
sd = np.std(voxels, axis=0)
print('outlier volume: ', np.argmax(sd))
# - Use slicing to remove outlier volume from rest volumes
rest_vol_sliced = rest_vol[..., 1:]
# - Make new mean for rest volumes, subtract from task mean
rest_sliced_mean = rest_vol_sliced.mean(axis=-1)
difference_vol_sliced = task_mean - rest_sliced_mean
# - show same slice from old and new difference volume
plt.imshow(difference_vol_sliced[:, :, 22])
plt.show()
