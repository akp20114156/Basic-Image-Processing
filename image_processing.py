from skimage import io, feature
from skimage.color import rgb2gray
import numpy as np
from skimage.util import random_noise
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter
from skimage.segmentation import slic
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

#1. Determine the size of the avengers imdb.jpg image. Produce a grayscale and a blackand-white representation of it.

# read image and show size
image = io.imread(fname="../data/image_data/avengers_imdb.jpg")
grayscale_image = rgb2gray(image)
print('Image size',image.size)

threshold =128
maxval =255
grayscale = (grayscale_image*256).astype('uint8')
im_bool = (grayscale > threshold)

im_binary = (grayscale > threshold)*maxval
binary_image = np.array(im_binary,dtype=np.uint8)

# display image
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(grayscale_image,cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
ax[2].imshow(binary_image)
ax[2].set_title("Black and White")

fig.tight_layout()
plt.show()

# 2. 
image = io.imread(fname="../data/image_data/bush_house_wikipedia.jpg")
noise_gs_img = random_noise(image,mode='gaussian',var=0.1)
filter_gs_img =gaussian(noise_gs_img, sigma=1, multichannel='False')
filter_uniform_img= uniform_filter(noise_gs_img,size=(9,9,1))
#display image
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(noise_gs_img)
ax[1].set_title("Gaussian noise")
ax[2].imshow(filter_gs_img)
ax[2].set_title("Gaussian mask")
ax[3].imshow(filter_uniform_img)
ax[3].set_title("Uniform mask")


fig.tight_layout()
plt.show()

#3.
image = io.imread(fname="../data/image_data/forestry_commission_gov_uk.jpg")
grayscale_image = color.rgb2gray(image)
segments = slic(image, n_segments=5, compactness=10, sigma=1,multichannel=True)
seg_image = color.label2rgb(segments, image, kind='overlay')

# display image
fig, axes = plt.subplots(1, 3, figsize=(15, 15))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Original")
ax[1].imshow(segments)
ax[1].set_title("k means clustering")
ax[2].imshow(seg_image)
ax[2].set_title("Segmented image")
fig.tight_layout()
plt.show()

#4.
image = io.imread(fname="../data/image_data/rolland_garros_tv5monde.jpg")
grayscale_image = color.rgb2gray(image)

edges1 = feature.canny(grayscale_image)
edges2 = feature.canny(grayscale_image, sigma=2)

# Classic straight-line Hough transform
h, theta, d = hough_line(grayscale_image)

# Line finding using the Probabilistic Hough Transform
lines = probabilistic_hough_line(edges1, threshold=10, line_length=60,line_gap=5)

# display image 

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Original")

ax[1].imshow(edges1,cmap=plt.cm.gray)
ax[1].set_title("Canny edges")

ax[2].imshow(grayscale_image, cmap=plt.cm.gray)
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines - Hough transform')

ax[3].imshow(edges1 * 0)
for line in lines:
    p0, p1 = line
    ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[3].set_xlim((0, image.shape[1]))
ax[3].set_ylim((image.shape[0], 0))
ax[3].set_title('Detected lines-Probabilistic Hough')

fig.tight_layout()
plt.show()