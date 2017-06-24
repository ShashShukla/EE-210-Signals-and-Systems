"""
Image processing via convolution with 2D kernels

Author: Shashwat Shukla
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sem_ic.jpg', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Normalised averaging filter
mean = (1. / 9) * np.ones((3, 3))

# Guassian filter
x = cv2.getGaussianKernel(10, 10)
gaussian = x * x.T

# Sharpen
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

# Edge detecting filters

# Laplacian_1
laplacian_1 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

# Laplacian_2
laplacian_2 = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

# Laplacian of Gaussian
lapguass = np.array([[0, 0, -1, 0, 0],
                     [0, -1, -2, -1, 0],
                     [-1, -2, 16, -2, -1],
                     [0, -1, -2, -1, 0],
                     [0, 0, -1, 0, 0]])

# Scharr along x axis
scharr = np.array([[-3, 0, 3],
                   [-10, 0, 10],
                   [-3, 0, 3]])
# Sobel along x axis
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
# Sobel along y axis
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])


filters = [mean, gaussian, sharpen, laplacian_1, laplacian_2, lapguass, scharr,
           sobel_x, sobel_y]
filter_name = ['Normalised mean', 'Gaussian', 'Sharpen', 'Laplacian_1',
               'Laplacian_2', 'Laplacian of Gaussian',
               'Scharr_x', 'Sobel_x', 'Sobel_y']

out = [cv2.filter2D(img, -1, x) for x in filters]

outfft = [np.fft.fft2(x) for x in out]
outfftshift = [np.fft.fftshift(x) for x in outfft]
out_mag = [20 * np.log(np.abs(x)) for x in outfftshift]

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()

for i in xrange(9):
    plt.subplot(3, 3, i + 1), plt.imshow(filters[i], cmap='gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

for i in xrange(9):
    plt.subplot(3, 3, i + 1), plt.imshow(out[i], cmap='gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

for i in xrange(9):
    plt.subplot(3, 3, i + 1), plt.imshow(out_mag[i], cmap='gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

plt.imshow(img, cmap='gray')
plt.show()

for i in xrange(9):
    plt.imshow(out[i], cmap='gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()