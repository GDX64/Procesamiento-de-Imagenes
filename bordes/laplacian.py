import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

threshold=18;

im=cv2.imread("../lena.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", im)
cv2.waitKey()

im=np.array(im, dtype='float')
bordes_matrix=[[0,1,0],[1, -4, 1],[0,1,0]]
bordes_matrix=np.array(bordes_matrix, dtype='float')

bordes=convolve(im, bordes_matrix)
cv2.imshow("bordes", np.abs(bordes))
cv2.waitKey()

bordes_mask=np.abs(bordes);
bordes_mask[bordes_mask>threshold]=255;
bordes_mask[bordes_mask<=threshold]=0;

cv2.imshow("bordes_mask", bordes_mask)
cv2.waitKey()



hist, bin_edges = np.histogram(im, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

plt.figure()
plt.plot(bin_centers, hist)

hist, bin_edges = np.histogram(bordes, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

plt.figure()
plt.plot(bin_centers, hist)

plt.show()
