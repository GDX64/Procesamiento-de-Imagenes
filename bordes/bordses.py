import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

threshold=20;

im=cv2.imread("../lena.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", im)
cv2.waitKey()

im=np.array(im, dtype='float')
bordes_matrix_x=[[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1]]
#bordes_matrix_x=[[-1,1],[-1,1]]
bordes_matrix_x=np.array(bordes_matrix_x, dtype='float')/4
bordes_matrix_y=[[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1],[1,1,1,1]]
#bordes_matrix_y=[[1,1],[-1,-1]]
bordes_matrix_y=np.array(bordes_matrix_y, dtype='float')/4

bordes_x=convolve(im, bordes_matrix_x)
bordes_y=convolve(im, bordes_matrix_y)
bordes=(bordes_x+bordes_y)/2;
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
