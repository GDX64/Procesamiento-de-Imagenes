import cv2
import numpy as np
import matplotlib.pyplot as plt

im=cv2.imread("camera.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("image", im)
cv2.waitKey()

IM=np.fft.fft2(im)
IM=np.fft.fftshift(IM)
IM_abs=np.log2(np.abs(IM))
IM_abs=np.array(IM_abs/IM_abs.max()*255, dtype="uint8")
IM_abs
cv2.imshow("Fourier", IM_abs)
cv2.waitKey()
