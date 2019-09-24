import cv2
import numpy as np
import matplotlib.pyplot as plt

im_grey=cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
im=cv2.imread("lena.png")
cv2.imshow("image", im)
cv2.imshow("image builtin grey", im_grey)
cv2.waitKey()

#%% Manual grey

def manual_grey(im):
    alpha=0.2126; beta=0.7152; gamma=0.0722
    transform=np.array((alpha,beta,gamma))
    im[1,2]*transform.transpose()
    im_grey=np.dot(im, transform.transpose())
    im_grey=np.array(im_grey, dtype="uint8")

    return im_grey

def manual_red(im):
    transform=np.array((1,0,0))
    im[1,2]*transform.transpose()
    im_trans=np.dot(im, transform.transpose())
    im_trans=np.array(im_trans, dtype="uint8")

    return im_trans

def manual_blue(im):
    transform=np.array((0,1,0))
    im[1,2]*transform.transpose()
    im_trans=np.dot(im, transform.transpose())
    im_trans=np.array(im_trans, dtype="uint8")

    return im_trans

def manual_green(im):
    transform=np.array((0,0,1))
    im[1,2]*transform.transpose()
    im_trans=np.dot(im, transform.transpose())
    im_trans=np.array(im_trans, dtype="uint8")

    return im_trans

my_im_grey = manual_grey(im)
my_im_grey
cv2.imshow("image_grey", my_im_grey)
cv2.waitKey()

#%% Lets put some noise
noise=np.random.binomial(1,0.5,my_im_grey.shape)*20-10
n_gim=np.array(my_im_grey+noise, dtype="uint8")

cv2.imshow("image_binomial_noise", n_gim)
cv2.waitKey()

#Just a constant
noise=8
n_gim=np.array(my_im_grey+noise, dtype="uint8")

cv2.imshow("image_binomial_noise", n_gim)
cv2.waitKey()

#Gausian error
noise=np.random.randn(my_im_grey.shape[0],my_im_grey.shape[1])*5
n_gim=np.array(my_im_grey+noise, dtype="uint8")

cv2.imshow("image_binomial_noise", n_gim)
cv2.waitKey()
