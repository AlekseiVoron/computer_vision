import cv2
import numpy as np


IMAGE = 'mountains.jpeg'
# IMAGE = 'new_york.jpg'
# IMAGE = 'ocean.jpg'
# IMAGE = 'pink.jpg'
# IMAGE = 'girl_with_hat.png'

ALPHA = 0.5
BETA = 1.0 - ALPHA
IMG1 = '800600_1.jpg'
IMG2 = '800600_2.jpg'


if __name__ == '__main__':
    img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(IMAGE)

    equ = cv2.equalizeHist(img)
    equ_res = np.hstack((img, equ))
    cv2.imshow('equ', equ_res)

    clahe = cv2.createCLAHE(4, (8, 8)).apply(img)
    clahe_res = np.hstack((img, clahe))
    cv2.imshow('clahe', clahe_res)

    gauss = cv2.GaussianBlur(img_color, (5, 5), 0)
    gauss_res = np.hstack((img_color, gauss))
    cv2.imshow('gauss', gauss_res)

    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    sobel_res = np.hstack((img, sobel))
    cv2.imshow('sobel', sobel_res)

    laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=3))
    laplacian_res = np.hstack((img, laplacian))
    cv2.imshow('laplacian', laplacian_res)

    src1 = cv2.imread(IMG1)
    src2 = cv2.imread(IMG2)
    alpha = 0.5
    beta = (1.0 - alpha)
    alpha_blending = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    cv2.imshow('alpha_blending', alpha_blending)

    cv2.waitKey(0)
