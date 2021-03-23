import cv2
import numpy as np

PYRAMIDS_DEPTH = 6

FOREGROUND = 'car.png'
BACKGROUND = 'road.jpg'

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]


def make_mask(input_image):
    copy_image = np.copy(input_image)
    for i in range(len(copy_image)):
        for j in range(len(copy_image[i])):
            copy_image[i][j] = WHITE if copy_image[i][j].any() else BLACK
    return copy_image


if __name__ == '__main__':
    A = cv2.imread(FOREGROUND)
    B = cv2.imread(BACKGROUND)
    m = make_mask(A)

    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(PYRAMIDS_DEPTH):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA = [gpA[PYRAMIDS_DEPTH - 1]]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[PYRAMIDS_DEPTH - 1]]
    gpMr = [gpM[PYRAMIDS_DEPTH - 1]]
    for i in range(PYRAMIDS_DEPTH - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        temp = cv2.pyrUp(gpA[i])
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, PYRAMIDS_DEPTH):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    cv2.imshow('Foreground', A)
    cv2.imshow('Background', B)
    cv2.imshow('Mask', m)
    cv2.imshow('Laplacian_pyramid_blending', ls_)

    cv2.waitKey(0)
