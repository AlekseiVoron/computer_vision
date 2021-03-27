import cv2
import numpy as np

FOREGROUND = 'car.png'
BACKGROUND = 'road.jpg'

WHITE = [1, 1, 1]
BLACK = [0, 0, 0]

NUM_LEVELS = 6


def gaussian_pyramid(img):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(NUM_LEVELS):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.uint8(lower))
    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        beta_mask = (1 - mask)
        ls = cv2.add(cv2.multiply(lb, mask), cv2.multiply(la, beta_mask))
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


def make_mask(input_image):
    m = np.zeros(np.shape(input_image), dtype='uint8')
    for i in range(len(input_image)):
        for j in range(len(input_image[i])):
            m[i][j] = WHITE if input_image[i][j].any() else BLACK
    return m


if __name__ == '__main__':
    img1 = cv2.imread(BACKGROUND)
    img2 = cv2.imread(FOREGROUND)
    mask = make_mask(img2)

    gaussian_pyr_1 = gaussian_pyramid(img1)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    gaussian_pyr_2 = gaussian_pyramid(img2)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    mask_pyr_final = gaussian_pyramid(mask)
    mask_pyr_final.reverse()

    add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    final = reconstruct(add_laplace)

    cv2.imshow('Final', final[NUM_LEVELS])
    cv2.waitKey(0)
