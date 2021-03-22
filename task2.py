import cv2
import numpy as np


IMAGE = 'mountains.jpeg'
# IMAGE = 'new_york.jpg'
# IMAGE = 'ocean.jpg'
# IMAGE = 'pink.jpg'
# IMAGE = 'girl_with_hat.png'

DELTA_A = 15
DELTA_B = 15
DELTA_L = 35


def saturation(channel, delta):
    new_channel = []
    for row in channel:
        new_row = []
        for pixel in row:
            new_pixel = 128
            if pixel > 128:
                new_pixel = cv2.add(np.uint8([pixel]), np.uint8([delta]))[0][0]
            if pixel < 128:
                new_pixel = cv2.subtract(np.uint8([pixel]), np.uint8([delta]))[0][0]
            new_row.append(new_pixel)
        new_row = np.array(new_row, dtype=np.uint8)
        new_channel.append(new_row)
    return np.array(new_channel, dtype=np.uint8)


def lightness(channel, delta):
    new_channel = []
    for row in channel:
        new_row = []
        for pixel in row:
            new_pixel = cv2.add(np.uint8([pixel]), np.uint8([delta]))[0][0]
            new_row.append(new_pixel)
        new_row = np.array(new_row, dtype=np.uint8)
        new_channel.append(new_row)
    return np.array(new_channel, dtype=np.uint8)


if __name__ == '__main__':
    img = cv2.imread(IMAGE, cv2.IMREAD_COLOR)
    cv2.imshow('img', img)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)

    new_l = lightness(l, DELTA_L)
    new_a = saturation(a, DELTA_A)
    new_b = saturation(b, DELTA_B)

    img_lab_sat = cv2.merge([l, new_a, new_b])
    img_sat = cv2.cvtColor(img_lab_sat, cv2.COLOR_LAB2BGR)
    cv2.imshow('img_saturation', img_sat)

    img_lab_lt = cv2.merge([new_l, a, b])
    img_lt = cv2.cvtColor(img_lab_lt, cv2.COLOR_LAB2BGR)
    cv2.imshow('img_lightness', img_lt)

    img_lab_both = cv2.merge([new_l, new_a, new_b])
    img_both = cv2.cvtColor(img_lab_both, cv2.COLOR_LAB2BGR)
    cv2.imshow('img_both', img_both)

    cv2.waitKey(0)
