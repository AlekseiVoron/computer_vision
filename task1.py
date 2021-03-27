import cv2
from math import log
from os import listdir, path
import matplotlib.pyplot as plt


FOLDER_IMGS = 'task1_images'
BATCH_SIZE = 100


if __name__ == '__main__':
    imgs = []
    pixels_sums = []
    for img_file in listdir(FOLDER_IMGS):
        img = cv2.imread(path.join(FOLDER_IMGS, img_file), cv2.IMREAD_GRAYSCALE)
        pixels_sum = log(cv2.sumElems(img[:BATCH_SIZE, :BATCH_SIZE])[0])
        pixels_sums.append(pixels_sum)
    Evs = [-4.0, -3.7, -3.3, -3.0, -2.7, -2.3, -2.0, -1.7, -1.3, -1.0, -0.7, -0.3,
           0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0]
    plt.xlabel('EV')
    plt.grid()
    plt.plot(Evs, pixels_sums)
    plt.show()
