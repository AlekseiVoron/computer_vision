import cv2


if __name__ == '__main__':
    def loading_displaying_saving():
        img = cv2.imread('girl.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imshow('girl', img)
        cv2.waitKey(0)
        cv2.imwrite('graygirl.jpg', img)

