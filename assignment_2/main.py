import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def padding(image, border_width):
   reflect = cv2.copyMakeBorder(image,border_width, border_width, border_width, border_width,cv2.BORDER_REFLECT)
   plt.plot(1), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
   plt.show()

def crop(image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imshow("Cropped", cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow("resized", resized_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def copy(image, emptyPictureArray):
    emptyPictureArray[:] = image[:]

    cv2.imshow("source image", image)
    cv2.imshow("Copied Image", emptyPictureArray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hsv(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", hsvImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()





def smoothing(image):
    dst = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)

    cv2.imshow("Gaussian Smoothing", np.hstack((image, dst)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hue_shifted(image, emptyPictureArray, hue):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    shift = int(round(hue / 2.0))
    h = ((h.astype(np.int16) + shift) % 180).astype(np.uint8)

    hsv_shifted = cv2.merge([h, s, v])
    hue_image = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

    emptyPictureArray[:] = hue_image

    cv2.imshow("Correct Hue Image :)", hue_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotation(image, rotation_angle):
    print(rotation_angle)
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        exit(1)

    cv2.imshow("bruh", rotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image = cv2.imread('lena-1.png')
    border_width = 100
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Current working directory:", os.getcwd())

    height, width, channel = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

    #padding(img_rgb, border_width)

    #crop(image, 80, 382, 80, 382)

    #resize(image, 200, 200)

    #copy(image, emptyPictureArray)

    #grayscale(image)

    #hsv(image)

    hue_shifted(image, emptyPictureArray, 50)

    #smoothing(image)

    #rotation(image, 90)


if __name__ == "__main__":
    main()