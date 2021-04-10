from dyslexia import io, preprocessing
import cv2
import numpy as np


def dilate(arr: np.ndarray, ite: int = 1) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(arr, kernel, iterations=ite)
    return dilation


def find_text_images(img):
    contours, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = "../../data/images/Sample_0.jpeg"
    image_orig = io.load_image(image_path, crop_border=False)

    print(image_orig.shape)

    image_gray = preprocessing.binarize(image_orig)

    dilated = dilate(image_gray, ite=6)

    plt.figure(figsize=(12, 12))
    plt.imshow(image_gray, interpolation='nearest')
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(dilated, interpolation='nearest')
    plt.show()


