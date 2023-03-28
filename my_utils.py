import cv2

IMAGE_SIZE = (320, 320)
MARGIN_BOTTOM = 100


def image_process(img):
    h, w, _ = img.shape
    x_start = int(w / 2 - IMAGE_SIZE[0] / 2)
    y_start = int(h / 2 - IMAGE_SIZE[1] / 2 - MARGIN_BOTTOM)
    x_end = int(w / 2 + IMAGE_SIZE[0] / 2)
    y_end = int(h / 2 + IMAGE_SIZE[1] / 2 - MARGIN_BOTTOM)
    img = img[y_start:y_end, x_start:x_end, :]
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img
