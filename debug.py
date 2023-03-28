import argparse

from my_utils import image_process
import torch
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("path", help="video path")
args = parser.parse_args()


def debug_key_point():
    path = args.path
    file_name = path.split("/")[-1].split(".")[0]

    detector = cv2.ORB_create()
    cap = cv2.VideoCapture(path)
    index = 0

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = image_process(frame)

        key_points = detector.detect(frame, None)
        # key_points, des = detector.compute(frame, key_points)
        frame = cv2.drawKeypoints(frame, key_points, None, color=(0, 255, 0), flags=0)
        cv2.imwrite(f"./debug/{file_name}-{index}.jpg", frame)
        index += 1


def debug_person():
    path = args.path
    file_name = path.split("/")[-1].split(".")[0]

    cap = cv2.VideoCapture(path)
    model = torch.hub.load("ultralytics/yolov5", "custom", "yolov5s.pt")

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break

        result = model(frame)
        result.show()
        break


if __name__ == "__main__":
    debug_person()
