import glob
import numpy as np
import cv2
from tqdm import tqdm
from my_utils import image_process

SCALE = 1


def main():
    video_files = glob.glob("./apex_clips/*.mp4")

    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for video_file in tqdm(video_files):
        track_list = process_video(orb, matcher, video_file)
        bullet_img = draw_bullet_track(track_list)
        filename = video_file.split("/")[-1].split(".")[0]
        cv2.imwrite(f"./bullet_img/{filename}.jpg", bullet_img)


def draw_bullet_track(track_list):
    bullet_canvas = np.zeros((800, 800, 3))
    start_point = np.array((bullet_canvas.shape[0] / 2, bullet_canvas.shape[1] / 2))
    end_point = None

    cv2.circle(bullet_canvas, start_point.astype(int), 3, (0, 0, 255), -1)  # Start

    for movment in track_list:
        end_point = np.add(start_point, movment * SCALE)
        cv2.line(
            bullet_canvas,
            start_point.astype(int),
            end_point.astype(int),
            (0, 255, 0),
            1,
        )
        start_point = end_point

    if end_point is not None:
        cv2.circle(bullet_canvas, end_point.astype(int), 3, (255, 0, 0), -1)  # End

    return bullet_canvas


def process_video(detector, matcher, path):

    last_key_point = None
    last_dest = None
    track_list = []
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = image_process(frame)

        if last_key_point is None:
            last_key_point, last_dest = detector.detectAndCompute(frame, None)
            # cv2.imwrite("first.jpg", frame)
            continue

        # cv2.imwrite("second.jpg", frame)
        current_key_point, current_dest = detector.detectAndCompute(frame, None)

        matches = matcher.match(last_dest, current_dest)
        matches = sorted(matches, key=lambda x: x.distance)

        movment = 0
        for match in matches[:10]:
            point_r = np.array(current_key_point[match.trainIdx].pt)
            point_l = np.array(last_key_point[match.queryIdx].pt)
            movment += np.subtract(point_l, point_r)
        movment = movment / 10

        track_list.append(movment)

        last_key_point = current_key_point
        last_dest = current_dest

    cap.release()

    return track_list


if __name__ == "__main__":
    main()
