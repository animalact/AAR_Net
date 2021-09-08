import os
import time
import argparse

import cv2

def writeImages(video):
    save_path = video.split(".")[0]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise FileExistsError
    cap = cv2.VideoCapture(video)
    success, image = cap.read()
    frame_num = 0
    while success:
        ret, frame = cap.read()
        if frame is None:
            break
        cv2.imwrite(os.path.join(save_path, "frame_%d.jpg" % frame_num), frame)

        frame_num += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/butlely/PycharmProjects/AAR_Net/test/cat-walkrun-080940"
    video = "/home/butlely/PycharmProjects/AAR_Net/output/tester/mmpose_cat-walkrun-080940.mp4"
    save_path = "/home/butlely/PycharmProjects/AAR_Net/output/tester"
    writeImages(video=video)
    print("Done")