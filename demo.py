import cv2
import time

from third_party.yolact_model.butler_test import setYolact, runYolact


PREV = time.time()

def checktime():
    global PREV
    print(time.time()-PREV, "s / iter")
    PREV = time.time()

def run(video, onnx):
    # load yolo settings
    sess, sess_info = setYolact(onnx)

    # load video frame
    if video.isdigit():
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video)

    while True:
        ret, frame = cap.read()
        checktime()
        if frame is None:
            print("frame is None")
            break

        # my code
        img = frame.copy()
        runYolact(img, sess=sess, sess_info=sess_info)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    video = "/home/butlely/PycharmProjects/mmlab/mmpose/works_dirs/002.mp4"
    onnx = "/home/butlely/PycharmProjects/yolo/yolact2onnx/yolact_model/yolact.onnx"

    run(video, onnx)