import os
import time

import cv2

from third_party.yolact_model.yolact import setYolact, runYolact
from third_party.mmpose_model.mmpose import setMmpose, runMmpose

PREV = time.time()

def checktime(text=""):
    global PREV
    if text:
        print(text)
    print(time.time()-PREV, "s / iter")
    PREV = time.time()

def run(video, yolo_onnx, mm_onnx, show=True, save_folder=""):
    # load yolo settings
    yolo_sess, yolo_sess_info = setYolact(yolo_onnx)
    checktime("yolo set")
    mm_sess, mm_sess_info = setMmpose(mm_onnx)
    checktime("mmpose set")

    # load video frame
    if video.isdigit():
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video)


    if save_folder:
        _, sample = cap.read()
        height, width, _ = sample.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vidname = video.split("/")[-1]
        yolact_writer_name = os.path.join(save_folder, "yolact_"+vidname)
        mmpose_writer_name = os.path.join(save_folder, "mmpose_"+vidname)
        yolact_writer = cv2.VideoWriter(yolact_writer_name, fourcc, 25, (width, height), True)
        mmpose_writer = cv2.VideoWriter(mmpose_writer_name, fourcc, 25, (width, height), True)

    i = 0
    while True:
        i += 1

        ret, frame = cap.read()
        checktime()
        if frame is None:
            print("frame is None")
            break

        # my code
        img = frame.copy()
        yolact_img, bbox, mask = runYolact(img, sess=yolo_sess, sess_info=yolo_sess_info)
        mmpose_img, coords = runMmpose(img, bbox, mm_sess, mm_sess_info )

        if save_folder:
            yolact_writer.write(yolact_img)
            mmpose_writer.write(mmpose_img)
        # runTranformer
        if show:
            cv2.imshow("demo", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # if save:


    yolact_writer.release()
    mmpose_writer.release()
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    video = "/home/butlely/PycharmProjects/mmlab/mmpose/works_dirs/002.mp4"
    output_folder = "/home/butlely/PycharmProjects/AAR_Net/output/tester/"
    video = "/home/butlely/PycharmProjects/AAR_Net/output/tester/cat-walkrun-080940.mp4"
    yolo_onnx = "/home/butlely/PycharmProjects/yolo/yolact2onnx/yolact_model/yolact_model.onnx"
    mm_onnx = "/home/butlely/PycharmProjects/AAR_Net/weights/mmpose/onnx/poodle_w48.onnx"
    run(video, yolo_onnx, mm_onnx, show=False, save_folder=output_folder)
    print("Done")