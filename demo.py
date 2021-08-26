import cv2
import time

from third_party.yolact_model.yolact import setYolact, runYolact
from third_party.mmpose_model.mmpose import setMmpose, runMmpose

PREV = time.time()

def checktime(text=""):
    global PREV
    if text:
        print(text)
    print(time.time()-PREV, "s / iter")
    PREV = time.time()

def run(video, yolo_onnx, mm_onnx, show=True, save=""):
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
    # if save:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     writer = cv2.VideoWriter(save, fourcc, 25.0, (640, 480))

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
        img, bbox, mask = runYolact(img, sess=yolo_sess, sess_info=yolo_sess_info)
        img, coords = runMmpose(img, bbox, mm_sess, mm_sess_info )

        # runTranformer
        if show:
            cv2.imshow("demo", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # if save:

    cap.release()
    # writer.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    video = "/home/butlely/PycharmProjects/mmlab/mmpose/works_dirs/002.mp4"
    yolo_onnx = "/home/butlely/PycharmProjects/yolo/yolact2onnx/yolact_model/yolact_model.onnx"
    mm_onnx = "/home/butlely/PycharmProjects/AAR_Net/weights/mmpose/onnx/poodle_w48.onnx"
    run(video, yolo_onnx, mm_onnx, save="output.mp4")