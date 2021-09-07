import os
import time
import argparse

import cv2

def loadVideo(folder):
    frames = []
    frame_names = []
    frame_list = sorted(os.listdir(folder), key=lambda name: int(name.split("_")[-1].split(".")[0]))
    for frame_name in frame_list:
        frame = os.path.join(folder, frame_name)
        frame_names.append(frame_name)
        frames.append(cv2.imread(frame))

    return frames, frame_names

def writeVideo(folder, show=False, save=""):
    # :params save => output folder
    folder = os.path.abspath(folder)
    if not os.path.exists(folder):
        print(folder, " is not existed")
        raise FileNotFoundError
    video_name = folder.split("/")[-1]
    frames, frame_names = loadVideo(folder)
    video = zip(frames, frame_names)
    sample = frames[0]
    height, width, _ = sample.shape
    if save:
        if not os.path.exists(save):
            os.makedirs(save)
        save_name = os.path.join(save, video_name+".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_name, fourcc, 25, (width, height), True)

    for frame, frame_name in video:
        if save:
            writer.write(frame)
        cv2.putText(frame, frame_name, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 2, cv2.LINE_AA)
        if show:
            cv2.imshow(video_name, frame)
        time.sleep(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/home/butlely/PycharmProjects/AAR_Net/test/cat-walkrun-080940"
    save_path = "/home/butlely/PycharmProjects/AAR_Net/output/tester"
    writeVideo(folder=video_path, show=False, save=save_path)