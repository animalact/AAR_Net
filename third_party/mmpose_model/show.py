import cv2
import matplotlib.pyplot as plt

def putCircle(frame, coords):
    thr = 0.1
    for coord in coords:
        x = int(coord[0])
        y = int(coord[1])
        t = coord[2]
        if t > thr and x > 0 and y > 0:
            frame = cv2.circle(frame, (x, y), radius=6, color=(0, 255, 0), thickness=10)
    return frame

def show(img, keypoints):
    plt.imshow(img)
    xs, ys = alignKeypoint(keypoints)
    plt.scatter(xs, ys)
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()


def alignKeypoint(keypoints):
    xs = []
    ys = []
    for keypoint in keypoints:
        xs.append(keypoint[0])
        ys.append(keypoint[1])
    return xs, ys