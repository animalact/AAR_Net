import cv2

def putCircle(frame, coords):
    thr = 0.5
    for coord in coords:
        x = int(coord[0])
        y = int(coord[1])
        t = coord[2]
        if t > thr and x > 0 and y > 0:
            frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=-1)
    return frame