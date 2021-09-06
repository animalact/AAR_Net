import h5py
import cv2
import numpy as np

def loadh5(h5file, vidname):
    with h5py.File(h5file) as f:
        data = f[vidname]['data'][:]
    return data

def visualize(data):
    data = data[..., np.newaxis]
    print(data.shape)
    cv2.imshow("wa", data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run():
    h5file = "/home/butlely/PycharmProjects/AAR_Net/data/yolact_h5/source_1.h5"
    vid = "20201123_dog-walkrun-003339.mp4"
    data = loadh5(h5file, vid)
    visualize(data[50])

run()