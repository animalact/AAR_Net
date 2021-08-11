import os
import json
import numpy as np
import matplotlib.pyplot as plt

def getJsonData(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def getVideoData(data, video_name):
    """
    data format -> { "data": [ { "frame_index": int,
                                 "skeleton": [{
                                                "pose": [x1, y1, x2, y2, ... ],
                                                "score": [1,1,1,...]
                                             }]
                                },
                                { "frame_index": int,
                                 "skeleton": [{
                                                "pose": [x1, y1, x2, y2, ... ],
                                                "score": [1,1,1,...]
                                             }]
                                }, ...
                            ]
                     "label": str               # action
                     "label_index":  int        # unique data index (not sure)
                     }
    """

    video = data.get(video_name, None)
    if video is None:
        raise FileNotFoundError

    # first category
    vid_data = video['data']                # data contains frame_idx, skeleton(pose, score)
    vid_label = video['label']              # action
    vid_label_index = video['label_index']  # unique index

    # init numpy array
    pose_np = []

    # data category
    for frame in vid_data:
        frame_idx = frame['frame_index']
        skeleton = frame['skeleton']
        pose = skeleton[0]['pose']
        score = skeleton[0]['score']
        pose_np.append(_getPose(pose))

    pose_np = np.array(pose_np)
    print(pose_np.shape)
    return pose_np, vid_label

def _getPose(pose):
    pose_reshape = []
    for i, p in enumerate(pose):
        if i % 2 == 1:
            pos = [pose[i-1], pose[i]]
            pose_reshape.append(pos)
    a = np.array(pose_reshape)
    return pose_reshape

if __name__ == "__main__":
    data = getJsonData("/home/butlely/PycharmProjects/utils/AAR_NET/etc/result/label_1.json")
    getVideoData(data, "20201117_dog-walkrun-002803.mp4")