import os
import json
import csv

import numpy as np

ACTS = {
    '꼬리를 흔든다': 7,
    '팔을 뻗어 휘적거림': 11,
    '옆으로 누워 있음': 10,
    '앞발을 뻗어 휘적거리는 동작': 11,
    '옆으로 눕는 동작': 10,
    '꼬리를 흔드는 동작': 7,
    '앞발로 꾹꾹 누름': 8,
    '발을 숨기고 웅크리고 앉음': 2,
    '납작 엎드림': 1,
    '허리를 아치로 세움': 9,
    '배를 보임': 6,
    '좌우로 뒹굴음': 4,
    '허리를 아치로 세우는 동작': 9,
    '앞발로 꾹꾹 누르는 동작': 8,
    '납작 엎드리는 동작': 1,
    '머리를 들이댐': 5,
    '머리를 들이대는 동작': 5,
    '배를 보이는 동작': 6,
    '좌우로 뒹구는 동작': 4,
    '발을 숨기고 웅크리고 앉는 동작': 2,
    '배를 보여주는 동작': 6,
    '그루밍함': 3,
    '그루밍하는 동작': 3,
    '걷거나 뛰는 동작': 12,
    '걷거나 뜀 ': 12,
    '걷거나 달리는 동작': 12
}

ACTS = {'걷거나 뜀': 13,
        '걷거나 뜀 ': 13,
        '걷거나 뛰는 동작': 13,
        '걷거나 달리는 동작': 13,
        '앞발 하나를 들어 올림': 3,
        '꼬리가 아래로 향함': 10,
        '마운팅': 9,
        '앞발 하나를 들어 올리는 동작': 3,
        '마운팅하는 동작': 9,
        '꼬리를 아래로 내리는 동작': 10,
        '엎드리기(몸체를 낮게 유지)': 5,
        '몸을 턴다': 4,
        '엎드리는 동작(몸체를 낮게 유지)': 5,
        '몸을 터는 동작': 4,
        '엎드리는 동작': 5,
        '앉기': 11,
        '빙글빙글 돈다': 8,
        '빙글빙들 돈다': 8,
        '앉는 동작': 11,
        '빙글빙글 도는 동작': 8,
        '두 앞발을 들어 올림': 2,
        '꼬리를 위로 올리고 흔듦': 7,
        '꼬리를 흔듦': 7,
        '두 앞발을 들어 올리는 동작': 2,
        '꼬리를 위로 올리고 흔드는 동작': 7,
        '배와 목을 보여주며 누움': 6,
        '머리를 앞으로 들이댐': 1,
        '몸을 긁는 동작': 12,
        '머리를 앞으로 들이미는 동작': 1,
        '배와 목을 보여주며 눕는 동작': 6,
        '몸을 긁음': 12,
        '머리를 들이대는 동작': 1}


def readAIhubJson(aihub_json):
    filename = aihub_json.split(".json")[0].split("/")[-1]
    label_num = aihub_json.split(".json")[0]
    with open(aihub_json, "r") as f:
        data = json.load(f)

    # 대분류
    meta = data['metadata']  # 중요 정보 포함
    anno = data['annotations']  # 프레임 한개가 하나의 딕셔너리를 갖는 리스트 반환 [ {}, {}, ...]

    # 소분류
    filename = filename
    action = meta['action']  # label 될 예정
    heigth = meta['height']
    width = meta['width']
    species = meta['species']
    breed = meta['animal']['breed']
    emotion = meta['inspect']['emotion']
    keypoints = getKeypoints(anno)

    metadata = {
            "filename": filename,
            "action": action,
            "action_id": ACTS[action],
            "height": heigth,
            "width": width,
            "species": species,
            "breed": breed,
            "emotion": emotion,
            "keypoints": keypoints,
        }

    return metadata

def getKeypoints( annotations):
    keypoints = []
    for annotation in annotations:
        keypoints.append(annotation['keypoints'])
    return keypoints

def toNumpy(keypoints):
    keypoint_in_video = []
    for keypoint in keypoints:
        keypoint_in_frame = []
        for i in range(1, 16):
            coord = keypoint[str(i)]
            if coord is None:
                keypoint_in_frame.extend([0, 0])
            else:
                keypoint_in_frame.append(coord['x'])
                keypoint_in_frame.append(coord['y'])
        keypoint_in_video.append(keypoint_in_frame)
    vid_np = np.array(keypoint_in_video, dtype='int16')
    return vid_np

def transformNpy(arr, frame_thr=-1):
    if frame_thr == -1:
        return arr

    assert frame_thr > 0

    frame_count = arr.shape[0]
    if frame_count <= frame_thr+1:
        return None
    start = frame_count//2-frame_thr//2
    end =  frame_count//2+frame_thr//2
    return arr[start:end]

def createCsv(csvfile, fieldnames):
    with open(csvfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def saveCsv(csvfile, metadata):
    fieldnames = list(metadata.keys())
    if not os.path.exists(csvfile):
        print("new csv file created")
        createCsv(csvfile, fieldnames)
    with open(csvfile, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(metadata)

def saveNpy(npyfile, arr):
    np.save(npyfile, arr)

def saveTxt(txtfile, dict):
    with open(txtfile, "w") as f:
        f.writelines(dict)

def main(label_id):
    # INPUT
    label_num = f"label_{label_id}"
    folder = f"/home/butlely/Desktop/Dataset/aihub/label/{label_num}"
    frame_thr = -1

    # OUTPUT
    savefolder = "/home/butlely/PycharmProjects/utils/AAR_NET/etc/points_npy_allframe/"
    label_folder = os.path.join(savefolder, label_num)
    if not os.path.exists(label_folder):
        os.mkdir(label_folder)

    csvfile = os.path.join(savefolder, f"{label_num}.csv")
    txtfile = os.path.join(savefolder, f"{label_num}.txt")

    if os.path.exists(csvfile):
        print("file already exists")
        return

    # for check details
    cur_id = 0
    arr_err = []
    check_actions = {}
    dic = {}
    jsonlist = sorted(list(os.listdir(folder)))
    json_count = len(jsonlist)
    for jsonfile in jsonlist:
        npyname = f'{jsonfile.split(".json")[0]}.npy'
        npyfile = os.path.join(savefolder, label_num, npyname)
        metadata = readAIhubJson(os.path.join(folder, jsonfile))

        arr = toNumpy(metadata.pop('keypoints'))
        arr = transformNpy(arr, frame_thr)
        if arr is None:
            arr_err.append(jsonfile)
            continue
        metadata['label_num'] = label_id
        metadata['frames'] = arr.shape[0]
        # saveNpy(npyfile, arr)
        saveCsv(csvfile, metadata)

        # this is just for checking actions
        action = metadata['action']
        count = check_actions.get(action, 0)
        count += 1
        check_actions[action] = count

        # this is just for check current vid_num
        cur_id += 1
        if cur_id % 10 == 0:
            print("", end="\r")
            print(f" {label_num} : {cur_id} / {json_count} is created", end="")
    print()
    print(check_actions)
    saveTxt(txtfile, str(check_actions))

for i in range(1,7):
    main(i)









