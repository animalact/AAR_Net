import json
import os

import matplotlib.pyplot as plt
import cv2

class AIhubToKinetics:
    def __init__(self, keypoint_number=15):
        self.keypoint_number = keypoint_number
        self.size = None            # (width, height)

    def readAIhubJson(self, aihub_json):
        with open(aihub_json, "r") as f:
            data = json.load(f)
        
        # 대분류
        meta = data['metadata']     # 중요 정보 포함
        anno = data['annotations']  # 프레임 한개가 하나의 딕셔너리를 갖는 리스트 반환 [ {}, {}, ...]

        # 소분류
        seq = meta['seq']           # label_index 될 예정
        action = meta['action']     # label 될 예정
        heigth = meta['height']
        width = meta['width']
        keypoints = self._getKeypoints(anno)
        aihub_data = {
            "seq": seq,
            "action": action,
            "size" : (width, heigth),
            "keypoints": keypoints
        }
        return aihub_data

    def convertToKinetic(self, aihub_json_file):
        """
        ['label_index', 'label', 'data'] is essential for kinetics
        """
        aihub_data = self.readAIhubJson(aihub_json_file)
        self.size = aihub_data['size']
        label_index = aihub_data['seq']
        label = aihub_data['action']
        data = self._extractData(aihub_data['keypoints'])
        
        kinetic_format = {
            'data': data,
            'label_index': label_index,
            'label': label
        }
        
        return kinetic_format

    def iterSrcFolder(self, src_folder, output):
        data = {}
        labels = sorted(os.listdir(src_folder))
        len_labels = len(labels)
        count = 0
        for label in labels:
            count += 1
            label_path = os.path.join(src_folder, label)
            key = label.split(".json")[0]
            data[key] = self.convertToKinetic(label_path)
            print(f"{label} ({count} / {len_labels}) is created")

        self.saveToJson(data, output)

    def saveToJson(self, data, filename):
        with open(filename, "w") as f:
            json.dump(data, f)
        
    def _getKeypoints(self, annotations):
        keypoints = []
        for annotation in annotations:
            keypoints.append(annotation['keypoints'])
        return keypoints

    def _extractData(self, keypoints):
        """
        :param keypoints    : [ {1:{x:12, y:21}, 2:null, ...}, ...
        :param return       : []
        """
        data = []
        for i, keypoint in enumerate(keypoints):
            frame_index = i+1
            pose = self._getPose(keypoint)
            score = self._getScore()
            frame_info = {
                'frame_index': frame_index,
                'skeleton': [{
                    "pose": pose,
                    "score": score
                }]
            }
            data.append(frame_info)
        return data

    def _getPose(self, keypoint):
        """
        :param  : {1: {x:12, y:21}, 2:null, ...}
        """
        pose = []
        for i in range(self.keypoint_number):
            kp = keypoint[str(i+1)]
            if kp is None:
                x, y = 0, 0
            else:
                (w, h) = self.size
                x = round(kp['x']/w, 3)
                y = round(kp['y']/h, 3)
            pose.extend([x,y])
        return pose

    def _getScore(self):
        score = [1]*self.keypoint_number
        return score


if __name__ == "__main__":
    """
    ## aihub_label_path = "/Users/song-yunsang/Desktop/Business/Butler/Dataset/test/aihub/label_1/20201117_dog-walkrun-002803.mp4.json"
    ## save_path = "/Users/song-yunsang/Desktop/Business/Butler/Dataset/test/aihub/label_1/kinetics.json"
    ## 아래 두줄 주석 풀고 주소 넣어서 사용
    """
    for i in range(1,10):
        label = f"label_{i}"
        aihub_label_path = f"/home/butlely/Desktop/Dataset/aihub/{label}"
        save_path = f"/home/butlely/PycharmProjects/utils/AAR_NET/etc/result/{label}.json"
        Converter = AIhubToKinetics()
        # kinectic_form = Converter.convertToKinetic(aihub_label_path)
        # Converter.saveToJson(kinectic_form, save_path)
        # # Converter._readKineticJson("")
        Converter.iterSrcFolder(aihub_label_path, save_path)





