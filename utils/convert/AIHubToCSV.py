# -*- coding:utf-8 -*-
"""
사용방법
demo
converter = AIHubToCSV()
converter.getDataFromJson(jsonpath) -> json 1개의 값 반환
converter.getDataList(label_folder) -> 폴더 안에 있는 json값 리스트 반환 (ex. ./라벨데이터_1 ) (6000개)
converter.iterLabelFolder(root_folder)  -> 루트 폴더 안에 있는 모든 라벨 폴더의 json값 합쳐서 1개 리스트 반환 (6만개)

주의사항
"라벨데이터_x" 은 인코딩 문제로 인하여 "label_x" 로 수정 후 진행
"원천데이터_x" 도 마찬가지로 "source_x"로 수정 후 진행

"""
import os
import csv
import json
import collections

class AIHubToCSV:
    def __init__(self):
        pass

    def iterLabelFolder(self, aihub_folder):
        """
        라벨데이터를 모두 순환하는 구조
        """
        data_list = []
        txts = []
        for folder_name in sorted(os.listdir(aihub_folder)):
            label_folder = os.path.join(aihub_folder, folder_name)
            new_data_list = self.getDataList(label_folder)
            data_list.extend(new_data_list)

            txt = f"{label_folder} is cleared. {len(new_data_list)} items created"
            txts.append(txt)
        return data_list, txts
            
    
    def getDataList(self, label_folder):
        """
        라벨 폴더를 입력하면 해당 폴더안의 json파일을 딕셔너리의 Data로 받아서 리스트에 쌓음
        """
        data_list = []

        folder = label_folder.split("/")[-1]
        for json_file in sorted(os.listdir(label_folder)):
            if json_file[-4:] != "json":
                continue
            json_path = os.path.join(label_folder, json_file)
            data = self.getDataFromJson(json_path)
            data['folder'] = folder
            data['video_path'] = "./source_" + folder.split("_")[-1] + "/" + data['videoname']
            data_list.append(data)

        return data_list


    def getDataFromJson(self, json_file):
        """
        json파일 한개를 읽어서 data 반환
        """
        with open(json_file) as f:
            vid_data = json.load(f)
        
        meta = vid_data['metadata']
        data = {
            "species": meta['species'],
            "breed": meta['animal']['breed'],
            "action": meta['action'],
            "height": meta['height'],
            "width": meta['width'],
            "seq": meta['seq'],
            "location": meta['location'],
            "duration": meta['duration'],
            "gender": meta['animal']['gender'],
            "age": meta['animal']['age'],
            "neuter": meta['animal']['neuter'],
            "own_pain": meta['owner']['pain'],
            "own_disease": meta['owner']['disease'],
            "own_emotion": meta['owner']['emotion'],
            "own_situation": meta['owner']['situation'],
            "own_animalCount": meta['owner']['animalCount'],
            "isp_action": meta['inspect']['action'],
            "isp_painDisease": meta['inspect']['painDisease'],
            "isp_abnormalAction": meta['inspect']['abnormalAction'],
            "isp_emotion": meta['inspect']['emotion'],
            "frames": len(vid_data['annotations']),
            "videoname": json_file.split("/")[-1].split(".json")[0],
            "labelname": json_file.split("/")[-1],
        }
        data = collections.OrderedDict(data)

        return data
    

    def saveAsCsv(self, data_list, output_path, txts=None):
        filename = os.path.join(output_path, "aihub_meta.csv")
        fieldnames = data_list[0].keys()
        
        with open(filename, "w", encoding="euc-kr") as f:
            csv_writer = csv.DictWriter(f, fieldnames)
            csv_writer.writeheader()
            for data in data_list:
                csv_writer.writerow(data)
        
        if txts:
            txt_name = os.path.join(output_path, "alert.txt")
            with open(txt_name, "w") as f:
                for txt in txts:
                    f.write(txt)
                    f.write("\n")


if __name__ == "__main__":
    # label_folder랑 out_folder만 수정 후 이용
    label_folder = "/Users/song-yunsang/Desktop/Business/Butler/Dataset/test/aihub/aihub_label/labels/"
    out_folder = "/Users/song-yunsang/Desktop/Business/Butler/Dataset/test/aihub/csv/"

    ATC = AIHubToCSV()
    data, txts = ATC.iterLabelFolder(label_folder)
    ATC.saveAsCsv(data, out_folder, txts)

