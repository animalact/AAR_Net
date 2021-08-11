# -*- coding: utf-8 -*-
import os
import json

from os.path import join as joinpath
from os.path import exists as isExisted
from os.path import abspath as abspath

from aihub_config import KEY_NAME, SKELETON, SPECIES, INFO, LICENSES

class AIHubConveter():

    def __init__(self):
        self.cat = None

    # todo : translate to eng
    def seperateBySpecies(self, source_folder, output_folder, count=True):
        
        annos = {}
        folder_name = source_folder.split("/")[-1]
        file_out = joinpath(output_folder, f'{folder_name}_species.json')
        if isExisted(file_out):
            print("Already File Exists")
            return file_out, output_folder
        
        for anno in sorted(os.listdir(source_folder)):
            filename = joinpath(source_folder, anno)
            with open(filename, "r") as f:
                info = json.load(f)
                species = info['metadata']['species']
                breed = info['metadata']['animal']['breed']
                breed = SPECIES[breed]

                breed_dict = annos.get(species, {})
                filename_list = breed_dict.get(breed, [])
                filename_list.append(joinpath(folder_name, anno))
                breed_dict[breed] = filename_list
                annos[species] = breed_dict


        with open(file_out, "w", encoding="utf-8") as f:
            json.dump(annos, f)

        if count == True:
            self.seperateByCounts(file_out, output_folder)
        
        return file_out, output_folder

    def seperateByCounts(self, jsonf, output_folder):
        """
        usage   -> after creating seperateBySpecies, use the json file with this param
        """

        with open(jsonf, "r") as f:
            info = json.load(f)
        cell = {}
        total = 0
        for spec in info:
            kind = {}
            for breed in info[spec]:
                count = len(info[spec][breed])
                total+=count
                kind[breed] = count
            cell[spec] = kind
            kind = sorted(kind.items(), key=lambda x:x[1])
        cell['total'] = total

        filename = joinpath(output_folder, jsonf.split("/")[-1].split(".")[0]+"_count.json")
        with open(filename, "w") as f:
            json.dump(cell, f)
        
        return filename
    
    def createCategory(self, species_count_folder, output_folder):
        """
        read all label_species.json which created by self.seperateByCounts
        :param return list
        [
            {
                supercategory   : CAT or DOG,
                id              : 0,
                name            : breed (ex. maltiz)
                keypoints       : ["nose", ...]
                skeleton        : [[0,2], ...]
            }, ...
        ]
        """
        check = []
        categories = []
        cat_id = 0
        cat_dict = {'breed':{"CAT":{},"DOG":{}}, 'id':{}}

        cat_main_file = joinpath(output_folder, f"aihub_categories.json")
        cat_id_file = joinpath(output_folder, f"aihub_category_index.json")

        if isExisted(cat_main_file) and isExisted(cat_id_file):
            print("Already File Exists")
            return cat_main_file, cat_id_file

        if not isExisted(species_count_folder):
            raise FileNotFoundError
        for species_file in os.listdir(species_count_folder):
            if not "count" in species_file:
                continue

            with open(joinpath(species_count_folder, species_file), "r") as f:
                data = json.load(f)
                for supercategory in data:
                    if supercategory == "total":
                        continue
                    breeds = data[supercategory].keys()
                    for breed in breeds:
                        if not breed in check:
                            category = {}
                            category['supercategory'] = supercategory
                            category['id'] = cat_id
                            category['name'] = breed
                            category['keypoints'] = KEY_NAME
                            category['skeleton'] = SKELETON
                            categories.append(category)

                            cat_dict['breed'][supercategory][breed] = cat_id
                            cat_dict['id'][cat_id] = {"supercategory":supercategory, 'breed' : breed}

                            check.append(breed)
                            cat_id += 1

        with open(cat_main_file, "w") as f:
            json.dump(categories, f)
        with open(cat_id_file, "w") as f:
            json.dump(cat_dict, f)

        return cat_main_file, cat_id_file


    # Important
    def convertToCOCO(self, label_folder, output_folder, resize=None, cat_file=None, config_file=None, only=False):
        """ 
        :param label_folder : ./path/Dataset/label_x
        :param category_file : ./path/Dataset/aihub_categories.json
        :param output_folder : ./path/Dataset/
        :param resize        : (x, y)
        :param config        : ./path/Dataset/config.json               # 중복 방지 확인 및 기록
        :param only          : ['maltese', ...]
        :return 
            {
                info:           {'description': 'AI Hub Dataset', 'url': 'https://aihub.or.kr/aidata/34146', 'version': '1.0', 'year': 2020}
                licenses:       [{'url': 'http://lifelibrary.synology.me', 'id': 0, 'name': 'Attribution-NonCommercial-ShareAlike License'}]
                images:         [{license: int, file_name: str, height: int, width: int, id: int,}, ...]
                annotations:    [{segmentation: [], num_keypoints: int, area: int, iscrowd: int, keypoints: [15x3], image_id: int, bbox: [4x1], category_id: int, id: int}]
                categories:     
                videos:         
            }
        """
        # config contained current img_id, vid_id, process
        if config_file:
            img_id, vid_id, process = self._loadConfig(config_file)
            if label_folder.split("/")[-1] in process:
                print("already existed on config.json")
                return None, config_file
        else:
            config_file = joinpath(output_folder, "config.json")
            img_id, vid_id, process = 0, 0, []

        # load default setting
        info_coco = INFO
        lic_coco = LICENSES
        if cat_file:
            cat_coco = self._loadCategoryFile(cat_file)
        else:
            cat_coco = self._createCategoryFile()
        
        if not self.cat:
            raise FileNotFoundError


        
        coco = {}
        coco_id = label_folder.split("/")[-1][-1]

        images = []
        annotations = []
        videos = []
        i = 0
        # iterate json file from folder 
        for anno in sorted(os.listdir(label_folder)):
            filename = joinpath(label_folder, anno)
            # video_name : /source_x/vid.mp4
            video_name = self._getVideoName(filename)                                   
            video = {"vid_name": video_name, 'images': [], 'id': vid_id}
            with open(filename, "r") as f:  
                # :param data   ->  keys : [file_videos, metadata, annotations]
                data = json.load(f)
                meta = data['metadata']

                # if resize, add this on meta
                meta['resize'] = resize

                # This is for select dog or cat 's breed
                if only:
                    breed = meta['animal']['breed']
                    if not SPECIES[breed] in only:
                        continue
                for frame in data['annotations']:
                    img_coco = self.createCOCOImageFormat(img_id, vid_id, filename, frame, meta)
                    anno_coco = self.createCOCOAnnotationFormat(img_id, frame, meta)

                    images.append(img_coco)
                    annotations.append(anno_coco)
                    video['images'].append(img_coco['id'])

                    img_id += 1
                    
            videos.append(video)
            vid_id += 1

        coco['info'] = info_coco
        coco['licenses'] = lic_coco
        coco['images'] = images
        coco['annotations'] = annotations
        coco['categories'] = cat_coco
        coco['videos'] = videos

        output_file = joinpath(output_folder, f"Aihub_COCO_{coco_id}.json")
        with open(output_file, "w") as f:
            json.dump(coco, f)

        self._updateConfig(config_file, label_folder, img_id, vid_id, process)

        return coco, config_file

    def _createCategoryFile(self):
        category = [{
            "supercategory": "Animal",
            "id": 0,
            "name": "animal",
            "keypoints": ["nose", "center_forehead", "end_mouth", "center_mouth", "neck", "front_right_shoulder", "front_left_shoulder", "front_right_ankle", "front_left_ankle", "back_right_femur", "back_left_femur", "back_right_ankle", "back_left_ankle", "tail_start", "tail_end"], 
            "skeleton": [[0, 1], [1, 2], [2, 3], [0, 4], [1, 4], [3, 4], [4, 13], [4, 5], [4, 6], [5, 7], [6, 8], [13, 9], [13, 10], [9, 11], [10, 12], [13, 14]]
        }]
        self.cat = category
        return category

    def _loadCategoryFile(self, cat_file):
        try:
            with open(cat_file, "r") as f:
                category = json.load(f)
            self.cat = category
            return category
        except:
            print("loadCategoryFile, cannot read CATEGORY FILE, plz check aihub_config.py")
            raise ValueError
    
    def createCOCOImageFormat(self, img_id, vid_id, filename, frame, meta):
        license = 0
        video_src = "source_" + filename.split(".json")[0].split("/")[-2].split("_")[-1]
        video_name = filename.split(".json")[0].split("/")[-1]
        image_name = f"frame_{frame['frame_number']}_timestamp_{frame['timestamp']}.jpg"
        filename = joinpath(video_src, video_name, image_name)

        if meta['resize']:
            height = meta['resize'][0]
            width = meta['resize'][1]
        else:
            height = meta['height']
            width = meta['width']
        id = img_id

        img_coco = {
            'license': license,
            'file_name': filename,
            'height': height,
            'width': width,
            'id':id,
            'vid_id':vid_id
        }
        return img_coco

    def createCOCOAnnotationFormat(self, img_id, frame, meta):
        segmentation = []
        num_keypoints = 15
        area = 0
        iscrowd = 0
        keypoints, num_keypoints = self._alignSkeleton(frame['keypoints'], meta)
        bbox = self._reformBbox(frame['bounding_box'], meta)
        category_id = self._findCategoryId(SPECIES[meta['animal']['breed']])
        anno_coco = {
            "segmentation": segmentation,
            "num_keypoints": num_keypoints,
            "area": area,
            "iscrowd": iscrowd,
            "keypoints": keypoints,
            "image_id": img_id,
            "bbox": bbox,
            "category_id": category_id,
            "id": img_id
        }
        return anno_coco
    
    def _loadConfig(self, config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        try:
            img_id = int(config['img_id']) + 1
            vid_id = int(config['vid_id']) + 1
            process = config['process']
        except:
            print("config file is not properly constructed")
            raise ValueError
        return img_id, vid_id, process
    
    def _updateConfig(self, config_file, source_folder, img_id, vid_id, process):
        label = source_folder.split("/")[-1]
        process.append(label)
        config = {
            'img_id': img_id,
            'vid_id': vid_id,
            'process': process
        }
        with open(config_file, "w") as f:
            json.dump(config, f)
        return

    def _getVideoName(self, filename):
        # ./path/label_x/vid.mp4.json   =>  ./path/source_x/vid.mp4
        video_src = "source_" + filename.split(".json")[0].split("/")[-2].split("_")[-1]
        video_name = filename.split(".json")[0].split("/")[-1]
        video_name = joinpath(video_src, video_name)
        return video_name

    def _alignSkeleton(self, meta_keypoints, meta):
        """
        :param return [x1, y1, v1, x2, y2, v2, ..., ]
        """
        keypoints = []
        num_keypoints = 0
        for i in range(15):
            coord = meta_keypoints[str(i+1)]
            if coord == None:
                keypoints.extend([0,0,0])
            else:
                if meta['resize']:
                    height = meta['height']
                    width = meta['width']
                    keypoint = [round(coord['x']/width*meta['resize'][0]), round(coord['y']/height*meta['resize'][1]), 2]
                else:
                    keypoint = [coord['x'], coord['y'], 2]
                keypoints.extend(keypoint)
                num_keypoints += 1

        return keypoints, num_keypoints

    def _reformBbox(self, meta_bbox, meta):
        """
        aihub bbox = [x, y, width, height] => (left_top = x,y)
        coco is same format
        """
        scale_width = meta['resize'][0] / meta['width']
        scale_height = meta['resize'][1] / meta['height']
        if meta['resize']:
            bbox = [
                round(meta_bbox['x']*scale_width),
                round(meta_bbox['y']*scale_height),
                round(meta_bbox['width']*scale_width),
                round(meta_bbox['height']*scale_height)
            ]
        else:
            bbox = [meta_bbox['x'], meta_bbox['y'], meta_bbox['width'], meta_bbox['height']]
        return bbox

    def _findCategoryId(self, breed):
        if len(self.cat) == 1:
            return 0

        for c in self.cat:
            if c['name'] == breed:
                return c['id']

        print("_findCategoryId, cannot find category id from self.cat")
        raise ValueError

if __name__ == "__main__":
    SRC_LABEL = "/home/butlely/Desktop/Dataset/aihub/"
    OUTPUT_ROOT = "/home/butlely/Desktop/Dataset/aihub/COCO/ksh_256"
    #
    # OUTPUT_SPECIES = joinpath(OUTPUT_ROOT, "species")
    # OUTPUT_CATEGORIES = joinpath(OUTPUT_ROOT, "categories")
    # OUTPUT_COCO = joinpath(OUTPUT_ROOT, "coco")
    #
    # def createFolderIfNotExists(path_list):
    #     for p in path_list:
    #         if not isExisted(p):
    #             os.mkdir(p)
    # createFolderIfNotExists([OUTPUT_SPECIES, OUTPUT_CATEGORIES, OUTPUT_COCO])
    #
    dataset = AIHubConveter()
    #
    # print("Start Seperating")
    # for i in range(9):
    #     dataset.seperateBySpecies(SRC_LABEL + f"label_{i+1}", OUTPUT_SPECIES)
    #     print(SRC_LABEL + f"label_{i+1}", 'is Done')
    #
    # print("Seperated Done")
    # print("-"*20)
    # print("Start Categorizing")
    #
    # cat, cat_id = dataset.createCategory(OUTPUT_SPECIES, OUTPUT_CATEGORIES)
    #
    # print("Categorizing Done")
    # print("-"*20)
    # print("Start Convert to COCO")

    config_file = False
    coco_total = None

    for i in range(6, 9):
        coco, config_file = dataset.convertToCOCO(SRC_LABEL + f'/label_{i+1}', OUTPUT_ROOT, resize=(256,256), only=['Korean short hair'])
        if coco_total is None:
            coco_total = coco
        else:
            coco_total['images'].extend(coco['images'])
            coco_total['annotations'].extend(coco['annotations'])
            coco_total['videos'].extend(coco['videos'])
        print(f'label_{i+1} is created')

    def splitCoco(cocototal):
        info = cocototal['info']
        lic = cocototal['licenses']
        imgs = cocototal['images']
        annos = cocototal['annotations']
        cat = cocototal['categories']

        tr = {"info":info,"licenses":lic, "images":[], "annotations":[], "categories":cat}
        te = {"info":info,"licenses":lic, "images":[], "annotations":[], "categories":cat}
        va = {"info":info,"licenses":lic, "images":[], "annotations":[], "categories":cat}
        # 7: 2: 1
        for i, img in enumerate(imgs):
            anno = annos[i]
            if i % 10 == 0:
                va['images'].append(img)
                va['annotations'].append(anno)
                continue
            if i % 10 == 1 or i % 10 == 2:
                te['images'].append(img)
                te['annotations'].append(anno)
                continue
            tr['images'].append(img)
            tr['annotations'].append(anno)
        print(len(tr['images']))
        print(len(tr['annotations']))
        print(len(te['images']))
        print(len(te['annotations']))
        print(len(va['images']))
        print(len(va['annotations']))
        return tr, te, va

    train_coco, test_coco, val_coco = splitCoco(coco_total)

    with open(joinpath(OUTPUT_ROOT,"coco_tr.json"), "w") as f:
        json.dump(train_coco, f)
    with open(joinpath(OUTPUT_ROOT,"coco_te.json"), "w") as f:
        json.dump(test_coco, f)
    with open(joinpath(OUTPUT_ROOT,"coco_va.json"), "w") as f:
        json.dump(val_coco, f)
    

