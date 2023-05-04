import json
import os
import xml.etree.ElementTree as ET
import glob

LICENSE= 1
DATE= "null"

def merge_bboxes_kpts (file_bboxes, file_kpts, save_path):
    tree = ET.parse(file_bboxes)
    root = tree.getroot()
    for object in root.iter("bndbox"):
        data_bboxes = [object.find("xmin").text, object.find("ymin").text, object.find("xmax").text, object.find("ymax").text]
        data_bboxes = [int(data_bboxes[0]), int(data_bboxes[1]), int(data_bboxes[2]), int(data_bboxes[3])]
        print(data_bboxes)
       
    with open(file_kpts, "r") as fk :
        kpts = json.load(fk)
        data_kpts = kpts["keypoints"]

    data_joined = {"bboxes" :[data_bboxes],
                   "keypoints": [data_kpts]} 

    json_filename =  save_path 
    with open(json_filename, 'w') as fp:
        json.dump(data_joined, fp)

def merge_bboxes_kpts_coco (file_bboxes, file_kpts, save_path):
    with open(file_bboxes, "r") as fbb :
        bboxes = json.load(fbb)
        data_bboxes = bboxes["annotations"][0]
        data_bboxes = [data_bboxes["bounding_box"]["x"], data_bboxes["bounding_box"]["y"], data_bboxes["bounding_box"]["w"], data_bboxes["bounding_box"]["h"]]
       
    with open(file_kpts, "r") as fk :
        kpts = json.load(fk)
        data_kpts = kpts["keypoints"]

    data_joined = {"bboxes" :[data_bboxes],
                   "keypoints": [data_kpts]} 

    json_filename =  save_path 
    with open(json_filename, 'w') as fp:
        json.dump(data_joined, fp)

def main():
    path_to_data = "/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/keypoint_rcnn_training_pytorch/dataset/"   
    video_list =  VIDEO_LIST

    count = 0
    for video_name in video_list:
        if 'analysisVideo' in video_name:
            key_list = os.listdir(path_to_data + video_name + '/' + 'keypoints/')
            for key_name in key_list:
                key_name = key_name.replace(".json", "")

                if 'analysisVideo' in key_name:
                    print(path_to_data + video_name + '/' + 'bboxes/' + key_name)
                    path_file_bboxes = glob.glob(path_to_data + video_name + '/' + 'bboxes/' + key_name + '_' + "*.xml")[0]
                    print(path_file_bboxes)
                    path_file_kpts = path_to_data + video_name + '/' + 'keypoints/' + key_name + ".json"
                    print(path_file_kpts)
                    json_path = path_to_data + video_name + '/' + 'annotations/'
                    # json_coco_path = path_to_data + video_name + '/' + 'annotations_coco/'
                    if not os.path.exists(json_path):
                        os.makedirs(json_path)
                    os.chdir(json_path)
                    save_path = json_path + key_name + ".json"
                    merge_bboxes_kpts (path_file_bboxes, path_file_kpts, save_path)
                    # # save in coco format
                    # if not os.path.exists(json_coco_path):
                    #     os.makedirs(json_coco_path)
                    # os.chdir(json_coco_path)
                    # save_path_coco = json_coco_path + key_name
                    # merge_bboxes_kpts_coco (path_file_bboxes, path_file_kpts, save_path_coco)
                    count += 1

if __name__ == "__main__":
    main()