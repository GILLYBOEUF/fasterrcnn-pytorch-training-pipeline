import json
import os
import xml.etree.ElementTree as ET
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])
        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)        

        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']
            
            # All objects are lower body
            bboxes_labels_original = ['lower body' for _ in bboxes_original]
            CLASS_LABEL = [
                    'left_hip',
                    'right_hip',
                    'left_knee',
                    'right_knee',
                    'left_ankle',
                    'right_ankle',
                ]

            CLASS_SIDE = [
                    'left',
                    'right',
                    'left',
                    'right',
                    'left',
                    'right'
                ]            

        if self.transform:   
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened, keypoints_labels=CLASS_LABEL, keypoints_sides=CLASS_SIDE)
            img = transformed['image']
            bboxes = transformed['bboxes']

            
            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
  
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,6, 2)).tolist()
            keypoints_transformed_labels = transformed['keypoints_labels']
            keypoints_transformed_sides = transformed['keypoints_sides']

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                #print(obj)
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)

            
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are lower bodies
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are lower bodies
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)

VIDEO_LIST = ["analysisVideo_DBggQ6XBeqyDrmNTf_ao6i9SQa6fYnNQfh2",
              "analysisVideo_ewcy6SDquEptdegDH_2mrHd9s4qun78LQQ7",
              "analysisVideo_HjETZBdbizhopSTgo_E6L28Pime9LEbrBhM",
              "analysisVideo_Sr4z8yTWABoHHRhMw_wdPjNuW4jibhrE72h",
              "analysisVideo_TypMY2AYcufjkH6bH_r37rGjZvLGiTnEZdi",
              "analysisVideo_WwWnyqjkF6TKE9JjL_CmuSkt85F5APAKBGJ",
              "analysisVideo_YnyaWiTbcLZzq6FZY_oAHy537pnMMsjqimy"]

LICENSE = 1
DATE_CAPTURED = "null"

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def get_coco_json(img, ann, save_path):
    inf = {
        'year':'2023',
        'version':'1',
        'description':'annotation of the human lower body (both bounding box and keypoints)',
        'contributor':'Marie-Alix Gillyboeuf',
        'url':'',
        'date_created':'2023-04-25'}
    lic = {'id': LICENSE,
           'url': '',
           'name': ''}
    cat = {
                'id':0,
                'name':'lower_body',
                'supercategory':'human',
                'keypoints':['left_hip','right_hip',
                              'left_knee','right_knee',
                              'left_ankle','right_ankle'],
                'skeleton': [[1,2],
                            [1,3], 
                            [2,4], 
                            [3,5],
                            [4,6]]
        }

    data_joined = {
        'info':inf,
        'licenses': lic,
        'categories': [cat],
        'images':img,
        'annotations':ann
    }
    json_filename =  save_path 
    with open(json_filename, 'w') as fp:
        json.dump(data_joined, fp)


def get_coco_images_from_obj(file_bboxes, file_name, id):
    # get size of the image
    tree = ET.parse(file_bboxes)
    obj = tree.getroot()
    size = obj.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    img = {
        'id':id,
        'license':LICENSE,
        'file_name':file_name,
        'height':height,
        'width':width,
        'date_captured':DATE_CAPTURED
    }

    return img

def get_coco_annotation_from_obj(file_bboxes, file_kpts, id):
    tree = ET.parse(file_bboxes)
    root = tree.getroot()
    for bndbox in root.iter("bndbox"):
        xmin = int(bndbox.findtext('xmin')) - 1
        ymin = int(bndbox.findtext('ymin')) - 1
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))
        assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin

    # get kpts
    with open(file_kpts, "r") as fk :
        kpts = json.load(fk)
        kpts = kpts["keypoints"]
        data_kpts = []
        for kpt in kpts:
            for k in kpt:
             data_kpts.append(k)

    ann = {
        'id':id,
        'image_id':id,
        'category_id':0,
        'area':o_width * o_height,
        'bbox':[xmin, ymin, o_width, o_height],
        "keypoints":data_kpts,
        "num_keypoints":6,
        'iscrowd':0,
        'segmentation':[]  # This script is not for segmentation
    }
    return ann

def merge_bboxes_kpts (file_bboxes, file_kpts, save_path):
    tree = ET.parse(file_bboxes)
    root = tree.getroot()
    width = int(root.findtext('width'))
    height = int(root.findtext('height'))
    for bndbox in root.iter("bndbox"):
        xmin = int(bndbox.findtext('xmin')) - 1
        ymin = int(bndbox.findtext('ymin')) - 1
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))
        # Resize bbox if xmax or ymax are out of the image
        if xmax > width:
            xmax = width
        if ymax > height:
            ymax = height
        assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        data_bboxes = [xmin, ymin, xmax, ymax]
       
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
    path_to_data = "/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/keypoint_rcnn_training_pytorch/dataset/pascal_voc_lower_body/"   
    video_list =  VIDEO_LIST
    img_list = []
    ann_list = []
    count = 0
    for video_name in video_list:
        if 'analysisVideo' in video_name:
            key_list = os.listdir(path_to_data + video_name + '/' + 'keypoints/')
            for key_name in key_list:
                key_name = key_name.replace(".json", "")

                if 'analysisVideo' in key_name:
                    path_file_bboxes = glob.glob(path_to_data + video_name + '/' + 'bboxes/' + key_name + '_' + "*.xml")[0]
                    path_file_kpts = path_to_data + video_name + '/' + 'keypoints/' + key_name + ".json"
                    json_path = path_to_data + video_name + '/' + 'annotations/'
                    if not os.path.exists(json_path):
                        os.makedirs(json_path)
                    os.chdir(json_path)
                    save_path = json_path + key_name + ".json"
                    merge_bboxes_kpts (path_file_bboxes, path_file_kpts, save_path)

                    # # save in coco format
                    file_name = key_name + '.jpg'
                    img = get_coco_images_from_obj(path_file_bboxes, file_name, count)
                    ann = get_coco_annotation_from_obj(path_file_bboxes, path_file_kpts, count)
                    img_list.append(img)
                    ann_list.append(ann)
                    # save_path_coco = json_coco_path + key_name
                    # merge_bboxes_kpts_coco (path_file_bboxes, path_file_kpts, save_path_coco)
                    count += 1

    save_coco_annotations = '/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/keypoint_rcnn_training_pytorch/dataset/coco_lower_body/coco.json'
    get_coco_json(img = img_list, ann = ann_list, save_path = save_coco_annotations)


if __name__ == "__main__":
    main()