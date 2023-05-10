import cv2
import json
import os


def ckeck_bbox_size(path_to_image, path_to_annotations):
    img = cv2.imread(path_to_image)
    height, width, _ = img.shape
    new_annotations = {}
    with open(path_to_annotations, 'r') as f:
        annotations = json.load(f)
        if (len(annotations['bboxes'])) == 4:
            bboxes = annotations['bboxes']
        else :
            bboxes = annotations['bboxes'][0]
        if bboxes[2] > width:
            print(width)
            print("resize xmax")
            bboxes[2] = width
        if bboxes[3] > height:
            print("resize ymax")
            bboxes[3] = height
        new_annotations['bboxes'] = [bboxes]
        if (len(annotations["keypoints"])) == 6:
            print("new")
            new_annotations["keypoints"] = [[annotations["keypoints"][0], annotations["keypoints"][1], annotations["keypoints"][2], annotations["keypoints"][3], annotations["keypoints"][4], annotations["keypoints"][5]]]
        else:
            new_annotations["keypoints"] = annotations["keypoints"]
        # save new annotations
        json_filename = path_to_annotations
        with open(json_filename, 'w') as fp:
            json.dump(new_annotations, fp)


def main():
    # Path to the directory containing the annotations
    path_to_annotations = 'data_kpts/test/annotations'
    # Path to the directory containing the images
    path_to_images = 'data_kpts/test/images'

    # Loop through each annotation file and check the bounding box
    for annotations_file in os.listdir(path_to_annotations):
        if ".DS_Store" in annotations_file:
            continue
        # Get the image name
        image_name = annotations_file.replace(".json", ".jpg")
        # Get the image path
        image_path = path_to_images + '/' + image_name
        # Check the bounding box
        ckeck_bbox_size(image_path, path_to_annotations + '/' + annotations_file)

if __name__ == "__main__":
    main()