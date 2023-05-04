import os
import xml.etree.ElementTree as ET


def main():
    path_to_data = "/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/fasterrcnn-pytorch-training-pipeline/data/train/"
    frame_list =  os.listdir(path_to_data)

    for frame_name in frame_list:
        if '.xml' in frame_name:
            print(path_to_data + frame_name)
            tree = ET.parse(path_to_data + frame_name)
            root = tree.getroot()
            for object in root.iter('name'):
                name= object.text
                object.text = 'lower_body'
                object.set('changed', 'yes')
                new_name= object.text
                print(name, new_name)
            tree.write(path_to_data + frame_name)

if __name__ == "__main__":
    main()