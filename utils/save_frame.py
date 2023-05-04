import os
import cv2
import numpy as np

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def extract_frame_from_video(video_path, video_name, save_path):
    # ================================== PREPARATION OF THE VIDEO =====================================

    # capture the video
    cap = cv2.VideoCapture(video_path + video_name)
    # print("Extracting frames from " + video_path + video_name)

    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    # Get the number of frames in the video
    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = int(cap.get(3))
    # height = int(cap.get(4))
    # print(width, height)
    # if (width > 720 and height > 480 and width > height):
    #         print("resize 720x480")
    # if (width > 480 and height > 720 and width < height):
    #         print("resize 480x720")

    # define the saving frame rate to save keypoints coordinates in a JSON file 
    # and also save the frame to later annotate them for bbox
    # here we will save 1 frame per second 
    SAVING_FRAMES_PER_SECOND = 1
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # define a variable to count the frame
    frame_num = 0
    count = 0
    # ======================================= PROCESS EACH FRAME =========================================
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            # print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break
        # if (width > 720 and height > 480 and width > height):
        #     resize = cv2.resize(img, (720, 480))
        # if (width > 480 and height > 720 and width < height):
        #     resize = cv2.resize(img, (480, 720))
        # else :
        #     
        resize = img.copy()
        # # create a circle to annonimize the patient
        # masked = resize.copy()
        # x_center = 75
        # y_center = 75
        # cv2.circle(masked, (x_center, y_center), 65, (0, 0, 0), -1)
        # trans = 0.9
        # image_new = cv2.addWeighted(masked, trans, resize, 1 - trans, 0)
        # get the duration by dividing the frame count by the FPS
        
        frame_duration = frame_num / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            os.chdir(save_path)

            count += 1

            #cv2.imwrite(video_name + "_" + str(count) + ".jpg", image_new) 
            cv2.imwrite(save_path + video_name + "_" + str(count) + ".jpg", resize)            
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass

        # increment the frame count
        frame_num += 1

        #cv2.imshow("Image", image_new)
        #cv2.imshow("Image", resize)
        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()

def main():
    # path to the folder containing the videos
    path_to_data = "/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/fasterrcnn-pytorch-training-pipeline/data_videos/full/"
    # list of the videos contained in the folder
    video_list =  os.listdir(path_to_data)

    # path where to save the extracted frames
    save_path = "/Users/marie-alix/Documents/PDM/MoveUP/Pose3D/fasterrcnn-pytorch-training-pipeline/data_videos/images_full/"
    number_of_videos = len(video_list)
    count = 1
    # for each video, extract 1 frame per second
    for video_name in video_list:
        if "analysisVideo" in video_name:
            if count > 30:
                extract_frame_from_video(video_path=path_to_data, video_name=video_name, save_path=save_path)
                print(str(count) + "/ " + str(number_of_videos))
            count += 1


if __name__ == "__main__":
    main()