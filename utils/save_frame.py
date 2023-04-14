def main():
    source = "video.mp4"
    cap = cv2.VideoCapture('source/%s' % source)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    count = 0
    rate = round((1/fps), 3) - 0.006 # rounds to 0.05 sec
    # 1 frame = 0.05 sec -> 18 frames = 0.9 or 1 sec
    sec = 0
    # 0.5
    # 0.1
    # 0.15
    # ...

    while True:
        sec = sec + rate
        cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000) # sec*1000 converts to ms
        ret, frame = cap.read() 
        if not ret:
            cap.release()
            break
        else:
            cv2.imwrite("../frames/" + "frame%d.jpg" % count, frame)
        count += 1;