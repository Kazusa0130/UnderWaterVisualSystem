import os
import cv2
import numpy as np
import sys

VIDEO_PATH = 0  # Use 0 for webcam
SAVE_PATH = "./"


def count_files_in_directory(directory_path):
    """Counts the number of files in the specified directory."""
    try:
        data_count = len([name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])
        return data_count
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        exit()

def main():
    count = count_files_in_directory(SAVE_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(SAVE_PATH+f"rawdata_{count}.avi", fourcc, 20, (1280, 480), isColor=True)

    camera = cv2.VideoCapture(VIDEO_PATH)
    if sys.platform.startswith("win"):
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)
        camera.set(cv2.CAP_PROP_EXPOSURE, -6)
    elif sys.platform.startswith("linux"):
        camera.set(cv2.CAP_PROP_BUFFERSIZE,1)
        camera.set(cv2.CAP_PROP_FPS, 20)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        camera.set(cv2.CAP_PROP_EXPOSURE, 100)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    while(True):
        ret, img = camera.read()
        if not ret:
            break
        RightImage = img[:, :640,:]
        LeftImage = img[:, 640:,:]
        img = cv2.hconcat([LeftImage, RightImage])
        print(img.shape)
        cv2.imshow('IR', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(img)
    out.release()
    cv2.destroyAllWindows()
            
if __name__ == "__main__":
    main()