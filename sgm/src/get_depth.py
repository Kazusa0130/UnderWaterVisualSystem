from module import sgm
import cv2

VIDEO_PATH = 0
WIDTH = 640
HEIGHT = 480
CONFIG_PATH = "../config/stereo_camera_npu/camera_parameters.yaml"

def main():
    sgm = sgm.SGBMWrapper()
    sgm.init(CONFIG_PATH, WIDTH, HEIGHT)
    cap = cv2.VideoCapture(VIDEO_PATH)

    while(True):
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read video frame, may have reached the end of the video.")
            break

        left = cv2.cvtColor(frame[:, 0:640], cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(frame[:, 640:1280], cv2.COLOR_BGR2GRAY)

        disp = sgm.compute(left, right)
        
        disp[disp < 0] = 0
        disp[disp > 1000] = 1000
        dis_color = cv2.UMat(disp)
        dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dis_color = cv2.applyColorMap(dis_color, cv2.COLORMAP_TURBO)