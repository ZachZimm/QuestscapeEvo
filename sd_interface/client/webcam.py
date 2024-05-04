import cv2
class Webcam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def __del__(self):
        self.cap.release()
