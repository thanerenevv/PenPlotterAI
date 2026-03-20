import cv2


class Camera:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}")

    def capture(self):
        self.cap.grab()
        ret, frame = self.cap.retrieve()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")
        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
