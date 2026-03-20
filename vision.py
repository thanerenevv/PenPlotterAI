import cv2
import numpy as np


class VisionProcessor:
    def __init__(self, canny_low=30, canny_high=100, min_contour_area=80, simplify_epsilon=1.5):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area
        self.simplify_epsilon = simplify_epsilon
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_face_roi(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(min(w, h) * 0.35)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        return frame[y1:y2, x1:x2], (x1, y1)

    def extract_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    def edges_to_contours(self, edges):
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for c in contours:
            if cv2.contourArea(c) < self.min_contour_area:
                continue
            simplified = cv2.approxPolyDP(c, self.simplify_epsilon, closed=False)
            if len(simplified) >= 2:
                result.append(simplified)
        return result

    def process(self, frame):
        result = self.detect_face_roi(frame)
        if result is not None:
            roi, offset = result
            edges = self.extract_edges(roi)
            contours = self.edges_to_contours(edges)
            shifted = []
            for c in contours:
                shifted_c = c.copy()
                shifted_c[:, 0, 0] += offset[0]
                shifted_c[:, 0, 1] += offset[1]
                shifted.append(shifted_c)
            return shifted, True
        edges = self.extract_edges(frame)
        contours = self.edges_to_contours(edges)
        return contours, False

    def render_preview(self, frame, contours, found_face):
        preview = frame.copy()
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 1)
        label = "Face detected" if found_face else "Full frame mode"
        color = (0, 255, 0) if found_face else (0, 200, 255)
        cv2.putText(preview, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            preview,
            f"Contours: {len(contours)}",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        return preview
