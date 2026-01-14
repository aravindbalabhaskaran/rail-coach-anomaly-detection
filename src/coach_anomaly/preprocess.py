import cv2
import numpy as np

def preprocess_bgr(img_bgr: np.ndarray) -> np.ndarray:
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, 5, 5, 7, 21)

    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    enh = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enh, (0, 0), 1.2)
    sharp = cv2.addWeighted(enh, 1.35, blur, -0.35, 0)
    return sharp
