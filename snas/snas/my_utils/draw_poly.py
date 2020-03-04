import cv2
import numpy as np
from PIL import Image


class DrawPoly():

    @classmethod
    def draw_poly(cls, img, boxes, texts):
        return Image.fromarray(cls.draw_poly_cv(np.array(img), boxes, texts))

    @staticmethod
    def draw_poly_cv(img, boxes, texts):
        draw_img = img.copy()
        for box, text in zip(boxes, texts):
            # reshaping to -1, 1, 2 is required by opencv
            pts = np.array(list(map(int, box))).reshape(-1, 1, 2)
            cv2.polylines(draw_img, [pts], True, (0, 255, 255))
            cv2.putText(draw_img,
                        text,
                        (min(pts[:, :, 0]), min(pts[:, :, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return draw_img
