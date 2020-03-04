import cv2
import numpy as np
import math
from PIL import Image


class EastTools(object):

    @staticmethod
    def resize(img, size, method="bilinear"):
        """using cv2 resize but PIL api
        Input:
            img: PIL image
        Output:
            img: PIL image
        """
        if method == Image.BILINEAR or method == "bilinear":
            method = cv2.INTER_AREA
        elif method == Image.CUBIC or method == "cubic":
            method = cv2.INTER_CUBIC
        elif method == Image.NEAREST or method == "nearest":
            method = cv2.INTER_NEAREST
        else:
            raise RuntimeError("wrong method {}".format(method))
        if not isinstance(size, (tuple, list)):
            size = (size, size)
        if isinstance(size ,list):
            size = tuple(size)
        im = np.array(img)
        im = cv2.resize(im, size, interpolation=method)
        return Image.fromarray(im)

    @staticmethod
    def pad_img(img, stride=32):
        '''pad image to be divisible by stride, pad in the right and bottom
        '''
        w, h = img.size
        canvas_w = w
        canvas_h = h

        canvas_h = canvas_h if canvas_h % stride == 0 else math.ceil(canvas_h / stride) * stride
        canvas_w = canvas_w if canvas_w % stride == 0 else math.ceil(canvas_w / stride) * stride

        canvas = Image.new(img.mode, (canvas_w, canvas_h))
        canvas.paste(img, (0,0))
        return canvas

    @classmethod
    def resize_img(cls, img, stride=32):
        '''resize image to be divisible by stride(e.g 32)
        '''
        w, h = img.size
        resize_w = w
        resize_h = h

        resize_h = resize_h if resize_h % stride == 0 else int(resize_h / stride) * stride
        resize_w = resize_w if resize_w % stride == 0 else int(resize_w / stride) * stride
        img = cls.resize(img, (resize_w, resize_h), Image.BILINEAR)
        ratio_h = resize_h / h
        ratio_w = resize_w / w

        return img, ratio_h, ratio_w

    @staticmethod
    def adjust_ratio(boxes, ratio_w, ratio_h):
        '''refine boxes
        Input:
            boxes  : detected polys <numpy.ndarray, (n,9) or (n,8)>
            ratio_w: ratio of width
            ratio_h: ratio of height
        Output:
            refined boxes in float32
        '''
        if boxes is None:
            return None
        if len(boxes) == 0:
            return None
        boxes[:,[0,2,4,6]] /= ratio_w
        boxes[:,[1,3,5,7]] /= ratio_h
        return boxes

    @staticmethod
    def resize_pad(img:Image, vertices:np.array, size:tuple)->(Image, np.array):
        """
        Input:
            vertices: vertices of text region <numpy.ndarray, (n, 8,) or (n, 4, 2)>
        Output:
            vertices: vertices of text region <numpy.ndarray, (n, 8,) or (n, 4, 2)>
        """
        tw, th = size
        tw, th = int(tw), int(th)
        canvas = Image.new(img.mode, (tw,th))
        w, h = img.width, img.height
        if float(w)/tw >= float(h)/th:
            nw = tw
            scale = float(nw)/w
            nh = int(h * scale)
            #img = img.resize((nw,nh))
            img = EastTools.resize(img, (nw,nh))
            topleft = (0, 0)
            canvas.paste(img, topleft)
        else:
            nh = th
            scale = float(nh)/h
            nw = int(w * scale)
            #img = img.resize((nw,nh))
            img = EastTools.resize(img, (nw,nh))
            topleft = (0, 0)
            canvas.paste(img, topleft)

        return canvas, vertices*scale
