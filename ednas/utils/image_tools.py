from PIL import Image
import cv2
import numpy as np


class ImageTools():

    @staticmethod
    def resize_pad(img:Image, size:tuple)->Image:
        tw, th = size
        tw, th = int(tw), int(th)
        canvas = Image.new(img.mode, (tw, th))
        w, h = img.width, img.height
        if float(w)/tw >= float(h)/th:
            nw = tw
            scale = float(nw)/w
            nh = int(h * scale)
            img = self.resize(img, (nw,nh))
            topleft = (0, (th-nh)//2)
            canvas.paste(img, topleft)
        else:
            nh = th
            scale = float(nh)/h
            nw = int(w * scale)
            img = self.resize(img, (nw,nh))
            topleft = ((tw-nw)//2, 0)
            canvas.paste(img, topleft)
        return canvas

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
