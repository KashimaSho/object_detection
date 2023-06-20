from typing import Any
import cv2
import numpy as np
from numpy import random

class Compose(object):
    '''
    データの拡張を行うクラス
    Args:
        transforms (List[Transform]): 変換処理のリスト
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
            return img, boxes, labels

class ConvertFromInts(object):
    '''
    変換処理その1
    ピクセルデータをintからfloat32に変換するクラス
    '''
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    '''
    変換処理その2
    アノテーションデータを正規化状態から元に戻すクラス
    '''
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        return image, boxes, labels

class RandomBrightness(object):
    '''
    変換処理その3
    輝度をランダムに変化させるクラス
    '''
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

class RandomContrast(object):
    '''
    変換処理その4
    コントラストをランダムに変化させるクラス
    '''
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper is needed to be >= lower'
        assert self.lower >= 0, 'contrast lower is needed to be non-negative'
    
    def __call__(self, image, boxes=None, labels=None):
        #image must be float
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class ConvertColor(object):
    '''
    変換処理その5
    RGBとHSVを相互変換するクラス
    '''
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current
    
    def __call__(self, image, boxes=None, labels=None):
        if self.current=='BGR' and self.transform=='HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current=='HSV' and self.transform=='BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

class RandomSaturation(object):
    '''
    変換処理その6
    彩度をランダムに変化させるクラス
    '''
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, 'contrast upper is needed to be >= lower'
        assert self.lower >= 0, 'contrast lower is needed to be non-negative'
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            #HSV色空間はH(色相), S(彩度), V(明度)
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels

class RandomHue(object):
    '''
    変換処理その7
    色相をランダムに変化させるクラス
    '''
    def __init__(self, delta=18.0):
        assert delta>=0.0 and delta<=360.0
        self.delta = delta
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class RandomLightingNoise(object):
    '''
    変換処理その8
    測光に歪みを加えるクラス
    '''
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 1))
    
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap) #defined below
            image = shuffle(image)
        return image, boxes, labels

class SwapChannels(object):
    '''
    変換処理その9
    色チャネルの並び順を変えるクラス

    Args:
        swaps (int triple): final order of channels. eg.(2, 1, 0)
    '''
    def __init__(self, swaps):
        self.swaps = swaps
    
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

class PhotometricDistort(object):
    '''
    変換処理その10
    輝度，彩度，色相，コントラストを変化させ，歪みを加えるクラス
    '''
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(current='BGR', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]

        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1]) #compose excepting last component
        else:
            distort = Compose(self.pd[1:]) #compose excepting first component
        
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)
    
class Expand(object):
    '''
    イメージをランダムに拡大するクラス
    キャンバスを拡大し, ランダムなtopとleftを選択する
    色味の平均値で塗りつぶす
    元の画像を拡大したキャンバスの上に配置
    '''
    def __init__(self, mean):
        self.mean = mean
        
    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels
        
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio-width)
        top = random.uniform(0, height*ratio-height)
        #create canvas
        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype
        )
        #fill with mean of image
        expand_image[:, :, :] = self.mean
        #put image at left-top
        expand_image[int(top):int(top+height), int(left):int(left+width)] = image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top)) #xmin, ymin
        boxes[:, 2:] += (int(left), int(top)) #xmax, ymax

        return image, boxes, labels

class RandomMirror(object):
    '''
    イメージの左右をランダムに反転するクラス
    '''
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class ToPercentCoodrs(object):
    '''
    アノテーションデータを正規化するクラス
    '''
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 1] /= height
        boxes[:, 2] /= width
        boxes[:, 3] /= height
        return image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size
    
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
    
    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels
