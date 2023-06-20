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
        boxes[:, 2] *= width
        boxes[:, 1] *= height
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
    BGRとHSVを相互変換するクラス
    '''
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current
    
    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
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
                      (2, 0, 1), (2, 1, 0))
    
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
            ConvertColor(transform='HSV'),
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
            dtype=image.dtype)
        #fill with mean of image
        expand_image[:, :, :] = self.mean
        #put image at left-top
        expand_image[int(top):int(top + height), 
                     int(left):int(left + width)] = image
        image = expand_image

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

class ToPercentCoords(object):
    '''
    アノテーションデータを正規化するクラス
    '''
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size
    
    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image,
                           (self.size, self.size))
        return image, boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
    
    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


'''
2セットのBBoxのIoUを計算する
Args:
    box_a: Multiple BBox, Shape: [num_boxes, 4]
    box_b: Single BBox, Shape: [4]
Return:
    jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
'''
def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

class RandomSampleCrop(object):
    '''
    イメージの特定の領域をランダムに切り出すクラス
    Args:
        img(Image): トレーニング中に入力されるイメージ
        boxes(Tensor): オリジナルのBBox
        labels(Tensor): BBoxのラベル
        mode(float tuple): IoU
    Return:
        (img, boxes, labels)
    '''
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )
        
        self.sample_options = np.array(self.sample_options, dtype=object)
    
    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou =float('inf')
            
            for _ in range(50):
                current_image = image
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2:
                    continue
                
                left = random.uniform(width - w)
                top = random.uniform(height - h)
                
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                overlap = jaccard_numpy(boxes, rect)
                
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], 
                                              :]
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2
                
                if not mask.any():
                    continue
                
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                
                return current_image, current_boxes, current_labels