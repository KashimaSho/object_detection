import torch
import torch.utils.data as data
import cv2
from voc import *

class PreprocessVOC2012(data.Dataset):
  '''
  pytorchのDatasetクラスを継承
  DataTransformでVOC2012データセットを前処理してデータを返す
  
  Return:
    preprocessed image(RGB Tensor)
    BBox and label(ndarray)
    width and height of image(int)
  '''
  def __init__(self, img_list, anno_list, phase, transform, get_bbox_label):
    self.img_list = img_list
    self.anno_list = anno_list
    self.phase = phase
    self.transform = transform
    self.get_bbox_label = get_bbox_label
    
  def __len__(self):
    '''
    イメージの数を返す
    pytorchのdataloaderでは必ず実装が必要
    '''
    return len(self.img_list)
  
  def __getitem__(self, index):
    '''
    前処理後のイメージとBBox座標とラベルの2次元配列を取得
    pytorchのdataloaderでは必ず実装が必要
    Parameter:
      index(int): 訓練または検証用イメージのインデックス
    Return:
      im(Tensor): 前処理後のイメージを格納したテンソル(3, 高さのピクセル数, 幅のピクセル数)
      bl(ndarray): BBoxとラベルの2次元配列
    '''
    
    im, bl, _, _ = self.pull_item(index)
    return im, bl
  
  def pull_item(self, index):
    '''
    Parameter:
      index(int): 訓練または検証用イメージのインデックス
    Return:
      im(Tensor): 前処理後のイメージを格納したテンソル(3, 高さのピクセル数, 幅のピクセル数)
      bl(ndarray): BBoxとラベルの2次元配列
      height(int): イメージの高さ
      width(int): イメージの幅
    '''
    img_path = self.img_list[index]
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    
    anno_file_path = self.anno_list[index]
    bbox_label = self.get_bbox_label(anno_file_path, width, height)
    
    img, boxes, labels = self.transform(img, self.phase, bbox_label[:, :4], bbox_label[:, 4])
    img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)
    boxlbl = np.hstack((boxes, np.expand_dims(labels, axis=1)))
    
    return img, boxlbl, height, width
  
