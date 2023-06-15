# 1.訓練、検証のイメージとアノテーションのファイルパスのリストを作成する関数

import os.path as osp
from typing import Any

def make_filepath_list(rootpath):
  
  '''
  データのパスを格納したリストを作成
  Parameters: 
    rootpath(str): データフォルダのルートパス
  Returns:
    train_img_list: 訓練用イメージリスト
    train_anno_list: 訓練用アノテーションリスト
    val_img_list: 検証用イメージリスト
    val_anno_list: 検証用アノテーションリスト
  '''
  
  imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
  annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')
  
  train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
  val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')
  
  train_img_list = list()
  train_anno_list = list()
  
  for line in open(train_id_names):
    file_id = line.strip() #空白スペースと改行の除去
    img_path = (imgpath_template % file_id) #%sをfile_idに置き換え
    anno_path = (annopath_template % file_id)
    train_img_list.append(img_path)
    train_anno_list.append(anno_path)
    
  val_img_list = list()
  val_anno_list = list()
  
  for line in open(val_id_names):
    file_id = line.strip() #空白スペースと改行の除去
    img_path = (imgpath_template % file_id) #%sをfile_idに置き換え
    anno_path = (annopath_template % file_id)
    val_img_list.append(img_path)
    val_anno_list.append(anno_path)
    
    
  return train_img_list, train_anno_list, val_img_list, val_anno_list



#2.BBoxの座標と正解ラベルをリスト化するクラス

import xml.etree.ElementTree as ElementTree
import numpy as np

class GetBBoxAndLabel(object):

  '''
  xmlファイルにはイメージサイズ、オブジェクト名(複数)、BBox、難易度などが含まれる
  イメージは左上(1,1)を原点として、BBoxは左上(xmin, ymin)と右下(xmax, ymax)の2点で表される
  そのため原点を(0,0)に修正する必要がある
  '''
  def __init__(self, classes):
    self.classes = classes
  
  def __call__(self, xml_path, width, height):
    '''
    Parameters:
      xml_path(str): xmlファイルのパス
      width(int): イメージの幅
      height(int): イメージの高さ
    Returns(ndarray):
      [[xmin, ymin, xmax, ymax, label_idx], ...]
    '''
    annotation = []
    xml = ElementTree.parse(xml_path).getroot()
    for obj in xml.iter('object'):
      difficult = int(obj.find('difficult').text)
      if difficult == 1:
        continue
      bndbox = []
      
      #xmlファイルの<name>要素を小文字&空白削除
      name = obj.find('name').text.lower().strip() 
      bbox = obj.find('bndbox')
      
      grid = ['xmin', 'ymin', 'xmax', 'ymax']
      for gr in grid:
        axis_value = int(bbox.find(gr).text) - 1
        if gr == 'xmin' or gr == 'xmax':
          axis_value /= width
        else:
          axis_value /= height
        bndbox.append(axis_value)
      
      #list.index(name)でリスト内のnameのインデックスを返す
      label_index = self.classes.index(name)
      bndbox.append(label_index)
      
      annotation += [bndbox]
    return np.array(annotation)