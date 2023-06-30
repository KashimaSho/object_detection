import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from voc import DataTransform

class SSDPredictions():
  '''
  SSDで物体検出を行う
  Attribute:
    eval_categories(list): クラス名(str)
    net(object): SSDモデル
    transform(object): 前処理クラス
  '''
  def __init__(self, eval_categories, net):
    self.eval_categories = eval_categories
    self.net = net
    color_mean = (104, 117, 123)
    input_size = 300
    self.transform = DataTransform(input_size, color_mean)
  
  def ssd_predict(self, image_file_path, confidence_threshold=0.5):
    '''
    SSDで物体検出を行い、確信度の高いBBoxの情報を返す
    Parameter:
      image_file_path(str): 画像のファイルパス
      confidence_threshold(float): 確信度の閾値
    Return:
      rgb_img: 画像のRGB値
      predict_bbox: 物体検出したBBoxの情報
      pre_dict_label_index: 物体検出したBBoxが予測する正解ラベル
      scores: 各BBoxごとの確信度
    '''
    img = cv2.imread(image_file_path)
    height, width, channels = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    phase = 'val'
    img_transformed, boxes, labels = self.transform(
      img,
      phase,
      '',
      ''
    )
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

    self.net.eval()
    x = img.unsqueeze(0)
    detections = self.net(x)
    
    predict_bbox = []
    pre_dict_label_index = []
    scores = []
    detections = detections.cpu().detach().numpy()
    find_index = np.where(detections[:, 0:, :, 0] >= confidence_threshold)
    detections = detections[find_index]
    
    for i in range(len(find_index[1])):
      if (find_index[1][i]) > 0:
        sc = detections[i][0]
        bbox = detections[i][1:] * [width, height, width, height]
        label_ind = find_index[1][i]-1
        predict_bbox.append(bbox)
        pre_dict_label_index.append(label_ind)
        scores.append(sc)
    
    return rgb_img, predict_bbox, pre_dict_label_index, scores
  
  
  def draw(self, rgb_img, bbox, label_index, scores, label_names):
    '''
    物体検出の予測結果を写真上に描画
    Parameter:
      rgb_img: 画像のRGB値
      bbox(list): 物体検出したBBoxのリスト
      scores(list): 物体の確信度
      label_names(list): ラベル名の配列
    '''
    num_classes = len(label_names)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_img)
    currentAxis = plt.gca()
    
    for i, bb in enumerate(bbox):
      label_name = label_names[label_index[i]]
      color = colors[label_index[i]]
      
      if scores is not None:
        sc = scores[i]
        display_txt = '{}: {:.2f}'.format(label_name, sc)
      else:
        display_txt = '{}: ans'.format(label_name)
      
      xy = (bb[0], bb[1])
      width = bb[2] - bb[0]
      height = bb[3] - bb[1]
      
      currentAxis.add_patch(plt.Rectangle(
        xy,
        width,
        height,
        fill=False,
        edgecolor=color,
        linewidth=2
      ))
      
      currentAxis.text(
        xy[0],
        xy[1],
        display_txt,
        bbox={'facecolor': color, 'alpha': 0.5}
      )
  
   
  def show(self, image_file_path, confidence_threshold):
    '''
    検出結果の出力
    Parameter:
      image_file_path(str): 画像のファイルパス
      confidence_threshold(float): 確信度の閾値
    '''
    rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(image_file_path, confidence_threshold)
    self.draw(
      rgb_img,
      bbox=predict_bbox,
      label_index=pre_dict_label_index,
      scores=scores,
      label_names=self.eval_categories
    )